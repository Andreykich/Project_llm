
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Startup Trends Multi‑Agent System
Architecture: Intake → (Validator-in) → Retrieval/Synthesis (ReAct+CoT, ≤6 шагов) → (Validator-out)
Constraints: single file, stdlib + requests + bs4; only external service is OpenRouter.
Your confirmed settings:
- Sources: specific URLs + landing pages with adjacent links (Habr sections/tags, pro media, blogs), ru/en
- Retro window: 180 days (MAX_DAYS_RETRO)
- Timezone for date normalization: Europe/Moscow
- Dedup: strict by url+title hash (soft duplicates allowed)
- Trend validity: ≥ 2 independent sources in the last 90 days
- Refusal policy: if no valid sources → show refusal prompt
- Model: deepseek/deepseek-v3.2-exp via OpenRouter; API key from .env
- JSON output schema is fixed by business requirement
"""

import os, json, time, hashlib, sqlite3, re, logging, datetime
from typing import List, Optional, Dict, Any, Tuple, Iterable
from contextlib import contextmanager

import requests
# --- .env autoload (stdlib-only, Windows-friendly) ---
import os

def _load_dotenv(path="config/.env"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                # Снимем обрамляющие кавычки, если есть
                if len(v) >= 2 and (v[0] == v[-1]) and v[0] in ("'", '"'):
                    v = v[1:-1]
                if k and (k not in os.environ):
                    os.environ[k] = v
    except FileNotFoundError:
        pass

# 👇 вызов функции

_load_dotenv()
print("✅ OPENROUTER_API_KEY найден:", bool(os.getenv("OPENROUTER_API_KEY")))

from bs4 import BeautifulSoup

# ----------------------------- Config / Constants -----------------------------

MODEL_NAME = "deepseek/deepseek-v3.2-exp"           # via OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # expected from .env
if not OPENROUTER_API_KEY:
    # Allow running offline for parser/DB tests; LLM calls will stub.
    logging.warning("OPENROUTER_API_KEY is not set. LLM calls will return a stub.")

DB_PATH = os.getenv("DB_PATH", "trends.db")
JSONL_PATH = os.getenv("JSONL_PATH", "corpus.jsonl")
USER_TZ = "Europe/Moscow"

MAX_DAYS_RETRO = 180
EVIDENCE_WINDOW_DAYS = 90   # ≥ 2 independent sources over this window
MAX_REACT_STEPS = 6
REQ_TIMEOUT = 30            # HTTP timeout per request
PIPELINE_HARD_TIMEOUT = 120 # total wall-clock cap (seconds)

# Generation defaults (per-agent may override)
GEN_DEFAULT = dict(max_tokens=1100, temperature=0.2, top_p=0.9)
GEN_STRICT  = dict(max_tokens=900,  temperature=0.1, top_p=0.9)
GEN_FRIENDLY= dict(max_tokens=900,  temperature=0.6, top_p=0.95)

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY or 'NO_KEY'}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Startup-Trends-Agent"
}

# ------------------------------- Utilities ------------------------------------

def now_utc_date() -> datetime.date:
    return datetime.datetime.utcnow().date()

def iso_date(d: datetime.date) -> str:
    return d.isoformat()

def normalize_lang(s: str) -> str:
    """Normalize language to 'ru' or 'en' (very light heuristic)."""
    if not s: 
        return "ru"
    s = s.lower()
    if "ru" in s or re.search(r"[А-Яа-яЁё]", s):
        return "ru"
    return "en"

def domain_of(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", url.strip())
    return m.group(1) if m else "unknown"

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@contextmanager
def sqlite(path: str):
    cx = sqlite3.connect(path)
    try:
        cx.row_factory = sqlite3.Row
        yield cx
        cx.commit()
    finally:
        cx.close()

# ------------------------------- SQLite DAO -----------------------------------

class SQLiteDAO:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._init()

    def _init(self):
        with sqlite(self.path) as cx:
            cx.execute("""
            CREATE TABLE IF NOT EXISTS docs(
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                lang TEXT NOT NULL,
                published_at TEXT NOT NULL,
                title TEXT NOT NULL,
                tags TEXT NOT NULL,
                summary TEXT,
                text TEXT NOT NULL,
                h_ud TEXT NOT NULL
            );
            """)
            cx.execute("CREATE INDEX IF NOT EXISTS idx_docs_date ON docs(published_at);")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_docs_ud   ON docs(h_ud);")

    @staticmethod
    def _hash_ud(url: str, title: str) -> str:
        return sha256(url.strip() + "||" + title.strip())

    def upsert_doc(self, d: Dict[str, Any]) -> None:
        h_ud = self._hash_ud(d["url"], d["title"])
        with sqlite(self.path) as cx:
            cx.execute("""
            INSERT INTO docs(id,url,domain,lang,published_at,title,tags,summary,text,h_ud)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                url=excluded.url, domain=excluded.domain, lang=excluded.lang,
                published_at=excluded.published_at, title=excluded.title,
                tags=excluded.tags, summary=excluded.summary, text=excluded.text, h_ud=excluded.h_ud
            """, (
                d["id"], d["url"], d["domain"], d["lang"], d["published_at"],
                d["title"], json.dumps(d.get("tags", []), ensure_ascii=False),
                d.get("summary",""), d["text"], h_ud
            ))

    def exists_ud(self, url: str, title: str) -> bool:
        h_ud = self._hash_ud(url, title)
        with sqlite(self.path) as cx:
            row = cx.execute("SELECT 1 FROM docs WHERE h_ud=? LIMIT 1;", (h_ud,)).fetchone()
            return row is not None

    def recent_docs(self, domain_filter: Optional[str]=None, days: int=90) -> List[Dict[str, Any]]:
        since = iso_date(now_utc_date() - datetime.timedelta(days=days))
        q = "SELECT * FROM docs WHERE published_at>=?"
        params = [since]
        if domain_filter:
            q += " AND (domain LIKE ? OR title LIKE ? OR text LIKE ?)"
            like = f"%{domain_filter}%"
            params += [like, like, like]
        q += " ORDER BY published_at DESC LIMIT 500;"
        with sqlite(self.path) as cx:
            rows = cx.execute(q, params).fetchall()
            return [dict(r) for r in rows]

# --------------------------------- Parser -------------------------------------

class Parser:
    UA = {"User-Agent": "Mozilla/5.0 (compatible; StartupTrendsBot/1.0)"}
    def __init__(self, dao: SQLiteDAO):
        self.dao = dao

    @staticmethod
    def _normalize_date(dt_str: str) -> str:
        # Try ISO first; otherwise fallback to today's date in USER_TZ context (date only).
        try:
            return datetime.date.fromisoformat(dt_str[:10]).isoformat()
        except Exception:
            return iso_date(now_utc_date())

    def fetch_and_store(self, url: str, lang_hint: str="ru", tags: Optional[List[str]]=None) -> bool:
        r = requests.get(url, headers=self.UA, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        title = (soup.title.text or "").strip() if soup.title else url
        # Only strict dedup (soft duplicates allowed by requirement)
        if self.dao.exists_ud(url, title):
            return False

        # Extract text
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join([t for t in ps if t])

        # Date from meta/time
        date_meta = soup.find("meta", {"property":"article:published_time"}) or \
                    soup.find("meta", {"name":"date"}) or soup.find("time")
        raw_date = ""
        if date_meta:
            raw_date = date_meta.get("content") or date_meta.get_text() or ""
        published_at = self._normalize_date(raw_date)

        lang = normalize_lang(lang_hint)
        doc = {
            "id": hashlib.md5(url.encode("utf-8")).hexdigest(),
            "url": url,
            "domain": domain_of(url),
            "lang": lang,
            "published_at": published_at,
            "title": title,
            "tags": tags or [],
            "summary": "",
            "text": text
        }
        self.dao.upsert_doc(doc)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return True

    def crawl_landing(self, landing_url: str, max_links: int = 20) -> List[str]:
        """Collect adjacent links from a landing page (same domain), capped by max_links."""
        r = requests.get(landing_url, headers=self.UA, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        dom = domain_of(landing_url)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = f"https://{dom}{href}"
            if href.startswith("http") and domain_of(href) == dom:
                links.append(href)
            if len(links) >= max_links:
                break
        return list(dict.fromkeys(links))  # unique preserve order

# ---------------------------- OpenRouter Client --------------------------------

class OpenRouterClient:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **gen) -> Dict[str, Any]:
        # Stub offline if no key
        if not OPENROUTER_API_KEY:
            return {"choices":[{"message":{"role":"assistant","content":"{ \"stub\": true }"}}]}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": gen.get("max_tokens", GEN_DEFAULT["max_tokens"]),
            "temperature": gen.get("temperature", GEN_DEFAULT["temperature"]),
            "top_p": gen.get("top_p", GEN_DEFAULT["top_p"]),
        }
        resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

# ------------------------------ ReAct Prompt ----------------------------------

REACT_SYSTEM = """You are a Retrieval/Synthesis agent following ReAct:
- Think (hidden), then one of Actions: [SEARCH_CORPUS, SUMMARIZE, FINAL_ANSWER], then Observe.
- Limit: at most 6 steps.
Tools:
  SEARCH_CORPUS: Input {"domain":"<str>", "days": <int>} → returns recent docs.
  SUMMARIZE: Input {"snippets":[...]} → derive ideas/niches/risks/notes.
  FINAL_ANSWER: Output ONLY final JSON matching the required schema.
Evidence policy: require >=2 independent sources within last 90 days, else trigger refusal policy.
"""

FINAL_SCHEMA_HINT = """Schema (strict keys):
{
  "idea": "...",
  "local_niches": [{"name":"...", "why_now":"..."}],
  "90_day_plan": [{"week":1,"tasks":[...]}, ...],
  "risks": [{"risk":"...", "mitigation":"..."}],
  "unit_economics_notes": ["..."],
  "sources": [{"url":"...", "title":"...", "date":"YYYY-MM-DD"}],
  "evidence": ["..."]
}
"""

# ------------------------------ Agents ----------------------------------------

class AgentBase:
    def __init__(self, client: OpenRouterClient):
        self.client = client

class IntakeAgent(AgentBase):
    SYSTEM = """Ты — Intake‑агент.
Верни только JSON вида: {"domain":"...", "location":"...", "details":"..."}
- domain (обязателен): отрасль/сфера ("кофейня", "EdTech", ...)
- location (опц.): город/страна, по умолчанию "Москва/Россия"
- details (опц.): дополнительные требования пользователя
Никаких комментариев. Только JSON."""
    def run(self, raw_user_text: str) -> Dict[str, Any]:
        msgs = [
            {"role":"system","content": self.SYSTEM},
            {"role":"user","content": raw_user_text}
        ]
        out = self.client.chat(msgs, **GEN_FRIENDLY)
        content = out["choices"][0]["message"]["content"]
        try:
            obj = json.loads(content)
        except Exception:
            obj = {"domain": raw_user_text.strip()[:40], "location":"Москва/Россия", "details":""}
        if not obj.get("location"):
            obj["location"] = "Москва/Россия"
        return obj

class ValidatorInAgent(AgentBase):
    SYSTEM = """Ты — Валидатор входа.
Проверь JSON UserQuery. Если нет domain — установи "general". Заполни пустой location="Москва/Россия". Верни только JSON."""
    def run(self, uq: Dict[str, Any]) -> Dict[str, Any]:
        if "domain" not in uq or not uq["domain"]:
            uq["domain"] = "general"
        if not uq.get("location"):
            uq["location"] = "Москва/Россия"
        uq.setdefault("details","")
        return uq

class RetrievalSynthesisAgent(AgentBase):
    def __init__(self, client: OpenRouterClient, dao: SQLiteDAO):
        super().__init__(client); self.dao = dao

    def _tool_search_corpus(self, domain: str, days: int=90) -> Dict[str, Any]:
        rows = self.dao.recent_docs(domain_filter=domain, days=days)
        items = []
        for r in rows[:80]:
            items.append({
                "url": r["url"], "title": r["title"], "date": r["published_at"],
                "text": (r["text"][:1200] + "...") if len(r["text"])>1200 else r["text"]
            })
        return {"results": items}

    def _has_min_evidence(self, items: List[Dict[str,Any]]) -> bool:
        # Independent = distinct domains; within EVIDENCE_WINDOW_DAYS handled by DAO query.
        domains = {domain_of(it["url"]) for it in items}
        return len(domains) >= 2

    def run(self, uq: Dict[str, Any]) -> str:
        start_ts = time.time()
        snippets, planned_domain = None, uq.get("domain","")
        # ReAct loop (light controller executed here, tools local)
        for _ in range(MAX_REACT_STEPS):
            if time.time() - start_ts > PIPELINE_HARD_TIMEOUT:
                break
            obs = f"Observation: {len(snippets['results'])} docs." if snippets else ""
            msgs = [
                {"role":"system","content": REACT_SYSTEM + "\n" + FINAL_SCHEMA_HINT},
                {"role":"user","content": f"UserQuery: {json.dumps(uq, ensure_ascii=False)}\n{obs}"}
            ]
            res = self.client.chat(msgs, **GEN_STRICT)
            content = res["choices"][0]["message"]["content"]

            act_m = re.search(r"Action:\s*([A-Z_]+)", content)
            inp_m = re.search(r"ActionInput:\s*(\{.*\}|\[.*\]|.+)$", content, re.S)
            act = act_m.group(1) if act_m else "FINAL_ANSWER"
            arg = inp_m.group(1).strip() if inp_m else "{}"

            if act == "SEARCH_CORPUS":
                try:
                    params = json.loads(arg)
                except Exception:
                    params = {"domain": planned_domain, "days": EVIDENCE_WINDOW_DAYS}
                domain_q = params.get("domain", planned_domain) or planned_domain
                days_q = int(params.get("days", EVIDENCE_WINDOW_DAYS))
                snippets = self._tool_search_corpus(domain_q, days=days_q)
                continue

            if act == "SUMMARIZE":
                # Summarization is delegated to LLM implicitly on next FINAL_ANSWER.
                continue

            # FINAL_ANSWER (or unknown) — enforce evidence/refusal policy here:
            items = snippets["results"] if snippets else []
            if not self._has_min_evidence(items):
                refusal = {
                  "idea": "",
                  "local_niches": [],
                  "90_day_plan": [],
                  "risks": [],
                  "unit_economics_notes": [],
                  "sources": [],
                  "evidence": [
                    'в базе данных нет информации по данному вопросу. хотите чтобы я все равно сказал рекомендацию на свое усмотрение? (не подкреплённое данными)'
                  ]
                }
                return json.dumps(refusal, ensure_ascii=False)

            # Ask LLM to synthesize final JSON (with strict instruction); include sources to force citations.
            synth_msgs = [
                {"role":"system","content": "Собери строго валидный JSON по заданной схеме. Укажи 'sources' из сниппетов (url,title,date). Локальные ниши — для Москвы/Россия. Краткие заметки по unit economics: CAC, LTV, GM, Payback."},
                {"role":"user","content": json.dumps({"UserQuery":uq,"snippets":items}, ensure_ascii=False)}
            ]
            fin = self.client.chat(synth_msgs, **GEN_STRICT)
            return fin["choices"][0]["message"]["content"]

        # Fallback if loop exhausted or timed out
        fallback = {
          "idea": "",
          "local_niches": [],
          "90_day_plan": [],
          "risks": [],
          "unit_economics_notes": [],
          "sources": [],
          "evidence": [
            "pipeline timeout or step limit; insufficient data to produce validated answer"
          ]
        }
        return json.dumps(fallback, ensure_ascii=False)

class ValidatorOutAgent(AgentBase):
    SYSTEM = """Ты — Выходной валидатор.
Проверь, что ответ — валидный JSON со всеми ключами схемы и датами YYYY-MM-DD. Исправляй минимально и верни JSON."""
    KEYS = ["idea","local_niches","90_day_plan","risks","unit_economics_notes","sources","evidence"]
    def run(self, final_json_text: str) -> str:
        try:
            obj = json.loads(final_json_text)
        except Exception:
            obj = {}
        for k in self.KEYS:
            obj.setdefault(k, [] if k!="idea" else "")
        # fix date format if needed
        for s in obj.get("sources", []):
            if "date" in s and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(s["date"]) or ""):
                s["date"] = iso_date(now_utc_date())
        return json.dumps(obj, ensure_ascii=False)

# ------------------------------ Orchestrator -----------------------------------

class Orchestrator:
    def __init__(self):
        self.dao = SQLiteDAO(DB_PATH)
        self.parser = Parser(self.dao)
        self.client = OpenRouterClient()
        self.intake = IntakeAgent(self.client)
        self.vin = ValidatorInAgent(self.client)
        self.rs = RetrievalSynthesisAgent(self.client, self.dao)
        self.vout = ValidatorOutAgent(self.client)

    def ingest(self, urls: List[str], lang_hint="ru", tags=None, crawl_adjacent=False, max_links=20) -> int:
        added = 0
        for u in urls:
            try:
                if crawl_adjacent:
                    links = [u] + self.parser.crawl_landing(u, max_links=max_links)
                else:
                    links = [u]
                for link in links:
                    if self.parser.fetch_and_store(link, lang_hint, tags):
                        added += 1
            except Exception as e:
                logging.warning(f"[INGEST] fail {u}: {e}")
        return added

    def run_query(self, user_text: str) -> str:
        uq = self.intake.run(user_text)
        uq = self.vin.run(uq)
        raw = self.rs.run(uq)
        out = self.vout.run(raw)
        return out

# --------------------------------- Demo ----------------------------------------

DEMO_URLS = [
    # put concrete seed URLs here (Habr sections/tags or specific posts, plus media/blogs)
    "https://habr.com/ru/companies/",
    "https://habr.com/ru/hubs/startups/"
]

def run_demo():
    logging.basicConfig(level=logging.INFO)
    orch = Orchestrator()

    # Ingest (landing + adjacent links, but still limited)
    added = orch.ingest(DEMO_URLS, lang_hint="ru", tags=["startup","trend"], crawl_adjacent=True, max_links=10)
    logging.info(f"Ingest added {added} docs.")

    # User query example
    user = "Сфера: кофейня; город: Москва; детали: подписка и бизнес-ланчи для офисных парков"
    print(orch.run_query(user))

if __name__ == "__main__":
    run_demo()
