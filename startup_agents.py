#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Startup Trends Multi-Agent System — Python 3.13+
# stdlib + requests + beautifulsoup4; OpenRouter client; SQLite+JSONL; ReAct-ish; robust parsing.

from __future__ import annotations

import os, re, json, hashlib, logging, sqlite3
from contextlib import contextmanager
from typing import Any
from datetime import datetime, UTC, date, timedelta

import requests
from bs4 import BeautifulSoup

# -------------------- .env autoload --------------------
def _load_dotenv(path: str = "config/.env") -> None:
    """Load environment variables from a .env file (stdlib-only)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                if k and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass

_load_dotenv()

# ------------------------------- Config ---------------------------------------
MODEL_NAME = "deepseek/deepseek-v3.2-exp"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger().info(f"✅ OPENROUTER_API_KEY найден: {bool(OPENROUTER_API_KEY)}")

DB_PATH = os.getenv("DB_PATH", "trends.db")
JSONL_PATH = os.getenv("JSONL_PATH", "corpus.jsonl")

MAX_DAYS_RETRO = 180
EVIDENCE_WINDOW_DAYS = 90
MAX_REACT_STEPS = 6
REQ_TIMEOUT = 30
PIPELINE_HARD_TIMEOUT = 120

DEMO_EVIDENCE_RELAX = os.getenv("DEMO_EVIDENCE_RELAX", "0") == "1"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY or 'NO_KEY'}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Startup-Trends-Agent",
}

# ------------------------------- Utils ----------------------------------------
def now_utc_date() -> date:
    return datetime.now(UTC).date()

def iso_date(d: date) -> str:
    return d.isoformat()

def domain_of(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", url.strip())
    return m.group(1) if m else "unknown"

@contextmanager
def sqlite_conn(path: str):
    cx = sqlite3.connect(path)
    try:
        cx.row_factory = sqlite3.Row
        yield cx
        cx.commit()
    finally:
        cx.close()

# -------------------------------- DAO -----------------------------------------
class SQLiteDAO:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._init()

    def _init(self) -> None:
        with sqlite_conn(self.path) as cx:
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
        return hashlib.sha256((url.strip() + "||" + title.strip()).encode()).hexdigest()

    def upsert_doc(self, d: dict[str, Any]) -> None:
        h_ud = self._hash_ud(d["url"], d["title"])
        with sqlite_conn(self.path) as cx:
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
                d.get("summary", ""), d["text"], h_ud
            ))

    def exists_ud(self, url: str, title: str) -> bool:
        h_ud = self._hash_ud(url, title)
        with sqlite_conn(self.path) as cx:
            return cx.execute("SELECT 1 FROM docs WHERE h_ud=? LIMIT 1;", (h_ud,)).fetchone() is not None

    def recent_docs_keywords(self, keywords: list[str], days: int = 90) -> list[dict[str, Any]]:
        since = iso_date(now_utc_date() - timedelta(days=days))
        with sqlite_conn(self.path) as cx:
            if not keywords:
                rows = cx.execute(
                    "SELECT * FROM docs WHERE published_at>=? ORDER BY published_at DESC LIMIT 500;", (since,)
                ).fetchall()
                return [dict(r) for r in rows]
            conds: list[str] = []
            params: list[Any] = [since]
            for kw in keywords:
                like = f"%{kw}%"
                conds.append("(title LIKE ? OR text LIKE ?)")
                params += [like, like]
            q = "SELECT * FROM docs WHERE published_at>=? AND (" + " OR ".join(conds) + ") ORDER BY published_at DESC LIMIT 500;"
            rows = cx.execute(q, params).fetchall()
            return [dict(r) for r in rows]

# ------------------------------- Parser ---------------------------------------
class Parser:
    HTTP_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.9",
        "Accept-Language": "ru,en;q=0.9",
    }

    def __init__(self, dao: SQLiteDAO):
        self.dao = dao

    @staticmethod
    def _normalize_date(dt_str: str) -> str:
        try:
            return datetime.fromisoformat(dt_str[:10]).date().isoformat()
        except Exception:
            return iso_date(now_utc_date())

    def fetch_and_store(self, url: str, *, lang_hint: str = "ru", tags: list[str] | None = None) -> bool:
        resp = requests.get(url, headers=self.HTTP_HEADERS, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        title = (soup.title.text.strip() if soup.title and soup.title.text else url)
        if self.dao.exists_ud(url, title):
            return False

        text = "\n".join([p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)])

        date_meta = soup.find("meta", {"property": "article:published_time"}) or \
                    soup.find("meta", {"name": "date"}) or soup.find("time")
        raw_date = (date_meta.get("content") or date_meta.get_text() or "") if date_meta else ""

        doc: dict[str, Any] = {
            "id": hashlib.md5(url.encode()).hexdigest(),
            "url": url,
            "domain": domain_of(url),
            "lang": lang_hint if lang_hint else ("ru" if re.search(r"[А-Яа-яЁё]", html) else "en"),
            "published_at": self._normalize_date(raw_date),
            "title": title,
            "tags": tags or [],
            "summary": "",
            "text": text,
        }
        self.dao.upsert_doc(doc)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return True

    def crawl_landing(self, landing_url: str, *, max_links: int = 20) -> list[str]:
        """Collect adjacent links with strict filters.
           - Habr: only /ru/(post|news|company)/..., reject /hubs/ and /articles/
           - restoran.ru: only /(msk/)?news/..."""
        resp = requests.get(landing_url, headers=self.HTTP_HEADERS, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        dom = domain_of(landing_url)
        out: list[str] = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()

            if href.startswith("//"):
                href = f"https:{href}"
            elif href.startswith("/"):
                href = f"https://{dom}{href}"

            if not href.startswith("http"):
                continue
            if href.startswith("mailto:") or href.startswith("javascript:") or href.endswith("#"):
                continue
            if domain_of(href) != dom:
                continue

            if dom.endswith("habr.com"):
                if "/hubs/" in href or "/articles/" in href:
                    continue
                if not re.match(r"^https://habr\.com/ru/(post|news|company)/", href):
                    continue

            if dom.endswith("restoran.ru") or dom.endswith("www.restoran.ru"):
                if not re.match(r"^https?://(www\.)?restoran\.ru/([a-z]{3}/)?news/", href):
                    continue

            href = re.sub(r"[#?].*$", "", href)
            out.append(href)
            if len(out) >= max_links:
                break

        return list(dict.fromkeys(out))

# ---------------------------- OpenRouter client --------------------------------
class OpenRouterClient:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def chat(self, messages: list[dict[str, str]], **gen) -> dict[str, Any]:
        if not OPENROUTER_API_KEY:
            return {"choices": [{"message": {"role": "assistant", "content": "{\"stub\": true}"}}]}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": gen.get("max_tokens", 900),
            "temperature": gen.get("temperature", 0.1),
            "top_p": gen.get("top_p", 0.9),
        }
        r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        return r.json()

# ---------------------------- Retrieval / Synthesis ----------------------------
KEYWORDS: dict[str, list[str]] = {
    "кофейня": ["кофейня", "кофе", "coffee", "cafe", "кафе", "horeca", "foodservice"],
    "horeca": ["horeca", "foodservice", "ресторан", "кафе", "кофе", "food&beverage", "f&b"],
    "retail": ["ритейл", "retail", "магазин", "ecommerce", "омниканал", "маркетплейс"],
}

class RetrievalSynthesisAgent:
    def __init__(self, client: OpenRouterClient, dao: SQLiteDAO):
        self.client = client
        self.dao = dao

    def _expand_kws(self, domain: str) -> list[str]:
        base = [domain] if domain else []
        for k, arr in KEYWORDS.items():
            if domain and k in domain.lower():
                base += arr
        return list(dict.fromkeys(base))

    def _search(self, domain: str, days: int = 90) -> list[dict[str, Any]]:
        rows = self.dao.recent_docs_keywords(self._expand_kws(domain), days=days)
        return [{
            "url": r["url"],
            "title": r["title"],
            "date": r["published_at"],
            "text": (r["text"][:1200] + "...") if len(r["text"]) > 1200 else r["text"],
        } for r in rows[:100]]

    def _evidence_ok(self, items: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        uniq = sorted({domain_of(i["url"]) for i in items})
        ok = len(items) >= 2 if DEMO_EVIDENCE_RELAX else len(uniq) >= 2
        return ok, uniq

    def _json_or_repair(self, text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    def _local_fallback(self, uq: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
        srcs = [{"url": it["url"], "title": it["title"][:80], "date": it["date"]} for it in items[:5]]
        return {
            "idea": f"{(uq.get('domain') or 'general').capitalize()}: MVP под Москву/Россия на основе собранных источников.",
            "local_niches": [
                {"name": "B2B-подписки у офисов", "why_now": "Предсказуемые расходы, гибридная работа"},
                {"name": "Лояльность/QR", "why_now": "Удержание дешевле привлечения"},
            ],
            "90_day_plan": [
                {"week": 1, "tasks": ["Карта конкурентов", "Опрос 30 респондентов", "MVP оффера"]},
                {"week": 2, "tasks": ["Пилот подписки 20–30 клиентов", "A/B цена/объем"]},
                {"week": 3, "tasks": ["Отчет по юнит-экономике", "Решение о масштабировании"]},
            ],
            "risks": [
                {"risk": "Слабая конверсия подписки", "mitigation": "A/B + B2B-партнёрства"},
                {"risk": "Высокая аренда", "mitigation": "Коворкинги / revenue-share"},
            ],
            "unit_economics_notes": ["CAC ≤ 20% ARPU", "GM ≥ 65–70%", "Payback ≤ 12 мес", "LTV/CAC > 3"],
            "sources": srcs,
            "evidence": [f"{domain_of(s['url'])}: {s['title']}" for s in srcs],
        }

    def run(self, uq: dict[str, Any]) -> str:
        items = self._search(uq.get("domain", ""), days=EVIDENCE_WINDOW_DAYS)
        ok, uniq = self._evidence_ok(items)
        logging.info(f"[EVIDENCE] items={len(items)}, unique_domains={uniq}")
        if not ok:
            return json.dumps({
                "idea": "",
                "local_niches": [],
                "90_day_plan": [],
                "risks": [],
                "unit_economics_notes": [],
                "sources": [],
                "evidence": ["в базе данных нет информации по данному вопросу. хотите чтобы я все равно сказал рекомендацию на свое усмотрение? (не подкреплённое данными)"],
            }, ensure_ascii=False)

        synth_msgs = [
            {
                "role": "system",
                "content": (
                    "Верни ТОЛЬКО valid JSON (без лишнего текста) строго такого вида:\n"
                    "{\n"
                    "  \"idea\": \"str\",\n"
                    "  \"local_niches\": [{\"name\":\"str\",\"why_now\":\"str\"}],\n"
                    "  \"90_day_plan\": [{\"week\":1,\"tasks\":[\"str\"]}],\n"
                    "  \"risks\": [{\"risk\":\"str\",\"mitigation\":\"str\"}],\n"
                    "  \"unit_economics_notes\": [\"str\"],\n"
                    "  \"sources\": [{\"url\":\"https://...\",\"title\":\"str\",\"date\":\"YYYY-MM-DD\"}],\n"
                    "  \"evidence\": [\"str\"]\n"
                    "}\n"
                    "Локальные ниши — Москва/Россия. Краткие заметки: CAC, LTV, GM, Payback."
                ),
            },
            {"role": "user", "content": json.dumps({"UserQuery": uq, "snippets": items}, ensure_ascii=False)},
        ]
        fin = OpenRouterClient().chat(synth_msgs, max_tokens=900, temperature=0.1, top_p=0.9)
        raw = fin["choices"][0]["message"].get("content", "")
        obj = self._json_or_repair(raw) or self._local_fallback(uq, items)
        return json.dumps(obj, ensure_ascii=False)

# ---------------------------- Intake / Validators ------------------------------
class IntakeAgent:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    def run(self, raw: str) -> dict[str, Any]:
        out = self.client.chat(
            [
                {"role": "system", "content": "Верни JSON {\"domain\":\"...\",\"location\":\"...\",\"details\":\"...\"} без текста вокруг. Location по умолчанию Москва/Россия."},
                {"role": "user", "content": raw},
            ],
            max_tokens=600, temperature=0.6, top_p=0.95,
        )
        try:
            content = out["choices"][0]["message"].get("content", "")
            obj = json.loads(content)
        except Exception:
            obj = {"domain": raw.strip()[:40], "location": "Москва/Россия", "details": ""}
        obj.setdefault("location", "Москва/Россия")
        obj.setdefault("details", "")
        return obj

class ValidatorInAgent:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    def run(self, uq: dict[str, Any]) -> dict[str, Any]:
        uq.setdefault("domain", "general")
        uq.setdefault("location", "Москва/Россия")
        uq.setdefault("details", "")
        return uq

class ValidatorOutAgent:
    KEYS = ["idea","local_niches","90_day_plan","risks","unit_economics_notes","sources","evidence"]

    def __init__(self, client: OpenRouterClient):
        self.client = client

    @staticmethod
    def _as_list(v: Any) -> list[Any]:
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    def run(self, text: str) -> str:
        try:
            obj = json.loads(text)
        except Exception:
            obj = {}

        for k in self.KEYS:
            obj.setdefault(k, [] if k != "idea" else "")

        # idea
        if not isinstance(obj["idea"], str):
            obj["idea"] = str(obj["idea"])

        # local_niches -> list[{name,why_now}]
        niches: list[dict[str, str]] = []
        for item in self._as_list(obj.get("local_niches")):
            if isinstance(item, dict):
                name = str(item.get("name", item.get("title", ""))).strip()
                why = str(item.get("why_now", "")).strip()
                if name:
                    niches.append({"name": name, "why_now": why})
            else:
                s = str(item).strip()
                if s:
                    niches.append({"name": s, "why_now": ""})
        obj["local_niches"] = niches

        # 90_day_plan -> list[{week:int,tasks:list[str]}]
        plan: list[dict[str, Any]] = []
        for idx, p in enumerate(self._as_list(obj.get("90_day_plan")), start=1):
            if isinstance(p, dict):
                week = int(p.get("week", idx))
                tasks = [str(t).strip() for t in self._as_list(p.get("tasks")) if str(t).strip()]
                if tasks:
                    plan.append({"week": week, "tasks": tasks})
            else:
                s = str(p).strip()
                if s:
                    plan.append({"week": idx, "tasks": [s]})
        obj["90_day_plan"] = plan

        # risks -> list[{risk,mitigation}]
        risks: list[dict[str, str]] = []
        for r in self._as_list(obj.get("risks")):
            if isinstance(r, dict):
                risks.append({"risk": str(r.get("risk","")).strip(), "mitigation": str(r.get("mitigation","")).strip()})
            else:
                s = str(r).strip()
                if s:
                    risks.append({"risk": s, "mitigation": ""})
        obj["risks"] = [r for r in risks if r["risk"]]

        # unit_economics_notes -> list[str]
        obj["unit_economics_notes"] = [str(x).strip() for x in self._as_list(obj.get("unit_economics_notes")) if str(x).strip()]

        # sources -> list[{url,title,date}]
        def _source(s: Any) -> dict[str, str]:
            if isinstance(s, dict):
                url = str(s.get("url", "")).strip()
                title = str(s.get("title", "")).strip()
                dt = str(s.get("date", "")).strip()
            else:
                sv = str(s).strip()
                url = sv if sv.startswith("http") else ""
                title = "" if url else sv
                dt = ""
            if not dt or not re.match(r"^\d{4}-\d{2}-\d{2}$", dt):
                dt = iso_date(now_utc_date())
            return {"url": url, "title": title, "date": dt}

        obj["sources"] = [_source(s) for s in self._as_list(obj.get("sources")) if s is not None]
        obj["evidence"] = [str(e).strip() for e in self._as_list(obj.get("evidence")) if str(e).strip()]

        return json.dumps(obj, ensure_ascii=False)

# ------------------------------- Orchestrator ----------------------------------
class Orchestrator:
    def __init__(self):
        self.dao = SQLiteDAO(DB_PATH)
        self.parser = Parser(self.dao)
        self.client = OpenRouterClient()
        self.intake = IntakeAgent(self.client)
        self.vin = ValidatorInAgent(self.client)
        self.rs = RetrievalSynthesisAgent(self.client, self.dao)
        self.vout = ValidatorOutAgent(self.client)

    def _fallback_seed(self, u: str) -> str | None:
        if "habr.com/ru/hub/startups" in u:
            return "https://habr.com/ru/news/"
        if "restoran.ru" in u:
            return "https://www.restoran.ru/msk/news/"
        return None

    def ingest(self, urls: list[str], *, crawl_adjacent: bool = False, max_links: int = 12,
               lang_hint: str = "ru", tags: list[str] | None = None) -> int:
        added = 0
        for u in urls:
            try:
                links = [u] + (self.parser.crawl_landing(u, max_links=max_links) if crawl_adjacent else [])
            except Exception as e:
                logging.warning(f"[INGEST] fail seed {u}: {e}")
                alt = self._fallback_seed(u)
                if alt:
                    logging.info(f"[INGEST] fallback → {alt}")
                    try:
                        links = [alt] + (self.parser.crawl_landing(alt, max_links=max_links) if crawl_adjacent else [])
                    except Exception as e2:
                        logging.warning(f"[INGEST] fallback fail {alt}: {e2}")
                        links = []
                else:
                    links = []
            for link in links:
                try:
                    if self.parser.fetch_and_store(link, lang_hint=lang_hint, tags=tags):
                        added += 1
                except Exception as e:
                    logging.warning(f"[INGEST] fail link {link}: {e}")
        return added

    def run_query(self, user_text: str) -> str:
        uq = self.intake.run(user_text)
        uq = self.vin.run(uq)
        raw = self.rs.run(uq)
        return self.vout.run(raw)

# --------------------------- Stable seed URLs ----------------------------------
SEED_URLS: list[str] = [
    "https://habr.com/ru/news/",            # Habr news (стабильно)
    "https://rb.ru/tag/startapy/",          # RB стартапы
    "https://rb.ru/story/horeca/",          # RB HoReCa
    "https://www.retail.ru/news/",          # Retail news
    # можно добавить 1–2 конкретные статьи, которые у тебя точно открываются:
    # "https://habr.com/ru/post/810000/",
    # "https://rb.ru/story/kofe-i-horeca-2025/"
]

def run_demo() -> None:
    orch = Orchestrator()
    added = orch.ingest(SEED_URLS, crawl_adjacent=False, lang_hint="ru", tags=["startup", "trend"])
    logging.info(f"Ingest added {added} docs.")
    user = "Сфера: кофейня; город: Москва; детали: подписка и бизнес-ланчи для офисных парков"
    print(orch.run_query(user))

if __name__ == "__main__":
    run_demo()
