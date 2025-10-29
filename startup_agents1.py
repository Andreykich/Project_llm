
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, hashlib, sqlite3, re, logging, datetime
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import requests
from bs4 import BeautifulSoup

def _load_dotenv(path="config/.env"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                k,v=line.split("=",1); k=k.strip(); v=v.strip()
                if (len(v)>=2) and ((v[0]==v[-1]) and v[0] in ("'",'"')): v=v[1:-1]
                if k and (k not in os.environ): os.environ[k]=v
    except FileNotFoundError:
        pass
_load_dotenv()

MODEL_NAME="deepseek/deepseek-v3.2-exp"
OPENROUTER_URL="https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger().info(f"✅ OPENROUTER_API_KEY найден: {bool(OPENROUTER_API_KEY)}")

DB_PATH=os.getenv("DB_PATH","trends.db")
JSONL_PATH=os.getenv("JSONL_PATH","corpus.jsonl")
EVIDENCE_WINDOW_DAYS=90
REQ_TIMEOUT=30

def now_utc_date(): return datetime.datetime.utcnow().date()
def iso_date(d): return d.isoformat()
def domain_of(url):
    m=re.match(r"^https?://([^/]+)", url.strip()); return m.group(1) if m else "unknown"

@contextmanager
def sqlite(path):
    cx=sqlite3.connect(path);
    try:
        cx.row_factory=sqlite3.Row; yield cx; cx.commit()
    finally:
        cx.close()

class SQLiteDAO:
    def __init__(self, path=DB_PATH):
        self.path=path; self._init()
    def _init(self):
        with sqlite(self.path) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS docs(
                id TEXT PRIMARY KEY,url TEXT,domain TEXT,lang TEXT,published_at TEXT,
                title TEXT,tags TEXT,summary TEXT,text TEXT,h_ud TEXT);""")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_docs_date ON docs(published_at);")
            cx.execute("CREATE INDEX IF NOT EXISTS idx_docs_ud   ON docs(h_ud);")
    @staticmethod
    def _hash_ud(url,title): return hashlib.sha256((url.strip()+"||"+title.strip()).encode()).hexdigest()
    def upsert_doc(self,d):
        h_ud=self._hash_ud(d["url"],d["title"])
        with sqlite(self.path) as cx:
            cx.execute("""INSERT INTO docs(id,url,domain,lang,published_at,title,tags,summary,text,h_ud)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET url=excluded.url,domain=excluded.domain,lang=excluded.lang,
            published_at=excluded.published_at,title=excluded.title,tags=excluded.tags,summary=excluded.summary,
            text=excluded.text,h_ud=excluded.h_ud""",
            (d["id"],d["url"],d["domain"],d["lang"],d["published_at"],d["title"],
             json.dumps(d.get("tags",[]),ensure_ascii=False),d.get("summary",""),d["text"],h_ud))
    def exists_ud(self,url,title):
        h_ud=self._hash_ud(url,title)
        with sqlite(self.path) as cx:
            return cx.execute("SELECT 1 FROM docs WHERE h_ud=? LIMIT 1",(h_ud,)).fetchone() is not None
    def recent_docs_keywords(self,keywords,days=90):
        since=iso_date(now_utc_date()-datetime.timedelta(days=days))
        with sqlite(self.path) as cx:
            q="SELECT * FROM docs WHERE published_at>=? AND ("; params=[since]; conds=[]
            for kw in keywords:
                like=f"%{kw}%"; conds.append("(title LIKE ? OR text LIKE ?)"); params += [like,like]
            q+= " OR ".join(conds)+") ORDER BY published_at DESC LIMIT 500;"
            return [dict(r) for r in cx.execute(q,params).fetchall()]

class Parser:
    UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language":"ru,en;q=0.9"}
    def __init__(self,dao): self.dao=dao
    @staticmethod
    def _normalize_date(dt_str):
        try: return datetime.date.fromisoformat(dt_str[:10]).isoformat()
        except: return iso_date(now_utc_date())
    def fetch_and_store(self,url,lang_hint="ru",tags=None):
        r=requests.get(url,headers=self.UA,timeout=REQ_TIMEOUT); r.raise_for_status()
        soup=BeautifulSoup(r.text,"html.parser")
        title=(soup.title.text or "").strip() if soup.title else url
        if self.dao.exists_ud(url,title): return False
        text="\n".join([p.get_text(" ",strip=True) for p in soup.find_all("p") if p.get_text(strip=True)])
        date_meta=soup.find("meta",{"property":"article:published_time"}) or soup.find("meta",{"name":"date"}) or soup.find("time")
        raw_date=(date_meta.get("content") or date_meta.get_text() or "") if date_meta else ""
        doc={"id":hashlib.md5(url.encode()).hexdigest(),"url":url,"domain":domain_of(url),"lang":"ru",
             "published_at":self._normalize_date(raw_date),"title":title,"tags":tags or [],"summary":"","text":text}
        self.dao.upsert_doc(doc)
        with open(JSONL_PATH,"a",encoding="utf-8") as f: f.write(json.dumps(doc,ensure_ascii=False)+"\n")
        return True

    def crawl_landing(self, landing_url: str, max_links: int = 20) -> list[str]:
        """Собирает смежные ссылки с лендинга, строго фильтруя домен и типы путей.
           - Habr: разрешаем ТОЛЬКО post|news|company, жёстко отсекаем /hubs/ и /articles/
           - restoran.ru: разрешаем только /news/ (включая региональные префиксы /msk/news/)
        """
        r = requests.get(landing_url, headers=self.UA, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        dom = domain_of(landing_url)
        links: list[str] = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()

            # Нормализация относительных и протокол-независимых ссылок
            if href.startswith("//"):
                href = f"https:{href}"
            elif href.startswith("/"):
                href = f"https://{dom}{href}"

            # Пропускаем якоря, mailto, javascript и т.п.
            if not href.startswith("http"):
                continue
            if href.startswith("mailto:") or href.startswith("javascript:") or href.endswith("#"):
                continue

            # Только тот же домен
            if domain_of(href) != dom:
                continue

            # Специальные правила по доменам
            if dom.endswith("habr.com"):
                # Жёсткий запрет на /hubs/ и /articles/
                if "/hubs/" in href or "/articles/" in href:
                    continue
                # Разрешаем ТОЛЬКО post|news|company
                if not re.match(r"^https://habr\.com/ru/(post|news|company)/", href):
                    continue

            if dom.endswith("restoran.ru") or dom.endswith("www.restoran.ru"):
                # Разрешаем только /news/ (включая региональные префиксы /msk/news/ и т.п.)
                if not re.match(r"^https?://(www\.)?restoran\.ru/([a-z]{3}/)?news/", href):
                    continue

            # По желанию — убрать query/fragment, чтобы меньше дублей
            href = re.sub(r"[#?].*$", "", href)

            links.append(href)
            if len(links) >= max_links:
                break

        # Уникализуем, сохраняя порядок
        return list(dict.fromkeys(links))


class OpenRouterClient:
    def __init__(self,model=MODEL_NAME): self.model=model
    def chat(self,messages,**gen):
        if not OPENROUTER_API_KEY:
            return {"choices":[{"message":{"role":"assistant","content":"{\"stub\":true}"}}]}
        payload={"model":self.model,"messages":messages,"max_tokens":gen.get("max_tokens",900),
                 "temperature":gen.get("temperature",0.1),"top_p":gen.get("top_p",0.9)}
        resp=requests.post(OPENROUTER_URL,headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"},
                           json=payload,timeout=REQ_TIMEOUT)
        resp.raise_for_status(); return resp.json()

KEYWORDS={"кофейня":["кофейня","кофе","coffee","cafe","кафе","horeca","foodservice"]}

class RetrievalSynthesisAgent:
    def __init__(self,client,dao): self.client=client; self.dao=dao
    def _search(self,domain,days=90):
        kws=list(dict.fromkeys([domain]+KEYWORDS.get(domain.lower(),[])))
        rows=self.dao.recent_docs_keywords(kws,days=days)
        return [{"url":r["url"],"title":r["title"],"date":r["published_at"],
                 "text":(r["text"][:1200]+"...") if len(r["text"])>1200 else r["text"]} for r in rows[:100]]
    def _evidence_ok(self,items):
        uniq=sorted(set(domain_of(i["url"]) for i in items)); return (len(uniq)>=2, uniq)
    def _json_or_repair(self,text):
        try: return json.loads(text)
        except: pass
        m=re.search(r"\{.*\}",text,re.S)
        if m:
            try: return json.loads(m.group(0))
            except: return None
        return None
    def _local_fallback(self,uq,items):
        srcs=[{"url":it["url"],"title":it["title"][:80],"date":it["date"]} for it in items[:5]]
        return {
            "idea": f"{uq.get('domain','general').capitalize()}: MVP под Москву/Россия на основе полученных источников.",
            "local_niches":[{"name":"B2B-подписки","why_now":"Предсказуемые расходы у офисов"},
                            {"name":"Лояльность/QR","why_now":"Удержание дешевле привлечения"}],
            "90_day_plan":[{"week":1,"tasks":["Карта конкурентов","Опрос 30 респондентов","MVP оффера"]},
                           {"week":2,"tasks":["Пилот подписки 20-30 клиентов","A/B цена/объем"]},
                           {"week":3,"tasks":["Отчет юнит-экономики","Решение о масштабировании"]}],
            "risks":[{"risk":"Слабая конверсия подписки","mitigation":"A/B + B2B партнёрства"},
                     {"risk":"Высокая аренда","mitigation":"Коворкинги/ревенью-шер"}],
            "unit_economics_notes":["CAC<=20% ARPU","GM>=65-70%","Payback<=12 мес","LTV/CAC>3"],
            "sources":srcs,
            "evidence":[f"{domain_of(s['url'])}: {s['title']}" for s in srcs]
        }
    def run(self,uq):
        items=self._search(uq.get("domain",""),days=EVIDENCE_WINDOW_DAYS)
        ok, uniq = self._evidence_ok(items)
        logging.info(f"[EVIDENCE] items={len(items)}, unique_domains={uniq}")
        if not ok:
            return json.dumps({"idea":"","local_niches":[],"90_day_plan":[],"risks":[],"unit_economics_notes":[],
                               "sources":[],"evidence":[
                "в базе данных нет информации по данному вопросу. хотите чтобы я все равно сказал рекомендацию на свое усмотрение? (не подкреплённое данными)"
            ]},ensure_ascii=False)
        msgs=[{"role":"system","content":"Верни ТОЛЬКО JSON со строгими ключами: idea, local_niches, 90_day_plan, risks, unit_economics_notes, sources, evidence. Локальные ниши — Москва/Россия. Без текста вокруг."},
              {"role":"user","content":json.dumps({"UserQuery":uq,"snippets":items},ensure_ascii=False)}]
        fin=OpenRouterClient().chat(msgs,max_tokens=900,temperature=0.1,top_p=0.9)
        raw=fin["choices"][0]["message"]["content"]
        obj=self._json_or_repair(raw)
        if obj is None: obj=self._local_fallback(uq,items)
        return json.dumps(obj,ensure_ascii=False)

class IntakeAgent:
    def __init__(self,client): self.client=client
    def run(self,raw):
        out=self.client.chat([{"role":"system","content":"Верни JSON {\"domain\":\"...\",\"location\":\"...\",\"details\":\"...\"} без текста вокруг. Location по умолчанию Москва/Россия."},
                              {"role":"user","content":raw}],max_tokens=600,temperature=0.6,top_p=0.95)
        try: obj=json.loads(out["choices"][0]["message"]["content"])
        except: obj={"domain":raw.strip()[:40],"location":"Москва/Россия","details":""}
        obj.setdefault("location","Москва/Россия"); obj.setdefault("details",""); return obj

class ValidatorInAgent:
    def __init__(self,client): self.client=client
    def run(self,uq):
        if not uq.get("domain"): uq["domain"]="general"
        if not uq.get("location"): uq["location"]="Москва/Россия"
        uq.setdefault("details",""); return uq

class ValidatorOutAgent:
    KEYS=["idea","local_niches","90_day_plan","risks","unit_economics_notes","sources","evidence"]
    def __init__(self,client): self.client=client
    def run(self,text):
        try: obj=json.loads(text)
        except: obj={}
        for k in self.KEYS: obj.setdefault(k, [] if k!="idea" else "")
        for s in obj.get("sources",[]):
            if "date" in s and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(s["date"]) or ""):
                s["date"]=iso_date(now_utc_date())
        return json.dumps(obj,ensure_ascii=False)

class Orchestrator:
    def __init__(self):
        self.dao=SQLiteDAO(DB_PATH); self.parser=Parser(self.dao)
        self.intake=IntakeAgent(OpenRouterClient()); self.vin=ValidatorInAgent(OpenRouterClient())
        self.rs=RetrievalSynthesisAgent(OpenRouterClient(),self.dao); self.vout=ValidatorOutAgent(OpenRouterClient())
    def ingest(self,urls,crawl_adjacent=True,max_links=12,lang_hint="ru",tags=None):
        added=0
        for u in urls:
            try:
                links=[u]+(self.parser.crawl_landing(u,max_links=max_links) if crawl_adjacent else [])
                for link in links:
                    try:
                        if self.parser.fetch_and_store(link,lang_hint,tags): added+=1
                    except Exception as e:
                        logging.warning(f"[INGEST] fail link {link}: {e}")
            except Exception as e:
                logging.warning(f"[INGEST] fail seed {u}: {e}")
        return added
    def run_query(self,user_text):
        uq=self.intake.run(user_text); uq=self.vin.run(uq); raw=self.rs.run(uq); return self.vout.run(raw)

SEED_URLS=[
    "https://habr.com/ru/hub/startups/",
    "https://rb.ru/tag/startapy/",
    "https://rb.ru/tag/horeca/",
    "https://www.retail.ru/",
    "https://restoran.ru/news/",
]

def run_demo():
    orch=Orchestrator()
    added=orch.ingest(SEED_URLS,crawl_adjacent=True,max_links=12,lang_hint="ru",tags=["startup","trend"])
    logging.info(f"Ingest added {added} docs.")
    user="Сфера: кофейня; город: Москва; детали: подписка и бизнес-ланчи для офисных парков"
    print(orch.run_query(user))

if __name__=="__main__":
    run_demo()
