#!/usr/bin/env python3
"""
Подсчёт метрик из eval_outputs/raw и eval_outputs/runs_index.jsonl.

Метрики:
- pass_rate (по статусу ok)
- latency p50/p90/p95
- sections_coverage: доля ответов, где найдены обязательные секции (по простым regex)
- sources_min2_rate: доля ответов, где найдено >=2 ссылок (http/https)
- (optional) json_valid_rate: если ваш финальный ответ в JSON и проходит schema (output_schema.json)

Также объединяет human_scores_template.csv (если заполнен) и считает средние баллы.
"""
import os, json, argparse, statistics, re, csv, math
from pathlib import Path

URL_RE = re.compile(r"https?://\S+")
def percentile(xs, p):
    xs=sorted(xs)
    if not xs: return None
    k=(len(xs)-1)*p
    f=math.floor(k); c=math.ceil(k)
    if f==c: return xs[int(k)]
    return xs[f]*(c-k)+xs[c]*(k-f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--index", default="eval_outputs/runs_index.jsonl")
    ap.add_argument("--raw_dir", default="eval_outputs/raw")
    ap.add_argument("--cases", default="eval/testcases.jsonl")
    ap.add_argument("--human", default="eval/human_scores_template.csv")
    ap.add_argument("--out", default="eval_outputs/metrics.json")
    args=ap.parse_args()

    # load cases to get must_sections
    cases={}
    with open(args.cases,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c=json.loads(line)
                cases[c["id"]]=c

    runs=[]
    with open(args.index,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))

    lat=[r.get("latency_sec") for r in runs if isinstance(r.get("latency_sec"), (int,float))]
    ok=[r for r in runs if r.get("status",{}).get("ok")]

    # sections and sources from final answer text
    sec_hits=[]
    src_hits=[]
    for r in runs:
        run_path=os.path.join(args.raw_dir, f"{r['id']}_run{r['run']}")
        ans_path=os.path.join(run_path,"05_final_answer.txt")
        text=""
        if os.path.exists(ans_path):
            text=open(ans_path,"r",encoding="utf-8",errors="ignore").read()

        must=cases.get(r["id"],{}).get("must_sections",[])
        found=0
        for s in must:
            # permissive: header can be "Идея:" or "## Идея" etc.
            if re.search(rf"(^|\n)\s*(#+\s*)?{re.escape(s)}\b", text, flags=re.IGNORECASE):
                found+=1
        sec_hits.append(found/len(must) if must else 0.0)

        urls=URL_RE.findall(text)
        src_hits.append(len(urls))

    metrics = {
        "n_runs": len(runs),
        "n_cases": len(cases),
        "repeats": max([r.get("run",1) for r in runs]) if runs else 0,
        "pass_rate_by_status_ok": round(len(ok)/len(runs), 3) if runs else None,
        "latency_sec": {
            "mean": round(statistics.mean(lat),3) if lat else None,
            "p50": round(percentile(lat,0.50),3) if lat else None,
            "p90": round(percentile(lat,0.90),3) if lat else None,
            "p95": round(percentile(lat,0.95),3) if lat else None,
        },
        "sections_coverage_mean": round(statistics.mean(sec_hits),3) if sec_hits else None,
        "sources_min2_rate": round(sum(1 for x in src_hits if x>=2)/len(src_hits),3) if src_hits else None,
        "sources_avg": round(statistics.mean(src_hits),3) if src_hits else None,
    }

    # human scores if filled
    if os.path.exists(args.human):
        rows=[]
        with open(args.human,"r",encoding="utf-8") as f:
            reader=csv.DictReader(f)
            for row in reader:
                # accept only filled rows
                try:
                    vals=[float(row[k]) for k in ["relevance_0_5","specificity_0_5","actionability_0_5","evidence_0_5"]]
                    if all(not math.isnan(v) for v in vals):
                        rows.append(vals)
                except Exception:
                    continue
        if rows:
            means=[statistics.mean([r[i] for r in rows]) for i in range(4)]
            metrics["human_scores_mean"]={
                "relevance": round(means[0],3),
                "specificity": round(means[1],3),
                "actionability": round(means[2],3),
                "evidence": round(means[3],3),
                "quality_mean": round(statistics.mean(means),3),
                "n_ratings": len(rows)
            }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)

    print("Saved:", args.out)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
