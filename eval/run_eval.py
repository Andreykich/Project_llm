#!/usr/bin/env python3
"""
Запуск воспроизводимой оценки (eval) для агентной системы Tunatic.

Что делает:
- читает eval/testcases.jsonl
- для каждого кейса запускает пайплайн 4 агентов
  1) DataCollectorAgent (общение/структурирование user_data)
  2) WebParserAgent (сбор источников)
  3) ValidatorAgent (фильтрация/валидация)
  4) DataAnalyzerAgent (итоговый совет/ответ)
- сохраняет "сырые" артефакты (входы/выходы агентов) и метаданные (latency, статусы)

Важно:
- Скрипт не "рисует" метрики — он собирает воспроизводимые логи.
- Метрики считаются отдельным скриптом compute_metrics.py
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import os, json, time, argparse, pathlib, traceback

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def now_ms():
    return int(time.time()*1000)

def safe_write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(obj))

def run_single_case(case, agents, out_dir, repeats=1):
    """Запуск одного тест-кейса repeats раз (для устойчивости)."""
    results=[]
    for r in range(repeats):
        run_id = f"{case['id']}_run{r+1}"
        run_path = os.path.join(out_dir, run_id)
        ensure_dir(run_path)

        meta = {"id": case["id"], "category": case["category"], "run": r+1, "t_start_ms": now_ms()}
        status = {"ok": True, "errors": []}

        t0=time.time()
        try:
            collector, parser, validator, analyzer = agents

            # 1) DataCollectorAgent: здесь проще всего НЕ имитировать диалог,
            # а формировать user_data из prompt. Если у вас в проекте есть
            # функция/метод, который извлекает JSON из текста — используйте её.
            user_data = {
                "raw_prompt": case["prompt"],
                # минимальный набор полей, которые чаще всего нужны дальше:
                "industry": case["category"],
                "city": "N/A",
                "idea": "N/A"
            }
            safe_write(os.path.join(run_path, "01_user_prompt.txt"), case["prompt"])
            safe_write(os.path.join(run_path, "02_user_data.json"), user_data)

            # 2) WebParserAgent: если в вашем пайплайне парсер использует urls,
            # можно поддерживать "нулевой" режим (без url) или расширить кейсы.
            parsed = None
            try:
                parsed = parser.parse_from_prompt(case["prompt"])  # если реализовано
            except Exception:
                # fallback: если нет такого метода — просто пропускаем и пишем заглушку
                parsed = {"parsed_sources": [], "note": "parse_from_prompt отсутствует; подключите парсер к urls/поиску."}
            safe_write(os.path.join(run_path, "03_parser_output.json"), parsed)

            # 3) ValidatorAgent
            validated = None
            try:
                validated = validator.validate_data(parsed if isinstance(parsed, dict) else {"data": parsed})
            except Exception:
                validated = {"is_valid": False, "issues": ["validator.validate_data failed - see logs"]}
            safe_write(os.path.join(run_path, "04_validator_output.json"), validated)

            # 4) DataAnalyzerAgent: основной итоговый текст/структура
            advice = analyzer.generate_advice(user_data)
            safe_write(os.path.join(run_path, "05_final_answer.txt"), advice)

        except Exception as e:
            status["ok"]=False
            status["errors"].append({"type": type(e).__name__, "msg": str(e), "trace": traceback.format_exc()})
        finally:
            t1=time.time()
            meta["latency_sec"]=round(t1-t0, 3)
            meta["status"]=status
            safe_write(os.path.join(run_path, "meta.json"), meta)
            results.append(meta)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="Корень проекта (где main.py / agents/)")
    ap.add_argument("--cases", default="eval/testcases.jsonl", help="Путь к тест-кейсам")
    ap.add_argument("--out", default="eval_outputs/raw", help="Папка для сырых логов")
    ap.add_argument("--repeats", type=int, default=1, help="Повторы на кейс (для устойчивости), например 3")
    args = ap.parse_args()

    os.chdir(args.project_root)
    ensure_dir(args.out)

    # Импорты проекта
    from agents.data_collector import DataCollectorAgent
    from agents.web_parser import WebParserAgent
    from agents.validator import ValidatorAgent
    from agents.data_analyzer import DataAnalyzerAgent
    from database.json_db import JSONDatabase

    db = JSONDatabase("data/database.json")
    collector = DataCollectorAgent()
    parser = WebParserAgent()
    validator = ValidatorAgent()
    analyzer = DataAnalyzerAgent(db)

    agents = (collector, parser, validator, analyzer)

    all_runs=[]
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            case=json.loads(line)
            all_runs.extend(run_single_case(case, agents, args.out, repeats=args.repeats))

    ensure_dir("eval_outputs")
    with open("eval_outputs/runs_index.jsonl","w",encoding="utf-8") as f:
        for r in all_runs:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

    print(f"Done. Runs: {len(all_runs)}. Raw logs in {args.out}. Index: eval_outputs/runs_index.jsonl")

if __name__ == "__main__":
    main()
