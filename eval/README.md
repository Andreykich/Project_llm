# Tunatic Evaluation Pack (для финальной защиты)

Этот пакет закрывает требования куратора по метрикам:
- полный набор test cases (>=50)
- воспроизводимый пайплайн прогона
- сырые логи каждого агента
- агрегирование метрик и шаблон ручной разметки

## Что внутри
- `eval/testcases.jsonl` — 50 тест-кейсов (стратифицированная выборка)
- `eval/run_eval.py` — запуск пайплайна и сохранение артефактов
- `eval/compute_metrics.py` — подсчёт метрик из логов
- `eval/rubric.md` — рубрика (0–5) для human/LLM-judge
- `eval/human_scores_template.csv` — шаблон разметки
- `eval/output_schema.json` — опциональная JSON-схема финального ответа

## Быстрый старт
Из корня проекта:

```bash
python eval/run_eval.py --project_root . --cases eval/testcases.jsonl --out eval_outputs/raw --repeats 1
python eval/compute_metrics.py --index eval_outputs/runs_index.jsonl --raw_dir eval_outputs/raw --cases eval/testcases.jsonl --human eval/human_scores_template.csv --out eval_outputs/metrics.json
```

### Для устойчивости (вероятностная система)
Рекомендуется:
- `--repeats 3` для 10–20 кейсов (или для всех, если позволяет время/лимиты)
- затем сравнить разброс по human-оценке и pass-rate

## Как приложить к отчёту
1) Добавить в репозиторий папку `eval/`
2) Приложить (или сослаться) на:
   - `eval/testcases.jsonl` (в полном объёме)
   - `eval_outputs/metrics.json`
   - 5–8 примеров из `eval_outputs/raw/TCxxx_run1/`

## Важно
`run_eval.py` содержит мягкий "адаптер" под ваш код. Если у вас парсер работает только по URL —
добавьте поле `urls` в тест-кейсы и используйте `parser.parse_website(url)` внутри `run_eval.py`.
