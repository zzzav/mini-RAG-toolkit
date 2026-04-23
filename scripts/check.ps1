# Основной quality gate для проекта.
$ErrorActionPreference = "Stop"

# Проверка линтером и автоисправление мелких проблем.
Write-Host "== ruff =="
python -m ruff check --fix .

# Проверка форматирования Black без изменения файлов.
Write-Host "== black (check) =="
python -m black --check .

# Прогон тестов.
Write-Host "== pytest =="
python -m pytest -q

# Регрессионная проверка retrieval-метрик на фиксированных baseline-данных.
Write-Host "== retrieval regression gate =="
python -m src.regress_cli `
  --index-bm25 .\eval\bm25_index_v1_rus.pkl `
  --index-vector .\eval\vindex_v1_rus.pkl `
  --dataset .\eval\eval_small.jsonl `
  --retriever fusion `
  --rerank `
  --rerank-top-n 10 `
  --proximity-window 5 `
  --top-k 5 `
  --min-recall 0.7 `
  --min-mrr 0.5

# Если все шаги прошли, качество репозитория в порядке.
Write-Host "OK"
