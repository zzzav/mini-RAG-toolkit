# scripts/check.ps1
$ErrorActionPreference = "Stop"

Write-Host "== ruff =="
python -m ruff check --fix .

Write-Host "== black (check) =="
python -m black --check .

Write-Host "== pytest =="
python -m pytest -q

Write-Host "== retrieval regression gate =="
python -m src.regress_cli `
  --index-bm25 .\eval\bm25_index_v1_rus.pkl `
  --index-vector .\eval\vindex_v1_rus.pkl `
  --dataset .\eval\eval_small.jsonl `
  --retriever fusion `
  --rerank `
  --rerank-top-n 10 `
  --proximity-window 5 `
  --k 5 `
  --min-recall 0.7 `
  --min-mrr 0.5

Write-Host "OK"