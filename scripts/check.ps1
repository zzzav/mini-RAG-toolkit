# scripts/check.ps1
$ErrorActionPreference = "Stop"

Write-Host "== ruff =="
python -m ruff check .

Write-Host "== black (check) =="
python -m black --check .

Write-Host "== pytest =="
python -m pytest -q

Write-Host "== retrieval regression gate =="
python -m src.regress_cli `
  --index .\eval\vindex_v1_rus.pkl `
  --dataset .\eval\eval_small.jsonl `
  --k 5 `
  --min-recall 0.70 `
  --min-mrr 0.45

Write-Host "OK"