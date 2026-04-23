# Baselines

## 1. Snapshot

Baseline снят на фиксированных данных репозитория:

- документы: `eval/docs/v1_rus`
- retrieval dataset: `eval/eval_small.jsonl`
- answer dataset: `eval/eval_answer_small.jsonl`
- индексы: `eval/bm25_index_v1_rus.pkl`, `eval/vindex_v1_rus.pkl`
- дата и время снимка: `2026-04-23 16:15:28 +10:00`

Ключевые параметры baseline-прогонов:

- `k = 5`
- `fusion_method = rrf`
- `fusion_top_n = 10`
- `rerank_top_n = 10`
- `proximity_window = 5`
- `llm = extract`

Индексы были уже свежими относительно `eval/docs/v1_rus`, поэтому пересборка не потребовалась.

## 2. Retrieval baselines

| strategy | k | recall_mean | mrr_mean | n |
|---|---:|---:|---:|---:|
| vector | 5 | 0.75 | 0.6875 | 8 |
| bm25 | 5 | 0.75 | 0.53125 | 8 |
| fusion | 5 | 0.75 | 0.5625 | 8 |
| rerank_vector | 5 | 0.75 | 0.6875 | 8 |
| rerank_bm25 | 5 | 0.75 | 0.53125 | 8 |

## 3. Answer baselines

| pipeline | n | contains_rate | no_info_accuracy |
|---|---:|---:|---:|
| vector_extract | 10 | 0.625 | 0.0 |
| bm25_extract | 10 | 0.625 | 0.0 |
| fusion_extract | 10 | 0.625 | 0.0 |
| rerank_vector_extract | 10 | 0.625 | 0.0 |
| rerank_bm25_extract | 10 | 0.625 | 0.0 |

## 4. Commands used

### Retrieval baselines

```powershell
.venv\Scripts\python.exe -m src.compare_retrievers_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_small.jsonl --top-k 5 --fusion-method rrf --fusion-top-n 10 --rerank-top-n 10 --proximity-window 5 --format json
```

### Answer baselines

```powershell
.venv\Scripts\python.exe -m src.compare_pipelines_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_answer_small.jsonl --top-k 5 --fusion-method rrf --fusion-top-n 10 --rerank-top-n 10 --proximity-window 5 --format json
```

## 5. Notes

- Индексы не пересобирались: `eval/bm25_index_v1_rus.pkl` и `eval/vindex_v1_rus.pkl` были актуальны относительно `eval/docs/v1_rus`.
- Retrieval baseline снимался через `compare_retrievers_cli.py`, answer baseline - через `compare_pipelines_cli.py`.
- В `compare_pipelines_cli.py` используется фиксированный `llm = extract`, это соответствует текущему коду проекта.
- Для answer baseline `no_info_accuracy` получилось `0.0` на текущем датасете.
