# Mini-RAG Toolkit

Учебный mini-RAG toolkit на Python для экспериментов с retrieval, rerank и RAG-пайплайнами.

## Что это

Проект собран как практический набор CLI для работы с небольшим RAG-стеком:

- несколько retriever'ов: `vector`, `bm25`, `fusion`
- rerank поверх retrieval
- retrieval evaluation
- answer evaluation
- сравнение retriever'ов
- сравнение end-to-end pipeline
- chunk experiments
- regression gate

Фокус проекта - не на сложной теории, а на воспроизводимых сценариях: построить индекс, прогнать поиск, посчитать метрики и сравнить варианты пайплайна.

## Быстрый маршрут

Если нужен самый короткий путь через проект, то обычно хватает такого порядка:

1. Построить индексы
   - `python -m src.vector_search --docs .\eval\docs\v1_rus --index-out .\eval\vindex_local.pkl`
   - `python -m src.rag_cli --docs .\eval\docs\v1_rus --index-type bm25 --index-out .\eval\bm25_index_local.pkl`
2. Прогнать поиск или RAG
   - `python -m src.rag_cli --index-vector .\eval\vindex_v1_rus.pkl --retriever vector --query "payment invoice" --llm extract`
3. Прогнать retrieval eval
   - `python -m src.eval_cli --index-vector .\eval\vindex_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever vector --top-k 5`
4. Прогнать answer eval
   - `python -m src.eval_answer_cli --index .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_answer_small.jsonl --retriever bm25 --llm extract --top-k 5`
5. Сравнить стратегии
   - `python -m src.compare_retrievers_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_small.jsonl --top-k 5`
   - `python -m src.compare_pipelines_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_answer_small.jsonl --top-k 5`
6. Прогнать chunk experiments
   - `python -m src.chunk_experiments_cli --docs .\eval\docs\v1_rus --dataset .\eval\eval_small.jsonl --top-k 5 --chunking-config "((200, 40), (400, 80), (800, 120))"`
7. Прогнать regression gate
   - `python -m src.regress_cli --index-bm25 .\eval\bm25_index_v1_rus.pkl --index-vector .\eval\vindex_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever fusion --rerank --rerank-top-n 10 --proximity-window 5 --top-k 5 --min-recall 0.7 --min-mrr 0.5`
   - `.\scripts\check.ps1`

## Что умеет toolkit

- строить vector index и BM25 index
- искать по vector / BM25 / fusion
- включать rerank поверх результатов поиска
- выводить контекст, источники и ответ в `rag_cli`
- считать retrieval-метрики `recall@k` и `mrr@k`
- считать answer-метрики `contains_rate` и `no_info_accuracy`
- сравнивать retriever'ы на одном датасете
- сравнивать end-to-end пайплайны на одном датасете ответов
- прогонять эксперименты по разным chunking config
- запускать regression gate через `scripts/check.ps1`

## Структура проекта

- `src/vector_search.py` - vector search, build/load index, CLI для индексации и поиска
- `src/bm25_search.py` - BM25 index и BM25 search
- `src/rag_cli.py` - основной RAG CLI: build mode и search mode
- `src/eval_cli.py` - retrieval eval
- `src/eval_answer_cli.py` - answer eval
- `src/compare_retrievers_cli.py` - сравнение retriever'ов
- `src/compare_pipelines_cli.py` - сравнение end-to-end pipeline
- `src/chunk_experiments_cli.py` - эксперименты по chunking config
- `src/regress_cli.py` - regression gate по порогам метрик
- `src/retrieval_types.py` - общие типы и наборы допустимых значений CLI
- `scripts/check.ps1` - основной quality gate для репозитория

## Быстрый старт

В репозитории нет `requirements.txt`, поэтому зависимости ставятся в локальное окружение вручную.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install numpy scikit-learn pytest ruff black
```

Основная проверка проекта:

```powershell
.\scripts\check.ps1
```

Что делает `scripts/check.ps1`:

- `ruff check --fix .`
- `black --check .`
- `pytest -q`
- regression gate через `src.regress_cli`

## Построение индексов

### Vector index

```powershell
python -m src.vector_search --docs .\eval\docs\v1_rus --index-out .\eval\vindex_local.pkl
```

### BM25 index

BM25 индекс строится через `rag_cli` в режиме build:

```powershell
python -m src.rag_cli --docs .\eval\docs\v1_rus --index-type bm25 --index-out .\eval\bm25_index_local.pkl
```

### Vector index через `rag_cli`

Если нужен тот же build flow через один CLI:

```powershell
python -m src.rag_cli --docs .\eval\docs\v1_rus --index-type vector --index-out .\eval\vindex_local.pkl
```

## Поиск / RAG

### Vector search

```powershell
python -m src.vector_search --index-in .\eval\vindex_v1_rus.pkl --query "payment invoice" --top-k 5
```

### RAG на vector

```powershell
python -m src.rag_cli --index-vector .\eval\vindex_v1_rus.pkl --retriever vector --query "payment invoice" --llm extract
```

### RAG на BM25

```powershell
python -m src.rag_cli --index-bm25 .\eval\bm25_index_v1_rus.pkl --retriever bm25 --query "payment invoice" --llm extract
```

### RAG с fusion

```powershell
python -m src.rag_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --retriever fusion --fusion-method rrf --fusion-top-n 5 --query "payment invoice" --llm extract
```

### Rerank

```powershell
python -m src.rag_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --retriever fusion --rerank --rerank-top-n 10 --proximity-window 5 --query "payment invoice" --llm extract
```

### Полезные флаги `rag_cli`

- `--format text|json`
- `--output <path>`
- `--show-prompt`
- `--context-only`
- `--inline-citations`
- `--use-synonyms`
- `--no-stop-words`
- `--filter-source ...`
- `--filter-ext ...`
- `--filter-source-contains ...`
- `--for-eval-jsonl-out`

## Retrieval evaluation

`eval_cli.py` считает retrieval-метрики:

- `recall@k`
- `mrr@k`

### Vector eval

```powershell
python -m src.eval_cli --index-vector .\eval\vindex_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever vector --top-k 5
```

### BM25 eval

```powershell
python -m src.eval_cli --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever bm25 --top-k 5
```

### Fusion eval

```powershell
python -m src.eval_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever fusion --fusion-method rrf --fusion-top-n 5 --top-k 5
```

### Rerank в eval

```powershell
python -m src.eval_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever fusion --rerank --rerank-top-n 10 --proximity-window 5 --top-k 5
```

### Формат retrieval eval dataset

Ожидается JSONL, где каждая строка выглядит так:

```json
{"query":"payment invoice","relevant":[{"source":"a.txt","idx":0}]}
```

Правила:

- `query` - непустая строка
- `relevant` - непустой список объектов
- каждый объект в `relevant` должен содержать `source` и `idx`
- `idx` должен быть неотрицательным `int`

## Answer evaluation

`eval_answer_cli.py` оценивает качество ответа RAG по двум метрикам:

- `contains_rate`
- `no_info_accuracy`

### Пример запуска

```powershell
python -m src.eval_answer_cli --index .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_answer_small.jsonl --retriever bm25 --llm extract --top-k 5
```

### Формат answer eval dataset

Ожидается JSONL, где каждая строка выглядит так:

```json
{"query":"payment invoice","expected_contains":["invoice","payment"],"expected_mode":"answer"}
```

или так:

```json
{"query":"totally unrelated query","expected_contains":[],"expected_mode":"no_info"}
```

Правила:

- `expected_mode` должен быть `answer` или `no_info`
- `query` должна быть непустой
- для `expected_mode=answer` список `expected_contains` должен быть непустым

## Compare retrievers

`compare_retrievers_cli.py` сравнивает:

- `vector`
- `bm25`
- `rerank_vector`
- `rerank_bm25`
- `fusion`

Метрики в отчёте:

- `recall_mean`
- `mrr_mean`
- `n`

### Пример запуска

```powershell
python -m src.compare_retrievers_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_small.jsonl --top-k 5
```

## Compare pipelines

`compare_pipelines_cli.py` сравнивает end-to-end пайплайны на answer-eval датасете и отдаёт:

- `vector`
- `bm25`
- `rerank_vector`
- `rerank_bm25`
- `fusion`

Метрики в отчёте:

- `contains_rate`
- `no_info_accuracy`
- `n`

### Пример запуска

```powershell
python -m src.compare_pipelines_cli --index-vector .\eval\vindex_v1_rus.pkl --index-bm25 .\eval\bm25_index_v1_rus.pkl --dataset .\eval\eval_answer_small.jsonl --top-k 5
```

## Chunk experiments

`chunk_experiments_cli.py` прогоняет BM25 retrieval на нескольких chunking config и возвращает список отчётов.

### Пример запуска

```powershell
python -m src.chunk_experiments_cli --docs .\eval\docs\v1_rus --dataset .\eval\eval_small.jsonl --top-k 5 --chunking-config "((200, 40), (400, 80), (800, 120))"
```

### Что возвращается

Для каждой конфигурации:

- `chunk_size`
- `overlap`
- `k`
- `retriever`
- `recall_mean`
- `mrr_mean`
- `n`

## Regression gate

`regress_cli.py` проверяет retrieval-метрики на порог:

- `min-recall`
- `min-mrr`

### Пример запуска

```powershell
python -m src.regress_cli --index-bm25 .\eval\bm25_index_v1_rus.pkl --index-vector .\eval\vindex_v1_rus.pkl --dataset .\eval\eval_small.jsonl --retriever fusion --rerank --rerank-top-n 10 --proximity-window 5 --top-k 5 --min-recall 0.7 --min-mrr 0.5
```

### Связь с `scripts/check.ps1`

`scripts/check.ps1` запускает regression gate как часть общей проверки репозитория. Это и есть основной quality gate проекта.

## Текущий workflow

1. Построить индекс:
   - `python -m src.vector_search --docs ... --index-out ...`
   - или `python -m src.rag_cli --docs ... --index-type bm25 --index-out ...`
2. Прогнать поиск / RAG:
   - `python -m src.rag_cli --index-vector ... --query ...`
3. Прогнать retrieval eval:
   - `python -m src.eval_cli --index-vector ... --dataset ...`
4. Сравнить стратегии:
   - `python -m src.compare_retrievers_cli ...`
   - `python -m src.compare_pipelines_cli ...`
5. Прогнать chunk experiments:
   - `python -m src.chunk_experiments_cli ...`
6. Прогнать regression gate:
   - `python -m src.regress_cli ...`
   - или весь quality gate через `.\scripts\check.ps1`

## Ограничения и заметки

- Это учебный проект, а не production toolkit.
- Метрики зависят от текущих datasets и chunking config.
- Сравнения здесь инженерные: они помогают быстро смотреть на поведение пайплайнов, а не заменяют академический benchmark.
- В репозитории нет отдельного `requirements.txt`, поэтому окружение собирается вручную.
