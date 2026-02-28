# День 4 — simple_search: mini‑RAG без векторов (bag-of-words)

## Что сделали
- Реализовали `src/simple_search.py` — простой поиск по документам без эмбеддингов:
  - загрузка docs
  - разбиение на чанки `chunk_text(chunk_size/overlap)` с валидацией
  - `Chunk(source, idx, text)`
  - скоринг `score_chunk` по словам (через `set`)
  - stop_words
  - очистка пунктуации
  - `boost_window` и контекст ±window вокруг совпадений
  - стабильная сортировка (-score, source, idx)
  - JSON-вывод результатов

## Тесты
- `tests/test_chunking.py` на разбиение и корректность overlap/валидаций.

## Результат
Появилась базовая retrieval-часть “мини‑RAG”: запрос → топ чанков → контекст.
