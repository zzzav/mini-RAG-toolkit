# День 6 — vector_search: “эмбеддинги” локально через sklearn

## Что сделали
- Реализовали `src/vector_search.py`:
  - эмбеддинги через `sklearn.TfidfVectorizer`
  - similarity через `cosine_similarity`
  - `VectorIndex` (vectorizer, matrix, chunks)
  - `search(top_k)`
  - `save/load` через pickle
  - `min_df=1`

## Тесты
- `tests/test_vector_search.py`:
  - top‑1 для запроса “invoice payment”
  - save/load через `tmp_path`
  - no-results для запроса без слов

## Результат
Получили воспроизводимый векторный retrieval без внешних API и без нейросетевых эмбеддингов.
