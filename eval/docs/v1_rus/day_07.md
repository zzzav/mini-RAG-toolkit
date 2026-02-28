# День 7 — RAG Answer: контекст + prompt + (mock) ответ

## Что сделали
- Собрали пайплайн “retrieval → контекст → prompt → answer” без внешнего LLM:
  - берем top-k чанков из vector_search (или tfidf)
  - `build_context` с лимитами `max_context_chars` и `per_chunk_chars`
  - `build_prompt` по шаблону “не выдумывать”
  - `MockLLM` для детерминированного ответа (для тестов)
- CLI для запуска RAG-ответа (text/json).

## Тесты
- Тесты на build_context лимиты, prompt, mock, e2e.

## Результат
Появился “закольцованный” мини‑RAG: можно на вход дать вопрос и получить ответ с опорой на retrieved контекст.
