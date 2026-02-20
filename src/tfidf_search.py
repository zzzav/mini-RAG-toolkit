import argparse
import json
import math
from pathlib import Path

from src.query_normalize import DEFAULT_STOP_WORDS, normalize_query
from src.simple_search import (
    Chunk,
    build_chunks,
    load_text_files,
    save_json,
)
from src.simple_search import (
    search as simple_search,
)


#####################################################
# tokenize  получение перечня слов в рамках
#           текстового блока
#####################################################
def tokenize(text: str, use_stop_words: bool = True) -> list[str]:
    stop_words = ()
    if use_stop_words:
        stop_words = DEFAULT_STOP_WORDS
    return normalize_query(text, stop_words=stop_words)


#####################################################
# count_tf  определение частоты слов в текстовом
#           блоке
#####################################################
def count_tf(tokens: list[str]) -> dict[str, int]:
    tf: dict[str, int] = {}

    for i_token in tokens:
        tf[i_token] = tf.get(i_token, 0) + 1

    return tf


#####################################################
# build_index   построение поисковых метрик
#
#####################################################
def build_index(chunks: list[Chunk], use_stop_words: bool = True) -> dict:

    # частота слов внутри чанков
    tf_list = []
    # частота слов среди чанков
    df = {}
    # веса слов
    idf = {}
    chunk_meta_list = []

    for chunk in chunks:
        tokens = tokenize(chunk.text, use_stop_words)
        tf_counts = count_tf(tokens)
        tf_list.append(tf_counts)
        unique_words = set(tf_counts.keys())

        for w in unique_words:
            df[w] = df.get(w, 0) + 1

        chunk_meta_list.append({"source": chunk.source, "idx": chunk.idx, "text": chunk.text})

    N = len(chunks)
    for w, count in df.items():
        idf[w] = math.log((N + 1) / (count + 1)) + 1

    # поисковые метрики
    index: dict = {}
    index["chunk_meta"] = chunk_meta_list
    index["tf"] = tf_list
    index["df"] = df
    index["idf"] = idf

    return index


#####################################################
# tfidf_search  поиск с использованием индексов
#
#####################################################
def tfidf_search(
    query: str, chunks: list[Chunk], index: dict, top_k: int = 5, use_stop_words: bool = True
) -> list[tuple[float, Chunk]]:

    scored: list[tuple[float, Chunk]] = []

    q_words = tokenize(query, use_stop_words)

    for i, chunk in enumerate(chunks):
        score = 0.0
        for w in q_words:
            tf_w = index["tf"][i].get(w, 0)
            if tf_w > 0:
                score += tf_w * index["idf"].get(w, 0.0)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: (-x[0], x[1].source, x[1].idx))

    return scored[:top_k]


#####################################################
# load_index    загрузка индекса из json
#
#####################################################
def load_index(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    return data


#####################################################
# build_parser  построение парсера аргументов CLI
#
#####################################################
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Поиск на основании поисковых индексов")
    p.add_argument("--docs", type=str, required=True, help="Папка с .txt документами")
    p.add_argument("--query", type=str, help="Поисковый запрос")
    p.add_argument("--top", type=int, default=5, help="Сколько результатов показать")
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument(
        "--build-index", dest="build_index", type=str, help="Построение поискового индекса"
    )
    p.add_argument(
        "--use-index", dest="use_index", type=str, help="Использование индексов при поиске"
    )
    return p


#####################################################
# main
#
#####################################################
def main() -> None:
    args = build_parser().parse_args()
    docs = load_text_files(args.docs)
    chunks = build_chunks(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    results = []
    index: dict = {}

    if args.build_index:
        index = build_index(chunks)
        save_json(args.build_index, index)

    if args.query:
        if args.use_index:
            index = load_index(args.use_index)
            results = tfidf_search(args.query, chunks, index)
        else:
            results = simple_search(args.query, chunks, top_k=args.top)

    print(f"CHUNKS={len(chunks)} RESULTS={len(results)}")
    for score, chunk in results:
        print(f"[{score}] {chunk.source} #{chunk.idx}")


if __name__ == "__main__":
    main()
