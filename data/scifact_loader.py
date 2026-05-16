#!/usr/bin/env python3

"""
data/scifact_loader.py
----------------------
Încarcă datele SciFact de pe HuggingFace.

- Queries: din repo-ul echipei (sample fix de 150, seed=0)
- Corpus + qrels: direct din BeIR/scifact (sursă oficială)
"""

from datasets import load_dataset


QUERIES_REPO = "andreiaalexa/adaptive-rag-scifact-150"
CORPUS_REPO  = "BeIR/scifact"
QRELS_REPO   = "BeIR/scifact-qrels"


def load_queries() -> list[dict]:
    """
    Încarcă cele 150 queries din repo-ul echipei de pe HuggingFace.
    Returnează: [{"query_id": ..., "text": ...}, ...]
    """
    ds = load_dataset(QUERIES_REPO, split="queries")
    return [{"query_id": row["_id"], "text": row["text"]} for row in ds]


def load_corpus() -> list[dict]:
    """
    Încarcă toate documentele din SciFact (5183 articole).
    Returnează: [{"doc_id": ..., "title": ..., "text": ...}, ...]
    """
    ds = load_dataset(CORPUS_REPO, "corpus", split="corpus")
    return [{"doc_id": row["_id"], "title": row["title"], "text": row["text"]} for row in ds]


def load_qrels() -> dict[str, list[str]]:
    """
    Încarcă etichetele gold: care document e relevant pentru care query.
    Returnează: {"query_id": ["doc_id", ...], ...}
    """
    ds = load_dataset(QRELS_REPO, split="test")
    qrels = {}
    for row in ds:
        if row["score"] > 0:
            qrels.setdefault(row["query-id"], []).append(row["corpus-id"])
    return qrels


if __name__ == "__main__":
    print("Încarc queries...")
    queries = load_queries()
    print(f"  {len(queries)} queries")
    print(f"  Exemplu: {queries[0]}")

    print("\nÎncarc corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus)} documente")
    print(f"  Exemplu: {corpus[0]['title']}")

    print("\nÎncarc qrels...")
    qrels = load_qrels()
    print(f"  {len(qrels)} queries cu documente relevante")

    print("\n✅ Loader OK!")
