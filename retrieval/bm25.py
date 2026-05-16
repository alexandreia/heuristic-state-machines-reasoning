"""
retrieval/bm25.py
-----------------
BM25 retriever built on top of rank_bm25 (BM25Okapi).

BM25 (Best Match 25) is the gold-standard sparse retriever used in the SMR
paper as the baseline (Tables 1, 2, 4). It scores documents by term frequency
and inverse document frequency, penalising very long documents.

BM25 formula for a term t in document d:
    score(t,d) = IDF(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

Where:
  - IDF(t)    : how rare the term is across the corpus
  - tf(t,d)   : how often the term appears in d
  - k1 = 1.5  : term frequency saturation (diminishing returns)
  - b  = 0.75 : document length normalisation

Fun fact: BM25 was published in 1994 by Robertson & Walker. It still
outperforms many neural models on keyword-heavy queries and is used inside
Elasticsearch by default.
"""

import re
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi


STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "and", "to", "for", "it",
    "with", "that", "are", "was", "on", "at", "by", "from", "or",
    "be", "this", "which", "have", "has", "its", "their", "they",
    "we", "our", "not", "but", "as", "if", "so", "do", "did",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-word chars, remove stopwords and short tokens."""
    return [
        w for w in re.findall(r"\w+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi with a clean interface that mirrors
    what the SMR paper's retriever does:
      - index(corpus) once
      - retrieve(query, k) many times
    """

    def __init__(self):
        self.doc_ids: List[str] = []
        self.bm25: BM25Okapi = None
        self._corpus: Dict[str, dict] = {}

    def index(self, corpus: Dict[str, dict]) -> None:
        """
        Build the BM25 index from the BEIR corpus dict.

        Args:
            corpus: {doc_id: {"title": str, "text": str}}
        """
        self._corpus = corpus
        self.doc_ids = list(corpus.keys())

        tokenized_docs = [
            tokenize(corpus[doc_id]["title"] + " " + corpus[doc_id]["text"])
            for doc_id in self.doc_ids
        ]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"BM25 index built: {len(self.doc_ids):,} documents")

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Return top-k (doc_id, bm25_score) pairs for a query.

        Args:
            query : the query string
            k     : how many documents to return

        Returns:
            list of (doc_id, score) sorted by score descending
        """
        if self.bm25 is None:
            raise RuntimeError("Call index() before retrieve()")

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i], reverse=True)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]

    def score(self, query: str, doc_id: str) -> float:
        """Score a single document against a query. Used during RERANK."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idx = self.doc_ids.index(doc_id)
        return float(scores[idx])

    def get_doc(self, doc_id: str) -> dict:
        """Fetch raw document content by ID."""
        return self._corpus.get(doc_id, {})


if __name__ == "__main__":
    from data.loader import load_scifact

    corpus, queries, _ = load_scifact()
    retriever = BM25Retriever()
    retriever.index(corpus)

    sample_query = list(queries.values())[0]
    results = retriever.retrieve(sample_query, k=5)

    print(f"\nQuery: {sample_query}")
    print("\nTop 5 results:")
    for rank, (doc_id, score) in enumerate(results, 1):
        title = corpus[doc_id]["title"]
        print(f"  {rank}. [{doc_id}] score={score:.3f}  {title[:70]}")
