"""
retrieval/dense_retriever.py
-----------------------------
Retriever dens bazat pe sentence-transformers.

Encodează toate documentele O SINGURĂ DATĂ și salvează pe disc (cache).
La fiecare query, encodează întrebarea și găsește cele mai similare documente.
"""

import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME  = "all-MiniLM-L6-v2"
CACHE_PATH  = "data/embeddings.pkl"


class DenseRetriever:

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.corpus = []        # lista de documente (dicts cu _id, title, text)
        self.embeddings = None  # numpy array: (nr_documente, 384)

    def index(self, corpus: list[dict], cache_path: str = CACHE_PATH):
        """
        Encodează corpus-ul și salvează embeddings pe disc.
        La rulări ulterioare, încarcă din cache — nu re-encodează.

        Args:
            corpus: lista de documente din load_corpus()
            cache_path: unde să salveze/încarce embeddings
        """
        self.corpus = corpus

        if Path(cache_path).exists():
            print(f"Încarc embeddings din cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self.embeddings = pickle.load(f)
            return

        print(f"Encodez {len(corpus)} documente... (doar prima dată)")
        texts = [f"{doc['title']} {doc['text']}" for doc in corpus]

        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True  # necesar pentru cosine similarity via dot product
        )

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"Embeddings salvate în: {cache_path}")

    def retrieve(self, query: str, k: int) -> tuple[list[dict], float]:
        """
        Găsește cele mai relevante k documente pentru o întrebare.

        Args:
            query: textul întrebării
            k: câte documente să returneze (setat de controller)

        Returns:
            (lista de k documente, scorul cosinus al celui mai bun document)
        """
        if k == 0:
            return [], 0.0

        query_vec = self.model.encode(query, normalize_embeddings=True)

        # dot product = cosine similarity (vectorii sunt normalizați)
        scores = self.embeddings @ query_vec

        top_k_indices = np.argsort(scores)[::-1][:k]
        top_score     = float(scores[top_k_indices[0]])

        return [self.corpus[i] for i in top_k_indices], top_score


# ── Test rapid ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.scifact_loader import load_corpus, load_queries

    print("Încarc corpus...")
    corpus = load_corpus()

    retriever = DenseRetriever()
    retriever.index(corpus)

    queries = load_queries()
    test_query = queries[0]["text"]
    print(f"\nQuery: {test_query}")

    results = retriever.retrieve(test_query, k=3)
    print(f"\nTop 3 documente:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc['title']}")

    print("\n✅ Retriever OK!")
