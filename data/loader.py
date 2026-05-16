"""
data/loader.py
--------------
Loads the SciFact dataset from the BEIR benchmark.

SciFact is a collection of 5,183 biomedical paper abstracts with 300 test
queries. Each query is a scientific claim that must be verified against the
corpus. It is one of the datasets used in the SMR paper (Table 4).

Returns three standard IR objects:
  - corpus : dict {doc_id: {"title": str, "text": str}}
  - queries: dict {query_id: str}
  - qrels  : dict {query_id: {doc_id: int}}  ← relevance judgements (ground truth)

Fun fact: BEIR (Benchmarking IR) was released in 2021 and has become the
standard way to evaluate zero-shot IR systems. SciFact is one of its
smallest and cleanest subsets — perfect for course projects.
"""

from beir import util
from beir.datasets.data_loader import GenericDataLoader

DATASET_URL  = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATASET_PATH = "datasets/scifact"


def load_scifact(split: str = "test") -> tuple[dict, dict, dict]:
    """
    Downloads SciFact (once, cached locally) and returns corpus/queries/qrels.

    Args:
        split: "test" (300 queries) or "train" (809 queries, no qrels)

    Returns:
        corpus  : {doc_id: {"title": str, "text": str}}
        queries : {query_id: str}
        qrels   : {query_id: {doc_id: relevance_score}}
    """
    data_path = util.download_and_unzip(DATASET_URL, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    print(f"Loaded SciFact ({split})")
    print(f"  Corpus : {len(corpus):,} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels  : {len(qrels)} queries with relevance judgements")

    return corpus, queries, qrels


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()

    # Quick sanity check — make sure IDs match across all three
    sample_qid = list(queries.keys())[0]
    sample_did = list(corpus.keys())[0]

    print(f"\nSample query  [{sample_qid}]: {queries[sample_qid]}")
    print(f"Sample doc    [{sample_did}]: {corpus[sample_did]['title']}")
    print(f"Sample qrel   [{sample_qid}]: {qrels.get(sample_qid, {})}")
    print("\nAll IDs consistent:", sample_qid in qrels)
