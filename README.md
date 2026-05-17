# Heuristic State Machine Reasoning for Information Retrieval

## Dataset

**SciFact** from the [BEIR benchmark](https://github.com/beir-cellar/beir)

## Key Concepts

**BM25** — sparse retriever based on term frequency and inverse document frequency. Used as the baseline and as the underlying retriever in both SMR variants.

**Pseudo-Relevance Feedback (PRF)** — assumes the top-k retrieved documents are relevant, extracts their most distinctive terms, and appends them to the query. This is the heuristic REFINE action.

**State equivalence** — if the query and document ranking are identical to the previous step, the system is in a loop and stops. This is the core anti-redundancy mechanism from the paper (Section 3.1).

**nDCG@10** — normalised Discounted Cumulative Gain at rank 10. Measures whether relevant documents appear near the top of the ranking. 1.0 = perfect, 0.0 = worst. Standard metric in IR research.
