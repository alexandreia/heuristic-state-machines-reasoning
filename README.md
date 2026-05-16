# Heuristic State Machine Reasoning for Information Retrieval

**Course project** 

---

## What This Project Is

Standard retrieval systems retrieve a fixed number of documents and stop. The SMR paper (EMNLP 2025) shows that a **state machine** cycling through *Refine / Rerank / Stop* actions can improve retrieval quality while using 74% fewer tokens than Chain-of-Thought reasoning. The paper uses an LLM to drive the state machine.

This project asks: **can we get similar gains with heuristic rules instead of an LLM?**

We replace the LLM policy with deterministic rules based on BM25 scores and query length, and replace LLM-based query rewriting with Pseudo-Relevance Feedback (Rocchio, 1971). No training, no GPU, no API key required.

---

=======
## Project Structure

```
.
├── requirements.txt
│
├── data/
│   └── loader.py          # loads SciFact (corpus, queries, qrels)
│
├── retrieval/
│   └── bm25.py            # BM25 index + retrieve() + score()
│
├── smr/
│   ├── state.py           # RetrievalState + equivalence check
│   ├── policy.py          # HeuristicPolicy — 4 ordered rules
│   ├── actions.py         # refine() via PRF, rerank() via BM25 rescore
│   └── engine.py          # main SMR loop
│
├── eval/
│   └── metrics.py         # nDCG@10 via pytrec_eval + action distribution
│
├── results/               # experiment outputs (JSON)
├── _archive/              # archived LLM wiring code (not used)
│
└── run_experiment.py      # the only file you call directly
```

---

## Dataset

**SciFact** from the [BEIR benchmark](https://github.com/beir-cellar/beir)

| Property | Value |
|----------|-------|
| Corpus   | 5,183 biomedical paper abstracts |
| Queries  | 300 test queries |
| Metric   | nDCG@10 (same as the paper) |

<<<<<<< HEAD
---

## Key Concepts

**BM25** — sparse retriever based on term frequency and inverse document frequency. Used as the baseline and as the underlying retriever in both SMR variants.

**Pseudo-Relevance Feedback (PRF)** — assumes the top-k retrieved documents are relevant, extracts their most distinctive terms, and appends them to the query. This is the heuristic REFINE action.

**State equivalence** — if the query and document ranking are identical to the previous step, the system is in a loop and stops. This is the core anti-redundancy mechanism from the paper (Section 3.1).

**nDCG@10** — normalised Discounted Cumulative Gain at rank 10. Measures whether relevant documents appear near the top of the ranking. 1.0 = perfect, 0.0 = worst. Standard metric in IR research.

=======

## References

- Lee et al. (2025). *From Token to Action: State Machine Reasoning to Mitigate Overthinking in IR.* EMNLP 2025 Findings.
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of IR.* NeurIPS 2021.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Rocchio (1971). *Relevance Feedback in Information Retrieval.*
