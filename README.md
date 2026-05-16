# Heuristic State Machine Reasoning for Information Retrieval

**Course project** 

---

## What This Project Is

Standard retrieval systems retrieve a fixed number of documents and stop. The SMR paper (EMNLP 2025) shows that a **state machine** cycling through *Refine / Rerank / Stop* actions can improve retrieval quality while using 74% fewer tokens than Chain-of-Thought reasoning. The paper uses an LLM to drive the state machine.

This project asks: **can we get similar gains with heuristic rules instead of an LLM?**

We replace the LLM policy with deterministic rules based on BM25 scores and query length, and replace LLM-based query rewriting with Pseudo-Relevance Feedback (Rocchio, 1971). No training, no GPU, no API key required.

---

<<<<<<< HEAD
=======
## How It Works

```
Query q₀
    │
    ▼
BM25 Retrieval → D₀ (top-10 docs)
    │
    ▼  State s = (q, D)
    │
┌───────────────────────────────────────┐
│  Heuristic Policy                     │
│                                       │
│  equivalent to previous state? → STOP │
│  BM25 score ≥ 8.0?             → STOP │
│  query < 3 words?              → REFINE│
│  BM25 score < 1.5?             → REFINE│
│  top scores bunched together?  → RERANK│
│  default                       → STOP  │
│                                       │
│  REFINE → PRF expands query           │
│           retrieve new docs           │
│           merge with existing list    │
│                                       │
│  RERANK → BM25 rescores all docs     │
│           against current query       │
└───────────────────────────────────────┘
    │
    ▼
Final (query, ranked docs) → nDCG@10
```

| Paper (LLM SMR)     | This project (Heuristic SMR) |
|---------------------|------------------------------|
| LLM policy          | 4 ordered heuristic rules    |
| LLM query rewriting | Pseudo-Relevance Feedback    |
| LLM reranking       | BM25 rescoring               |
| State equivalence   | Identical                    |
| Hallucination guard | Identical                    |

---

>>>>>>> 2d29f60 (Adding synonym based system)
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
Downloaded automatically on first run.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

```bash
# Quick sanity check — 5 queries, prints every step
python run_experiment.py --n 5 --verbose

# Full experiment (all 300 queries)
python run_experiment.py

# Tune hyperparameters
python run_experiment.py --stop_high 10 --stop_low 2 --n_terms 4 --max_steps 6
```

### All flags

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--n` | all 300 | number of queries to run |
| `--top_k` | 10 | docs retrieved per step |
| `--max_steps` | 8 | hard cap on reasoning steps |
| `--stop_high` | 8.0 | BM25 score → stop immediately |
| `--stop_low` | 1.5 | BM25 score → trigger REFINE |
| `--min_words` | 3 | query length → trigger REFINE |
| `--n_feedback` | 3 | PRF: feedback docs for expansion |
| `--n_terms` | 5 | PRF: new terms added per REFINE |
| `--verbose` | off | print per-query step trace |

### Test individual modules

```bash
python -m data.loader        # check SciFact loads correctly
python -m retrieval.bm25     # test retrieval on a sample query
python -m smr.engine         # trace one query through the state machine
python -m eval.metrics       # sanity check nDCG computation
```

---

## Output

```
Method               nDCG@10   Avg Steps   REFINE%   RERANK%
───────────────────────────────────────────────────────────────
BM25 Baseline         0.6650        1.00      0.0%      0.0%
Heuristic SMR         0.68xx        x.xx     xx.x%     xx.x%
```

Full results saved to `results/results.json`.
>>>>>>> 2d29f60 (Adding synonym based system)

---

## References

- Lee et al. (2025). *From Token to Action: State Machine Reasoning to Mitigate Overthinking in IR.* EMNLP 2025 Findings.
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of IR.* NeurIPS 2021.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Rocchio (1971). *Relevance Feedback in Information Retrieval.*
