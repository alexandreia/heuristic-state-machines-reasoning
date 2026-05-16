# Heuristic State Machine Reasoning for Information Retrieval

**Course project** — inspired by [Lee et al., 2025 — *From Token to Action: State Machine Reasoning to Mitigate Overthinking in IR*](https://aclanthology.org/2025.findings-emnlp.xxx)

---

## What This Project Is

Standard retrieval systems retrieve a fixed number of documents and stop. The SMR paper (EMNLP 2025) shows that a **state machine**—cycling through structured *Refine / Rerank / Stop* actions—can improve retrieval quality while using 74% fewer tokens than Chain-of-Thought reasoning.

This project asks: **can we get similar gains with heuristic rules instead of an LLM?**

We replace the LLM policy with deterministic rules based on BM25 scores and query length, and replace LLM-based query rewriting with classical Pseudo-Relevance Feedback (Rocchio, 1971). No training, no GPU, no API key required.

---

## How It Works

Each query goes through a state machine loop:

```
Query q₀
    │
    ▼
BM25 Retrieval → D₀ (top-10 docs)
    │
    ▼ State s₀ = (q₀, D₀)
    │
┌───────────────────────────────────┐
│  Heuristic Policy reads state     │
│                                   │
│  score ≥ 8.0?      → STOP        │
│  query < 3 words?  → REFINE      │
│  score < 1.5?      → REFINE      │
│  scores bunched?   → RERANK      │
│  otherwise         → STOP        │
│                                   │
│  REFINE → expand query via PRF    │
│           retrieve new docs       │
│           merge with existing     │
│                                   │
│  RERANK → BM25 rescore all docs  │
│           against current query   │
│                                   │
│  Equivalent state? → STOP early  │
└───────────────────────────────────┘
    │
    ▼
Final (query, ranked docs) → nDCG@10
```

This maps directly to the paper:

| Paper (SMR)         | This project              |
|---------------------|---------------------------|
| LLM policy          | Heuristic rules           |
| LLM query rewriting | Pseudo-Relevance Feedback |
| LLM reranking       | BM25 rescoring            |
| State equivalence   | Identical (unchanged)     |
| Hallucination guard | Identical (unchanged)     |

---

## Project Structure

```
.
├── requirements.txt
│
├── data/
│   └── loader.py          # loads SciFact from BEIR (corpus, queries, qrels)
│
├── retrieval/
│   └── bm25.py            # BM25 index + retrieve() + score()
│
├── smr/
│   ├── state.py           # RetrievalState dataclass + equivalence check
│   ├── policy.py          # HeuristicPolicy — 4 ordered rules
│   ├── actions.py         # refine() via PRF,  rerank() via BM25 rescore
│   └── engine.py          # main SMR loop
│
├── llm/                   # optional — LLM-based SMR (paper's original)
│   ├── client.py          # unified OpenAI / Ollama client
│   ├── prompt.py          # paper's exact Table 5 prompt
│   └── engine.py          # LLM SMR loop with JSON parsing + retry logic
│
├── eval/
│   └── metrics.py         # nDCG@10 via pytrec_eval + action distribution
│
├── results/               # experiment outputs (JSON)
│
└── run_experiment.py      # main script — BM25 vs Heuristic SMR
```

---

## Dataset

**SciFact** from the [BEIR benchmark](https://github.com/beir-cellar/beir)

| Property | Value |
|----------|-------|
| Corpus   | 5,183 biomedical paper abstracts |
| Queries  | 300 test queries |
| Task     | Verify scientific claims against the corpus |
| Metric   | nDCG@10 (same as the paper) |

Downloaded automatically on first run.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) download NLTK data if using extended PRF
python -c "import nltk; nltk.download('wordnet')"
```

---

## Running Experiments

### Quick sanity check (5 queries, verbose trace)
```bash
python run_experiment.py --n 5 --verbose
```

### Full experiment — BM25 baseline vs Heuristic SMR
```bash
python run_experiment.py
```

### Tune hyperparameters
```bash
python run_experiment.py \
  --stop_high 10.0 \   # BM25 score threshold to stop early
  --stop_low  1.0  \   # BM25 score threshold to trigger REFINE
  --max_steps 6    \   # max reasoning steps per query
  --n_feedback 2   \   # PRF: how many top docs to use for expansion
  --n_terms 4          # PRF: how many new terms to add to query
```

### With an LLM (paper's original approach)
```bash
# Option A: Ollama (local, free)
brew install ollama
ollama serve                      # separate terminal
ollama pull qwen2.5:7b            # ~5GB, closest to paper's model

python run_experiment.py --n 20 --llm --backend ollama --model qwen2.5:7b

# Option B: OpenAI (cloud)
export OPENAI_API_KEY=sk-...
python run_experiment.py --n 20 --llm --backend openai --model gpt-4o-mini
```

### Test individual modules
```bash
python -m data.loader        # check corpus loads correctly
python -m retrieval.bm25     # test retrieval on a sample query
python -m smr.engine         # trace a single query through the state machine
python -m eval.metrics       # sanity check on nDCG computation
```

---

## Output

Results are printed as a comparison table and saved to `results/results.json`:

```
Method               nDCG@10   Avg Steps   REFINE%   RERANK%
───────────────────────────────────────────────────────────────
BM25 Baseline         0.6650        1.00      0.0%      0.0%
Heuristic SMR         0.68xx        x.xx     xx.x%     xx.x%
LLM SMR (qwen2.5:7b)  0.73xx       xx.xx     xx.x%     xx.x%
```

*(Exact numbers depend on your hyperparameters and model.)*

---

## Key Concepts

**BM25** — sparse retriever based on term frequency and inverse document frequency. Used as the baseline and as the underlying retriever in both SMR variants.

**Pseudo-Relevance Feedback (PRF)** — assumes the top-k retrieved documents are relevant, extracts their most distinctive terms, and appends them to the query. This is the heuristic REFINE action.

**State equivalence** — if the query and document ranking are identical to the previous step, the system is in a loop and stops. This is the core anti-redundancy mechanism from the paper (Section 3.1).

**nDCG@10** — normalised Discounted Cumulative Gain at rank 10. Measures whether relevant documents appear near the top of the ranking. 1.0 = perfect, 0.0 = worst. Standard metric in IR research.

---

## References

- Lee et al. (2025). *From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval.* EMNLP 2025 Findings.
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of IR.* NeurIPS 2021.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Rocchio (1971). *Relevance Feedback in Information Retrieval.* The SMART Retrieval System.
