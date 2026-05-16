# Adaptive RAG for Token-Efficiency

**Course:** Language Technology / Information Retrieval
**Dataset:** SciFact (BEIR)
**Model:** Mistral 7B via Ollama (local)

---

## Motivation

Most RAG (Retrieval-Augmented Generation) systems retrieve a fixed number of documents — say the top 10 — and pass them all to the LLM, regardless of how complex the query is. This wastes tokens.

**Our question:** Can a simple heuristic controller decide *how many* documents to retrieve per query, saving tokens without hurting answer quality?

This is inspired by [SMR (Lee et al., 2025)](https://arxiv.org/abs/2505.xxxxx), which reduces token usage by 74.4% using a state machine. We investigate whether a lightweight, training-free alternative achieves similar efficiency gains.

---

## Research Questions

1. Does adaptive document retrieval reduce prompt token usage compared to fixed-k retrieval?
2. Does token reduction come at the cost of answer quality (Token F1) or retrieval quality (nDCG@10)?

---

## Pipeline

```
Query
  │
  ▼
HeuristicController.decide(query, top_score)
  │ returns k (number of docs to retrieve)
  ▼
DenseRetriever.retrieve(query, k)
  │ returns top-k documents from SciFact corpus
  ▼
OllamaLLM.ask(query, docs)
  │ returns answer + prompt token count
  ▼
Evaluation: Token F1, nDCG@10, avg prompt tokens
```

---

## Experimental Configurations

| Config | Description | k |
|---|---|---|
| `no_retrieval` | Closed-book baseline — Mistral answers alone | 0 |
| `fixed_5` | Always retrieve 5 documents | 5 |
| `fixed_10` | Always retrieve 10 documents | 10 |
| `heuristic` | Adaptive: based on query length + relevance score | 0–10 |

### Heuristic Controller Rules

**Rule 1 (query length):** short queries (≤6 words) → k=3; long queries → k=8

**Rule 2 (relevance score, inspired by SMR's Stop action):** if the top document's cosine similarity score is below 0.4, retrieve nothing (k=0) — the corpus likely doesn't contain a useful answer.

---

## Evaluation Metrics

**Answer quality:**
- **Token F1** — token overlap between model answer and gold document text

**Retrieval quality:**
- **nDCG@10** — normalized Discounted Cumulative Gain; measures whether relevant documents appear high in the ranked list

**Token efficiency:**
- **Avg prompt tokens** — average `prompt_eval_count` from Ollama per query
- **Token savings %** — reduction vs `fixed_10` baseline

---

## Dataset

- **Source:** [SciFact](https://huggingface.co/datasets/BeIR/scifact) via HuggingFace (BeIR benchmark)
- **Corpus:** 5,183 biomedical abstracts
- **Evaluation queries:** 150-query fixed sample — [andreiaalexa/adaptive-rag-scifact-150](https://huggingface.co/datasets/andreiaalexa/adaptive-rag-scifact-150)
- **Seed:** `0` (fixed for reproducibility)

---

## Code Structure

```
project/
├── data/
│   └── scifact_loader.py       # loads queries, corpus, qrels from HuggingFace
├── retrieval/
│   └── dense_retriever.py      # sentence-transformers encoder + cosine search
├── budget/
│   └── heuristic.py            # HeuristicController, FixedController, NoRetrievalController
├── llm/
│   └── ollama_client.py        # calls Mistral 7B via Ollama REST API
├── eval/
│   └── metrics.py              # token_f1, macro_f1, ndcg_at_k, mean_ndcg
├── final_script/
│   └── run_pilot.py            # orchestrates full experiment, saves results/final.json
├── prepare_dataset.py          # samples + uploads queries to HuggingFace
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (separate terminal)
ollama serve
ollama pull mistral:7b-instruct

# Quick test (5 queries)
python final_script/run_pilot.py --n 5

# Full experiment (150 queries, custom params)
python final_script/run_pilot.py --fixed_k1 5 --fixed_k2 10 --k_short 3 --k_long 8 --threshold 6
```

Results are saved to `results/final.json`.

---

## Team Roles

| Person | Code | Report Section |
|---|---|---|
| P1 | `retrieval/`, `data/scifact_loader.py` | Background & Related Work |
| P2 | `budget/heuristic.py` | Method (controller) |
| P3 | `compress/` | Method (compression) |
| P4 | `llm/`, `eval/`, `scripts/` | Experiments & Results |

---

## Key References

- Lewis et al., *RAG for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
- Liu et al., *Lost in the Middle*, TACL 2023
- Jeong et al., *Adaptive-RAG*, NAACL 2024
- Lee et al., *State Machine Reasoning (SMR)*, 2025
