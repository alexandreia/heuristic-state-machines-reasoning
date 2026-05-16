"""
run_experiment.py
-----------------
Main experiment script. Compares BM25 baseline vs Heuristic SMR on SciFact.

Replicates the paper's experimental structure (Section 4):
  - BM25 baseline        → plain retrieval, no reasoning
  - Heuristic SMR        → state machine with rule-based policy

Optionally runs LLM SMR if --llm flag is passed.

Usage:
    # Quick test (5 queries, default params)
    python run_experiment.py --n 5 --verbose

    # Full experiment
    python run_experiment.py

    # Tune heuristic hyperparameters
    python run_experiment.py --stop_high 10 --stop_low 2 --max_steps 6

    # Enable LLM mode (needs Ollama running)
    python run_experiment.py --n 20 --llm --backend ollama --model qwen2.5:7b

Results saved to results/results.json
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from data.loader import load_scifact
from retrieval.bm25 import BM25Retriever
from smr.engine import run_smr
from smr.policy import HeuristicPolicy
from eval.metrics import compute_ndcg, state_to_run_entry, summarise_traces, print_results_table


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SMR experiment on SciFact")

    # Dataset
    p.add_argument("--n",          type=int,   default=None,
                   help="Number of queries to run (default: all 300)")
    p.add_argument("--split",      type=str,   default="test",
                   help="Dataset split: 'test' (default) or 'train'")

    # Retrieval
    p.add_argument("--top_k",      type=int,   default=10,
                   help="Documents to retrieve per step (default: 10)")

    # SMR hyperparameters
    p.add_argument("--max_steps",  type=int,   default=8,
                   help="Max reasoning steps per query (default: 8)")
    p.add_argument("--stop_high",  type=float, default=8.0,
                   help="BM25 score threshold for STOP-high (default: 8.0)")
    p.add_argument("--stop_low",   type=float, default=1.5,
                   help="BM25 score threshold for REFINE (default: 1.5)")
    p.add_argument("--min_words",  type=int,   default=3,
                   help="Min query words before REFINE triggers (default: 3)")
    p.add_argument("--n_feedback", type=int,   default=3,
                   help="PRF feedback docs for REFINE (default: 3)")
    p.add_argument("--n_terms",    type=int,   default=5,
                   help="PRF expansion terms for REFINE (default: 5)")

    # LLM mode (optional)
    p.add_argument("--llm",        action="store_true",
                   help="Also run LLM-based SMR (needs Ollama or OpenAI key)")
    p.add_argument("--backend",    type=str,   default="ollama",
                   choices=["ollama", "openai"],
                   help="LLM backend (default: ollama)")
    p.add_argument("--model",      type=str,   default="qwen2.5:7b",
                   help="LLM model name (default: qwen2.5:7b)")

    # Output
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-query traces")
    p.add_argument("--out",        type=str,   default="results/results.json",
                   help="Output file path")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# BM25 baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_bm25_baseline(queries, retriever, qrels, top_k):
    """
    Plain BM25: retrieve top_k docs per query, no reasoning.
    This is the direct comparison point from the paper's Tables 1-4.
    """
    print("\n── BM25 Baseline ──")
    run   = {}
    dummy_trace = {"steps": 1, "actions": ["STOP"], "stop_reason": "baseline"}
    traces = []

    for qid, query in tqdm(queries.items(), desc="BM25"):
        results = retriever.retrieve(query, k=top_k)
        run[qid] = {doc_id: score for doc_id, score in results}
        traces.append(dummy_trace)

    ndcg_scores = compute_ndcg(run, qrels)
    return ndcg_scores["mean"], traces, run


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic SMR
# ─────────────────────────────────────────────────────────────────────────────

def run_heuristic_smr(queries, retriever, corpus, qrels, args):
    """
    Heuristic SMR: state machine with rule-based policy and PRF.
    This is our main contribution — the heuristic version of the paper.
    """
    print("\n── Heuristic SMR ──")

    policy = HeuristicPolicy(
        stop_score_high=args.stop_high,
        stop_score_low=args.stop_low,
        min_query_words=args.min_words,
    )

    run    = {}
    traces = []

    for qid, query in tqdm(queries.items(), desc="SMR"):
        final_state, trace = run_smr(
            query=query,
            retriever=retriever,
            corpus=corpus,
            policy=policy,
            top_k=args.top_k,
            max_steps=args.max_steps,
            n_feedback_docs=args.n_feedback,
            n_expansion_terms=args.n_terms,
            verbose=args.verbose,
        )
        run_entry = state_to_run_entry(qid, final_state)
        run.update(run_entry)
        traces.append(trace)

    ndcg_scores = compute_ndcg(run, qrels)
    return ndcg_scores["mean"], traces, run


# ─────────────────────────────────────────────────────────────────────────────
# LLM SMR (optional)
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_smr_experiment(queries, retriever, corpus, qrels, args):
    """Optional: LLM-based SMR using the paper's exact prompt."""
    from llm.engine import run_llm_smr

    print(f"\n── LLM SMR ({args.backend} / {args.model}) ──")
    run    = {}
    traces = []

    for qid, query in tqdm(queries.items(), desc="LLM-SMR"):
        final_state, trace = run_llm_smr(
            query=query,
            retriever=retriever,
            corpus=corpus,
            backend=args.backend,
            model=args.model,
            top_k=args.top_k,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        run_entry = state_to_run_entry(qid, final_state)
        run.update(run_entry)
        traces.append(trace)

    ndcg_scores = compute_ndcg(run, qrels)
    return ndcg_scores["mean"], traces, run


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading SciFact...")
    corpus, queries, qrels = load_scifact(split=args.split)

    # Optionally limit to first n queries (for quick iteration)
    if args.n is not None:
        query_ids = list(queries.keys())[:args.n]
        queries   = {qid: queries[qid] for qid in query_ids}
        qrels     = {qid: qrels[qid] for qid in query_ids if qid in qrels}
        print(f"Running on {args.n} queries")

    # ── Build BM25 index ──────────────────────────────────────────────────
    retriever = BM25Retriever()
    retriever.index(corpus)

    # ── Run experiments ───────────────────────────────────────────────────
    all_results = {}

    # 1. BM25 baseline
    bm25_ndcg, bm25_traces, bm25_run = run_bm25_baseline(
        queries, retriever, qrels, args.top_k
    )
    all_results["BM25 Baseline"] = {
        "ndcg": bm25_ndcg,
        **summarise_traces(bm25_traces),
    }

    # 2. Heuristic SMR
    smr_ndcg, smr_traces, smr_run = run_heuristic_smr(
        queries, retriever, corpus, qrels, args
    )
    all_results["Heuristic SMR"] = {
        "ndcg": smr_ndcg,
        **summarise_traces(smr_traces),
    }

    # 3. LLM SMR (optional)
    if args.llm:
        llm_ndcg, llm_traces, llm_run = run_llm_smr_experiment(
            queries, retriever, corpus, qrels, args
        )
        all_results[f"LLM SMR ({args.model})"] = {
            "ndcg": llm_ndcg,
            **summarise_traces(llm_traces),
        }

    # ── Print results table ───────────────────────────────────────────────
    print("\n")
    print_results_table(all_results)

    delta = smr_ndcg - bm25_ndcg
    print(f"\nHeuristic SMR vs BM25: {delta*100:+.2f}% nDCG@10")

    # ── Save to JSON ──────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    save_data = {
        "args": vars(args),
        "results": all_results,
        "runs": {
            "bm25": bm25_run,
            "smr":  smr_run,
        },
    }
    with open(args.out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
