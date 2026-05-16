"""
run_experiment.py
-----------------
Compares three systems on SciFact:
  1. BM25 Baseline    — plain retrieval, no reasoning
  2. Heuristic SMR (PRF)      — state machine + Pseudo-Relevance Feedback
  3. Heuristic SMR (Synonyms) — state machine + WordNet synonym expansion

Usage:
    python run_experiment.py --n 5 --verbose
    python run_experiment.py
    python run_experiment.py --stop_high 20 --n_terms 4
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


def parse_args():
    p = argparse.ArgumentParser(description="Heuristic SMR experiment on SciFact")
    p.add_argument("--n",          type=int,   default=None)
    p.add_argument("--split",      type=str,   default="test")
    p.add_argument("--top_k",      type=int,   default=10)
    p.add_argument("--max_steps",  type=int,   default=8)
    p.add_argument("--stop_high",  type=float, default=15.0,
                   help="BM25 score threshold for immediate STOP (default: 15.0)")
    p.add_argument("--min_words",  type=int,   default=3)
    p.add_argument("--n_feedback", type=int,   default=3)
    p.add_argument("--n_terms",    type=int,   default=5)
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--out",        type=str,   default="results/results.json")
    return p.parse_args()


def run_bm25_baseline(queries, retriever, qrels, top_k):
    print("\n── BM25 Baseline ──")
    run = {}
    for qid, query in tqdm(queries.items(), desc="BM25"):
        results  = retriever.retrieve(query, k=top_k)
        run[qid] = {doc_id: score for doc_id, score in results}
    ndcg   = compute_ndcg(run, qrels)
    traces = [{"steps": 1, "actions": ["STOP"], "stop_reason": "baseline",
               "queries": [q]} for q in queries.values()]
    return ndcg["mean"], traces, run


def run_smr_experiment(name, refine_mode, queries, retriever, corpus, qrels, args):
    print(f"\n── {name} ──")
    policy = HeuristicPolicy(
        stop_score_high=args.stop_high,
        min_query_words=args.min_words,
    )
    run = {}; traces = []
    for qid, query in tqdm(queries.items(), desc=name):
        final_state, trace = run_smr(
            query=query,
            retriever=retriever,
            corpus=corpus,
            policy=policy,
            top_k=args.top_k,
            max_steps=args.max_steps,
            n_feedback_docs=args.n_feedback,
            n_expansion_terms=args.n_terms,
            refine_mode=refine_mode,
            verbose=args.verbose,
        )
        run.update(state_to_run_entry(qid, final_state))
        traces.append(trace)
    ndcg = compute_ndcg(run, qrels)
    return ndcg["mean"], traces, run


def main():
    args = parse_args()

    print("Loading SciFact...")
    corpus, queries, qrels = load_scifact(split=args.split)
    if args.n:
        qids    = list(queries.keys())[:args.n]
        queries = {q: queries[q] for q in qids}
        qrels   = {q: qrels[q]   for q in qids if q in qrels}
        print(f"Running on {args.n} queries")

    retriever = BM25Retriever()
    retriever.index(corpus)

    # ── Run all three systems ─────────────────────────────────────────────
    bm25_ndcg,  bm25_traces,  bm25_run  = run_bm25_baseline(
        queries, retriever, qrels, args.top_k)

    prf_ndcg,   prf_traces,   prf_run   = run_smr_experiment(
        "SMR (PRF)",      "prf",      queries, retriever, corpus, qrels, args)

    syn_ndcg,   syn_traces,   syn_run   = run_smr_experiment(
        "SMR (Synonyms)", "synonyms", queries, retriever, corpus, qrels, args)

    # ── Results ───────────────────────────────────────────────────────────
    results = {
        "BM25 Baseline":   {"ndcg": bm25_ndcg, **summarise_traces(bm25_traces)},
        "SMR (PRF)":       {"ndcg": prf_ndcg,  **summarise_traces(prf_traces)},
        "SMR (Synonyms)":  {"ndcg": syn_ndcg,  **summarise_traces(syn_traces)},
    }

    print("\n")
    print_results_table(results)
    print(f"\nSMR (PRF)      vs BM25: {(prf_ndcg - bm25_ndcg)*100:+.2f}%")
    print(f"SMR (Synonyms) vs BM25: {(syn_ndcg - bm25_ndcg)*100:+.2f}%")

    Path("results").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
