"""
eval/metrics.py
---------------
Evaluation metrics, matching the paper's setup exactly.

Primary metric: nDCG@10 (Normalised Discounted Cumulative Gain at rank 10)
  — used in Tables 1, 2, 3, 4 of the SMR paper
  — computed via pytrec_eval, which is the standard tool in IR research

Secondary metrics we track (not in the paper, useful for your report):
  - REFINE / RERANK / STOP action counts and ratios (replicates Table 3)
  - Average steps per query
  - Step distribution (how many queries stop at step 1, 2, 3, ...)

nDCG@k intuition:
  - A perfect ranking (all relevant docs at the top) scores 1.0
  - A random ranking scores near 0.0
  - Documents at rank 1 contribute more than at rank 10 (log discount)
  - "Normalised" means we divide by the ideal score, so it's always [0,1]
"""

from collections import Counter, defaultdict
from typing import Dict, List
import pytrec_eval


# ─────────────────────────────────────────────────────────────────────────────
# nDCG@k via pytrec_eval
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndcg(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute nDCG@k for every query and return per-query scores + mean.

    Args:
        run   : {query_id: {doc_id: score}}   — your system's ranked output
        qrels : {query_id: {doc_id: relevance}} — ground truth from dataset
        k     : cutoff rank (default 10, matching paper)

    Returns:
        {query_id: ndcg_score, ..., "mean": float}
    """
    # pytrec_eval expects qrels with int relevance
    qrels_int = {
        qid: {did: int(rel) for did, rel in docs.items()}
        for qid, docs in qrels.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_int, {f"ndcg_cut_{k}"}
    )
    scores = evaluator.evaluate(run)

    result = {qid: v[f"ndcg_cut_{k}"] for qid, v in scores.items()}
    result["mean"] = sum(result.values()) / len(result) if result else 0.0
    return result


def state_to_run_entry(query_id: str, state) -> Dict[str, Dict[str, float]]:
    """
    Convert a final RetrievalState into a run entry for compute_ndcg.
    Scores are inverted ranks (10, 9, 8, ...) so pytrec_eval gets a
    proper ranking even when BM25 scores are 0 for some docs.
    """
    n = len(state.documents)
    return {
        query_id: {
            doc_id: float(n - rank)
            for rank, doc_id in enumerate(state.documents)
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Action distribution analysis  (replicates Table 3 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_traces(traces: List[Dict]) -> Dict:
    """
    Aggregate trace statistics across all queries.

    Args:
        traces: list of trace dicts from run_smr() or run_llm_smr()

    Returns:
        {
            "mean_steps"        : float,
            "refine_ratio"      : float,   # fraction of actions that were REFINE
            "rerank_ratio"      : float,
            "stop_reasons"      : Counter,
            "step_distribution" : {step_count: num_queries},
        }
    """
    all_actions  = []
    step_counts  = []
    stop_reasons = Counter()

    for trace in traces:
        all_actions.extend(trace.get("actions", []))
        step_counts.append(trace.get("steps", 0))
        stop_reasons[trace.get("stop_reason", "unknown")] += 1

    action_counts = Counter(all_actions)
    total_actions = len(all_actions) or 1   # avoid division by zero

    step_distribution = Counter(step_counts)

    return {
        "mean_steps"       : sum(step_counts) / len(step_counts) if step_counts else 0,
        "refine_ratio"     : action_counts.get("REFINE", 0) / total_actions,
        "rerank_ratio"     : action_counts.get("RERANK", 0) / total_actions,
        "action_counts"    : dict(action_counts),
        "stop_reasons"     : dict(stop_reasons),
        "step_distribution": dict(step_distribution),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict]) -> None:
    """
    Print a comparison table matching the paper's style.

    Args:
        results: {"method_name": {"ndcg": float, "mean_steps": float, ...}}
    """
    print(f"\n{'Method':<20} {'nDCG@10':>10} {'Avg Steps':>11} {'REFINE%':>9} {'RERANK%':>9}")
    print("─" * 63)
    for name, r in results.items():
        ndcg        = r.get("ndcg", 0.0)
        mean_steps  = r.get("mean_steps", 0.0)
        refine_pct  = r.get("refine_ratio", 0.0) * 100
        rerank_pct  = r.get("rerank_ratio", 0.0) * 100
        print(f"{name:<20} {ndcg:>10.4f} {mean_steps:>11.2f} {refine_pct:>8.1f}% {rerank_pct:>8.1f}%")


if __name__ == "__main__":
    # Minimal sanity check
    qrels = {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 1}}
    run   = {"q1": {"d1": 2.0, "d2": 1.0}, "q2": {"d3": 1.5, "d4": 0.5}}

    scores = compute_ndcg(run, qrels)
    print(f"nDCG@10 scores: {scores}")
    # q1: d1 is rank 1, relevant → perfect → 1.0
    # q2: d3 is rank 1, relevant → perfect → 1.0
    # mean: 1.0
