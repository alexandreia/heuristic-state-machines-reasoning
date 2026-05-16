"""
smr/engine.py
-------------
The main SMR loop. Accepts a refine_mode parameter so you can
compare PRF vs synonym expansion in the same experiment.
"""

from typing import Dict, List, Optional, Tuple
from smr.state import RetrievalState
from smr.policy import HeuristicPolicy, REFINE, RERANK, STOP
from smr.actions import refine_prf, refine_synonyms, rerank


def run_smr(
    query: str,
    retriever,
    corpus: Dict[str, dict],
    policy: Optional[HeuristicPolicy] = None,
    top_k: int = 10,
    max_steps: int = 8,
    n_feedback_docs: int = 3,
    n_expansion_terms: int = 5,
    refine_mode: str = "synonyms",     # "synonyms" or "prf"
    verbose: bool = False,
) -> Tuple[RetrievalState, Dict]:
    """
    Run the heuristic SMR loop for a single query.

    Args:
        refine_mode : "synonyms" — WordNet expansion (recommended)
                      "prf"      — Pseudo-Relevance Feedback (original)
    """
    if policy is None:
        policy = HeuristicPolicy()

    # Choose refine function based on mode
    if refine_mode == "synonyms":
        refine_fn = lambda state: refine_synonyms(
            state, retriever, corpus,
            n_total_synonyms=n_expansion_terms,
            top_k=top_k,
        )
    else:
        refine_fn = lambda state: refine_prf(
            state, retriever, corpus,
            n_feedback_docs=n_feedback_docs,
            n_expansion_terms=n_expansion_terms,
            top_k=top_k,
        )

    # ── Initialise ────────────────────────────────────────────────────────
    initial_results = retriever.retrieve(query, k=top_k)
    state = RetrievalState(
        query=query,
        documents=[d for d, _ in initial_results],
        scores={d: s for d, s in initial_results},
    )
    previous_state = None
    trace = {
        "steps": 0,
        "actions": [],
        "queries": [query],
        "stop_reason": "max_steps",
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Query: '{query}'")
        print(f"Refine mode: {refine_mode}")
        print(f"Initial top: {state.documents[0] if state.documents else '—'} "
              f"(score={state.top_score():.2f})")

    # ── Main loop ─────────────────────────────────────────────────────────
    for step in range(max_steps):
        action = policy.select_action(state, previous_state)
        trace["actions"].append(action)
        trace["steps"] = step + 1

        if verbose:
            print(f"Step {step+1}: {action}  "
                  f"(top_score={state.top_score():.2f}, "
                  f"words={len(state.query.split())})")

        if action == STOP:
            if previous_state and state == previous_state:
                trace["stop_reason"] = "equivalent_state"
            elif state.top_score() >= policy.stop_high:
                trace["stop_reason"] = "high_score"
            else:
                trace["stop_reason"] = "policy_stop"
            break

        previous_state = state

        if action == REFINE:
            state = refine_fn(state)
            trace["queries"].append(state.query)
            if verbose:
                added = len(state.query) - len(previous_state.query)
                print(f"  → '{state.query[:80]}'  (+{added} chars)")

        elif action == RERANK:
            state = rerank(state, retriever)

    else:
        trace["stop_reason"] = "max_steps"

    if verbose:
        print(f"Final top-3: {state.documents[:3]}")

    return state, trace
