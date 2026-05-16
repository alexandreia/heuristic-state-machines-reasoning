"""
smr/engine.py
-------------
The main SMR loop — wires state, policy, and actions together.

This is the core algorithm from the paper (Section 3), implemented with
heuristic rules instead of an LLM. The structure is identical to the paper:

  1. Initialise state s0 = (q0, D0)
  2. Loop up to max_steps:
       a. Check equivalence (redundancy guard)
       b. Policy selects action
       c. Execute action → new state
  3. Return final state

The function also returns a trace dict so you can analyse what the system
did — action distributions, number of steps, etc. (replicates Table 3 of
the paper).
"""

from typing import Dict, List, Optional, Tuple
from smr.state import RetrievalState
from smr.policy import HeuristicPolicy, REFINE, RERANK, STOP
from smr.actions import refine, rerank


def run_smr(
    query: str,
    retriever,                          # BM25Retriever
    corpus: Dict[str, dict],
    policy: Optional[HeuristicPolicy] = None,
    top_k: int = 10,
    max_steps: int = 8,
    n_feedback_docs: int = 3,
    n_expansion_terms: int = 5,
    verbose: bool = False,
) -> Tuple[RetrievalState, Dict]:
    """
    Run the heuristic SMR loop for a single query.

    Args:
        query             : the original user query
        retriever         : BM25Retriever (already indexed)
        corpus            : raw corpus dict for PRF term extraction
        policy            : HeuristicPolicy (default params if None)
        top_k             : documents to retrieve per step
        max_steps         : hard cap on reasoning steps (paper uses 16)
        n_feedback_docs   : PRF parameter — feedback docs for REFINE
        n_expansion_terms : PRF parameter — new terms added per REFINE
        verbose           : print step-by-step trace

    Returns:
        (final_state, trace)

        final_state : RetrievalState with final (query, ranked docs)
        trace       : {
            "steps"       : int,
            "actions"     : list of action strings taken,
            "queries"     : list of query strings at each step,
            "stop_reason" : "equivalent_state" | "high_score" | "llm_stop"
                            | "max_steps"
          }
    """
    if policy is None:
        policy = HeuristicPolicy()

    # ── Step 0: Initialise ────────────────────────────────────────────────
    initial_results = retriever.retrieve(query, k=top_k)
    state = RetrievalState(
        query=query,
        documents=[d for d, _ in initial_results],
        scores={d: s for d, s in initial_results},
    )
    previous_state = None

    trace = {"steps": 0, "actions": [], "queries": [query], "stop_reason": "max_steps"}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Query: '{query}'")
        print(f"Initial top doc: {state.documents[0] if state.documents else 'none'} "
              f"(score={state.top_score():.2f})")

    # ── Main loop ─────────────────────────────────────────────────────────
    for step in range(max_steps):
        action = policy.select_action(state, previous_state)
        trace["actions"].append(action)
        trace["steps"] = step + 1

        if verbose:
            print(f"Step {step+1}: {action}  "
                  f"(top_score={state.top_score():.2f}, "
                  f"query_words={len(state.query.split())})")

        if action == STOP:
            # Determine why we stopped (for analysis)
            if previous_state is not None and state == previous_state:
                trace["stop_reason"] = "equivalent_state"
            elif state.top_score() >= policy.stop_high:
                trace["stop_reason"] = "high_score"
            else:
                trace["stop_reason"] = "policy_stop"
            break

        previous_state = state

        if action == REFINE:
            state = refine(
                state, retriever, corpus,
                n_feedback_docs=n_feedback_docs,
                n_expansion_terms=n_expansion_terms,
                top_k=top_k,
            )
            trace["queries"].append(state.query)

        elif action == RERANK:
            state = rerank(state, retriever)

    else:
        # Loop exhausted max_steps without STOP
        trace["stop_reason"] = "max_steps"

    if verbose:
        print(f"Final query:  '{state.query[:80]}'")
        print(f"Final top-3:  {state.documents[:3]}")
        print(f"Stop reason:  {trace['stop_reason']}")

    return state, trace


if __name__ == "__main__":
    from data.loader import load_scifact
    from retrieval.bm25 import BM25Retriever

    corpus, queries, qrels = load_scifact()
    retriever = BM25Retriever()
    retriever.index(corpus)

    sample_query = list(queries.values())[0]
    final_state, trace = run_smr(
        sample_query, retriever, corpus, verbose=True
    )
    print(f"\nTrace: {trace}")
