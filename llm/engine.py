"""
llm/engine.py
-------------
LLM-based SMR loop — the paper's original approach, wired to a real model.

This module is the LLM counterpart to smr/engine.py. The state machine
structure is identical — only the policy and action execution change:
  - Policy: LLM reads state → outputs JSON action (instead of heuristic rules)
  - REFINE: LLM rewrites query (instead of PRF)
  - RERANK: LLM reorders docs (instead of BM25 rescoring)

Temperature strategy (Appendix A.3 of the paper):
  Start at T=0 (deterministic). If the output is malformed JSON,
  raise temperature by 0.1 and retry. This expands the output space
  just enough to recover from generation failures.

Usage:
    from llm.engine import run_llm_smr
    final_state, trace = run_llm_smr(
        query="do masks prevent COVID transmission",
        retriever=retriever,
        corpus=corpus,
        backend="ollama",
        model="qwen2.5:7b",
        verbose=True,
    )
"""

import json
import re
from typing import Dict, Tuple, Optional
from smr.state import RetrievalState
from llm.client import llm_call
from llm.prompt import build_prompt


# ── JSON parser ────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """
    Extract JSON from LLM output. LLMs often wrap JSON in markdown fences:
        ```json
        {...}
        ```
    This handles that gracefully.
    """
    text = raw.strip()
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def _call_with_retry(
    prompt: str,
    backend: str,
    model: str,
    max_retries: int = 3,
) -> dict:
    """
    Call the LLM and parse its JSON output.
    On failure: raise temperature by 0.1 and retry (paper Appendix A.3).
    On all retries exhausted: return STOP as a safe fallback.
    """
    temperature = 0.0  # paper default
    for attempt in range(max_retries):
        try:
            raw    = llm_call(prompt, backend=backend, model=model, temperature=temperature)
            result = _parse_json(raw)

            valid_actions = {"refine query", "re-rank", "stop"}
            if result.get("action") not in valid_actions:
                raise ValueError(f"Unknown action: {result.get('action')}")

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            temperature += 0.1
            print(f"  [Retry {attempt+1}/{max_retries}] Parse error: {e}. "
                  f"Retrying at T={temperature:.1f}")

    print("  [Fallback] All retries failed — issuing STOP")
    return {"action": "stop"}


# ── Main LLM SMR loop ──────────────────────────────────────────────────────

def run_llm_smr(
    query: str,
    retriever,
    corpus: Dict[str, dict],
    backend: str = "ollama",
    model: str = "qwen2.5:7b",
    top_k: int = 10,
    max_steps: int = 16,
    verbose: bool = False,
) -> Tuple[RetrievalState, Dict]:
    """
    Run the LLM-based SMR loop for a single query.

    This is the paper's exact algorithm. One LLM call per step does both:
      1. Policy decision (which action?)
      2. Action execution (what is the new query / new ranking?)

    Args:
        query    : original user query
        retriever: BM25Retriever (already indexed)
        corpus   : raw corpus for building prompts
        backend  : "ollama" or "openai"
        model    : model string
        top_k    : documents to retrieve per REFINE step
        max_steps: hard cap (paper uses 16)
        verbose  : print step-by-step trace

    Returns:
        (final_state, trace)
    """
    # ── Initialise ────────────────────────────────────────────────────────
    initial_results = retriever.retrieve(query, k=top_k)
    state = RetrievalState(
        query=query,
        documents=[d for d, _ in initial_results],
        scores={d: s for d, s in initial_results},
    )
    previous_state = None

    trace = {
        "steps": 0, "actions": [], "queries": [query],
        "stop_reason": "max_steps", "llm_calls": 0,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Backend: {backend} | Model: {model}")
        print(f"Query: '{query}'")

    # ── Main loop ─────────────────────────────────────────────────────────
    for step in range(max_steps):

        # Redundancy check — no LLM call needed
        if previous_state is not None and state == previous_state:
            trace["stop_reason"] = "equivalent_state"
            if verbose: print(f"Step {step+1}: Equivalent state → STOP (no LLM call)")
            break

        previous_state = state

        # ── ONE LLM call per step ─────────────────────────────────────────
        prompt   = build_prompt(state.query, state.documents, corpus)
        response = _call_with_retry(prompt, backend, model)
        trace["llm_calls"] += 1
        trace["steps"] = step + 1
        # ─────────────────────────────────────────────────────────────────

        action = response["action"]
        reason = response.get("reason", "")
        trace["actions"].append(action)

        if verbose:
            print(f"Step {step+1}: {action.upper()}")
            if reason: print(f"  Reason: {reason[:100]}")

        if action == "stop":
            trace["stop_reason"] = "llm_stop"
            break

        elif action == "refine query":
            new_query = response.get("refined_query", state.query)
            if verbose: print(f"  '{state.query[:60]}' → '{new_query[:60]}'")

            # Python handles retrieval — not the LLM
            new_results  = retriever.retrieve(new_query, k=top_k)
            new_ids      = [d for d, _ in new_results]
            new_scores   = {d: s for d, s in new_results}

            # Merge: keep old docs, append new ones
            merged_ids    = list(state.documents)
            merged_scores = dict(state.scores)
            for doc_id, score in zip(new_ids, new_results):
                if doc_id not in merged_scores:
                    merged_ids.append(doc_id)
                    merged_scores[doc_id] = score[1]

            state = RetrievalState(new_query, merged_ids, merged_scores)
            trace["queries"].append(new_query)

        elif action == "re-rank":
            reranked = response.get("reranked", state.documents)

            # Hallucination guard (paper Section 3.2):
            # Remove doc IDs the LLM hallucinated (not in our list)
            valid   = [d for d in reranked if d in state.scores]
            # Re-append anything the LLM accidentally dropped
            missing = [d for d in state.documents if d not in valid]
            final   = valid + missing

            if verbose:
                print(f"  Top before: {state.documents[0]} → Top after: {final[0]}")

            state = RetrievalState(state.query, final, state.scores)

    if verbose:
        print(f"\nFinal: '{state.query[:60]}'")
        print(f"Top-3: {state.documents[:3]}")
        print(f"LLM calls: {trace['llm_calls']} | Stop: {trace['stop_reason']}")

    return state, trace
