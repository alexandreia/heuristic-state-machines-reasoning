"""
smr/policy.py
-------------
The heuristic policy that replaces the LLM judge in SMR.

In the paper (Section 3.3), the LLM reads the current state (q, D) and
selects one of {REFINE, RERANK, STOP}. Here we replace that LLM call with
deterministic rules based on observable signals in the state.

Decision logic (applied in order):
  1. Redundant state?               → STOP   (state machine check, no LLM needed)
  2. Strong top-doc signal?         → STOP   (we already have a great result)
  3. Query too short / vague?       → REFINE (expand before judging ranking)
  4. Top-doc signal very weak?      → REFINE (search harder, different terms)
  5. Scores bunched together?       → RERANK (good docs present, wrong order)
  6. Default                        → STOP   (conservative fallback)

This maps directly to the paper's prompt heuristics (Table 5):
  "Choose REFINE when the query is vague or results unsatisfactory"
  "Prefer STOP only when confident no further improvement is possible"
"""

from typing import Optional
from smr.state import RetrievalState


# ── Action constants ───────────────────────────────────────────────────────

REFINE = "REFINE"
RERANK = "RERANK"
STOP   = "STOP"


class HeuristicPolicy:
    """
    Rule-based action selector. Parameters are tunable hyperparameters
    you can experiment with (see run_experiment.py --help).
    """

    def __init__(
        self,
        stop_score_high: float = 8.0,   # BM25 score → definitely stop
        stop_score_low:  float = 1.5,   # BM25 score → definitely refine
        min_query_words: int   = 3,     # shorter than this → refine first
        score_gap_ratio: float = 0.10,  # top/2nd score gap < 10% → rerank
    ):
        """
        Args:
            stop_score_high : If top-doc BM25 score >= this, results are great → STOP
            stop_score_low  : If top-doc BM25 score <  this, results are poor  → REFINE
            min_query_words : Queries shorter than this get REFINE regardless of score
            score_gap_ratio : If (score1 - score2) / score1 < ratio, scores are
                              bunched → doc ordering is uncertain → RERANK
        """
        self.stop_high  = stop_score_high
        self.stop_low   = stop_score_low
        self.min_words  = min_query_words
        self.gap_ratio  = score_gap_ratio

    def select_action(
        self,
        state: RetrievalState,
        previous_state: Optional[RetrievalState],
    ) -> str:
        """
        Given the current state and the previous state, return one of
        REFINE / RERANK / STOP.

        Args:
            state          : current (query, documents) state
            previous_state : the state from the previous step (or None)

        Returns:
            one of the three action strings
        """
        # ── Rule 0: Equivalent state → STOP (paper Section 3.1) ──────────
        # This is the core anti-redundancy mechanism from the paper.
        if previous_state is not None and state == previous_state:
            return STOP

        top   = state.top_score()
        words = len(state.query.split())

        # ── Rule 1: Very high signal → already great → STOP ───────────────
        if top >= self.stop_high:
            return STOP

        # ── Rule 2: Short query → expand before doing anything else ───────
        # Short queries are almost always underspecified.
        if words < self.min_words:
            return REFINE

        # ── Rule 3: Very weak signal → corpus has nothing useful → REFINE ─
        # This is the analogue of SMR's REFINE for "unsatisfactory results".
        if top < self.stop_low:
            return REFINE

        # ── Rule 4: Scores are bunched → ranking is uncertain → RERANK ────
        # If the top two docs score almost the same, their order is arbitrary
        # and a BM25 rescore with the (possibly refined) query may fix it.
        if len(state.documents) >= 2:
            score1 = state.scores.get(state.documents[0], 0.0)
            score2 = state.scores.get(state.documents[1], 0.0)
            if score1 > 0 and (score1 - score2) / score1 < self.gap_ratio:
                return RERANK

        # ── Default: moderate signal, query is fine → STOP ────────────────
        return STOP


if __name__ == "__main__":
    from smr.state import RetrievalState

    policy = HeuristicPolicy()
    s0 = RetrievalState("LLM", ["d1", "d2"], {"d1": 0.8, "d2": 0.7})
    s1 = RetrievalState("What is a Large Language Model", ["d1", "d2"], {"d1": 9.5, "d2": 6.1})
    s2 = RetrievalState("What is a Large Language Model", ["d1", "d2"], {"d1": 9.5, "d2": 6.1})

    print(policy.select_action(s0, None))    # → REFINE  (query too short)
    print(policy.select_action(s1, s0))      # → STOP    (high score)
    print(policy.select_action(s2, s1))      # → STOP    (equivalent state)
