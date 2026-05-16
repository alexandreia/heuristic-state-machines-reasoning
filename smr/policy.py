"""
smr/policy.py
-------------
The heuristic policy that replaces the LLM judge in SMR.

In the paper (Section 3.3), the LLM reads the current state (q, D) and
selects one of {REFINE, RERANK, STOP}. Here we replace that with rules.

Design philosophy — "explore first, stop when confident":
  The paper's LLM heavily favours REFINE early (Table 3: 55–63% of actions
  on hard queries are REFINE). A conservative "STOP by default" policy
  wastes the state machine entirely. Instead we bias toward exploration
  and only stop when we have positive evidence the results are good.

Decision logic (applied in order):
  0. Equivalent state?              → STOP   (we're in a loop — cut it)
  1. First step AND score very high → STOP   (already great, no point)
  2. First step                     → REFINE (always try to improve once)
  3. Score improved vs last step?   → RERANK (docs changed, fix ordering)
  4. Score is high enough now       → STOP   (good enough after refinement)
  5. Top-2 scores are bunched       → RERANK (ordering uncertain)
  6. Default                        → STOP   (no improvement signal)
"""

from typing import Optional
from smr.state import RetrievalState


REFINE = "REFINE"
RERANK = "RERANK"
STOP   = "STOP"


class HeuristicPolicy:

    def __init__(
        self,
        stop_score_high: float = 15.0,  # BM25 score → definitely stop
        min_query_words: int   = 3,     # shorter than this → refine first
        score_gap_ratio: float = 0.20,  # top/2nd gap < 20% → rerank
    ):
        """
        Args:
            stop_score_high : If top-doc score >= this, results are great → STOP
                              Set based on your corpus. SciFact BM25 scores
                              typically range 1–25, so 15 is a high-confidence bar.
            min_query_words : Queries shorter than this always get REFINE first.
            score_gap_ratio : If (score1 - score2) / score1 < ratio,
                              ranking is uncertain → RERANK.
        """
        self.stop_high = stop_score_high
        self.min_words = min_query_words
        self.gap_ratio = score_gap_ratio

    def select_action(
        self,
        state: RetrievalState,
        previous_state: Optional[RetrievalState],
    ) -> str:
        top   = state.top_score()
        words = len(state.query.split())
        is_first_step = previous_state is None

        # ── Rule 0: Equivalent state → STOP ──────────────────────────────
        # We're in a loop — both query and ranking are unchanged.
        # This is the core anti-redundancy check from Section 3.1.
        if not is_first_step and state == previous_state:
            return STOP

        # ── Rule 1: First step + very high score → already excellent ──────
        # The retriever got lucky. No need to do anything.
        if is_first_step and top >= self.stop_high:
            return STOP

        # ── Rule 2: First step → always try to improve once ───────────────
        # Inspired by the paper's "explore-first" pattern (Table 3).
        # Short queries obviously need expansion. But even long queries
        # benefit from PRF: the original query may miss domain vocabulary.
        if is_first_step:
            return REFINE

        # ── From here: we've done at least one REFINE ─────────────────────

        # ── Rule 3: Score improved after REFINE → fix the ordering ────────
        # New documents joined the list. The ranking was computed with the
        # old query — rescore everything against the refined query.
        if previous_state is not None:
            prev_top = previous_state.top_score()
            if top > prev_top * 1.05:   # score went up by > 5% → rerank
                return RERANK

        # ── Rule 4: Score is now high enough → stop ───────────────────────
        if top >= self.stop_high * 0.7:  # 70% of the high threshold
            return STOP

        # ── Rule 5: Top-2 scores are bunched → ordering is uncertain ──────
        if len(state.documents) >= 2:
            s1 = state.scores.get(state.documents[0], 0.0)
            s2 = state.scores.get(state.documents[1], 0.0)
            if s1 > 0 and (s1 - s2) / s1 < self.gap_ratio:
                return RERANK

        # ── Default: no signal to act → stop ──────────────────────────────
        return STOP


if __name__ == "__main__":
    from smr.state import RetrievalState

    policy = HeuristicPolicy()
    s0 = RetrievalState("LLM", ["d1", "d2"], {"d1": 2.0, "d2": 1.8})
    s1 = RetrievalState("What is a Large Language Model transformer", ["d1","d2","d3"], {"d1": 8.0, "d2": 5.0, "d3": 2.0})
    s2 = RetrievalState("What is a Large Language Model transformer", ["d1","d2","d3"], {"d1": 8.0, "d2": 5.0, "d3": 2.0})

    print(policy.select_action(s0, None))   # → REFINE (first step)
    print(policy.select_action(s1, s0))     # → RERANK (score improved)
    print(policy.select_action(s2, s1))     # → STOP   (equivalent state)
