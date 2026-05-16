"""
smr/state.py
------------
The structured state representation at the core of SMR.

In the paper (Section 3.1), a state is defined as:
    s_t = (q_t, D_t)

where q_t is the current query and D_t is the ranked list of top-k doc IDs.

Two states are EQUIVALENT if both the query and the document ranking are
identical — this is the redundancy check that triggers early stopping.
Without this check, CoT-based systems keep reasoning in circles (Figure 1a).
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class RetrievalState:
    """
    A single reasoning state in the SMR state machine.

    Attributes:
        query     : current query string (may have been refined from the original)
        documents : ranked list of doc_ids, best first
        scores    : {doc_id: bm25_score} — kept separately so we can re-sort
    """
    query: str
    documents: List[str]
    scores: Dict[str, float] = field(default_factory=dict)

    def __eq__(self, other: "RetrievalState") -> bool:
        """
        Equivalence check from Section 3.1:
        s_t == s_{t-1}  iff  query unchanged AND document list unchanged.
        If true, the system is in a redundant loop → STOP.
        """
        if not isinstance(other, RetrievalState):
            return False
        return self.query == other.query and self.documents == other.documents

    def top_score(self) -> float:
        """BM25 score of the highest-ranked document. Used by the policy."""
        if not self.documents:
            return 0.0
        return self.scores.get(self.documents[0], 0.0)

    def __repr__(self) -> str:
        top = self.documents[0] if self.documents else "none"
        return (f"State(query='{self.query[:50]}...', "
                f"top_doc={top}, top_score={self.top_score():.3f}, "
                f"n_docs={len(self.documents)})")
