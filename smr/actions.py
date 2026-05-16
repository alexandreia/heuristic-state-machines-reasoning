"""
smr/actions.py
--------------
The three executable actions in the SMR state machine.

Mirrors Section 3.2 of the paper:
  REFINE : update query to better reflect user intent         (Equation 3)
  RERANK : adjust document ordering without changing query    (Equation 4)
  STOP   : terminate — return current state as final output

In the paper, REFINE and RERANK are done by the LLM. Here we replace them
with classical IR techniques:
  REFINE → Pseudo-Relevance Feedback (Rocchio, 1971)
  RERANK → BM25 re-scoring with the (possibly refined) query

Fun fact: Rocchio's algorithm predates the internet by 20 years. It was
designed for physical card-catalogue IR systems and still works remarkably
well — a testament to how fundamental the idea of "look at what you found,
then search again" really is.
"""

import re
from collections import Counter
from typing import List, Tuple, Dict
from smr.state import RetrievalState


STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "and", "to", "for", "it",
    "with", "that", "are", "was", "on", "at", "by", "from", "or",
    "be", "this", "which", "have", "has", "its", "their", "they",
    "we", "our", "not", "but", "as", "if", "so", "do", "did",
}


def _tokenize(text: str) -> List[str]:
    return [
        w for w in re.findall(r"\w+", text.lower())
        if w not in STOPWORDS and len(w) > 3
    ]


# ─────────────────────────────────────────────────────────────────────────────
# REFINE — Pseudo-Relevance Feedback
# ─────────────────────────────────────────────────────────────────────────────

def refine(
    state: RetrievalState,
    retriever,                  # BM25Retriever
    corpus: Dict[str, dict],
    n_feedback_docs: int = 3,
    n_expansion_terms: int = 5,
    top_k: int = 10,
) -> RetrievalState:
    """
    REFINE action: expand the query using Pseudo-Relevance Feedback (PRF),
    then retrieve new documents and merge them with the existing list.

    PRF assumes the top-n documents are relevant and extracts their most
    distinctive terms to expand the query. This is the heuristic analogue
    of the LLM rewriting the query based on retrieved context.

    Paper note (Appendix A.1): new documents are APPENDED to the existing
    list rather than replacing it. This preserves context from earlier steps.

    Args:
        state            : current (query, documents, scores)
        retriever        : BM25Retriever with .retrieve() and .score()
        corpus           : raw corpus dict for reading document text
        n_feedback_docs  : how many top docs to use for PRF (default 3)
        n_expansion_terms: how many new terms to add to query (default 5)
        top_k            : how many docs to retrieve with new query

    Returns:
        new RetrievalState with expanded query and merged document list
    """
    # ── Step 1: collect terms from top-n feedback documents ───────────────
    term_counts = Counter()
    for doc_id in state.documents[:n_feedback_docs]:
        doc = corpus.get(doc_id, {})
        text = doc.get("title", "") + " " + doc.get("text", "")
        term_counts.update(_tokenize(text))

    # ── Step 2: filter out terms already in the query ─────────────────────
    existing_terms = set(_tokenize(state.query))
    new_terms = [
        term for term, _ in term_counts.most_common(30)
        if term not in existing_terms
    ][:n_expansion_terms]

    expanded_query = state.query + " " + " ".join(new_terms)

    # ── Step 3: retrieve with expanded query ──────────────────────────────
    new_results = retriever.retrieve(expanded_query, k=top_k)
    new_ids    = [d for d, _ in new_results]
    new_scores = {d: s for d, s in new_results}

    # ── Step 4: merge — keep existing docs, append genuinely new ones ─────
    # (paper Section 3.2 + Appendix A.1)
    merged_ids    = list(state.documents)
    merged_scores = dict(state.scores)
    for doc_id, score in zip(new_ids, new_results):
        if doc_id not in merged_scores:
            merged_ids.append(doc_id)
            merged_scores[doc_id] = score[1]

    return RetrievalState(
        query=expanded_query,
        documents=merged_ids,
        scores=merged_scores,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RERANK — BM25 re-scoring
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    state: RetrievalState,
    retriever,                  # BM25Retriever
) -> RetrievalState:
    """
    RERANK action: re-score all documents in the current list using the
    current (possibly refined) query, then sort by new score.

    This is the heuristic analogue of the LLM judging document relevance.
    When the query has been refined, the original BM25 scores were computed
    against a different query — re-scoring fixes that mismatch.

    Paper note (Section 3.2): hallucination guards are applied.
    Here the equivalent is: we only sort existing docs, never add new ones.

    Args:
        state     : current (query, documents, scores)
        retriever : BM25Retriever with .score(query, doc_id)

    Returns:
        new RetrievalState with same docs, re-sorted by new BM25 scores
    """
    # Re-score every doc in the current list against the current query
    new_scores = {
        doc_id: retriever.score(state.query, doc_id)
        for doc_id in state.documents
    }

    # Sort by new score (highest first)
    reranked_ids = sorted(state.documents,
                          key=lambda d: new_scores[d], reverse=True)

    return RetrievalState(
        query=state.query,
        documents=reranked_ids,
        scores=new_scores,
    )
