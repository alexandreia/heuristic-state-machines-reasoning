"""
smr/actions.py
--------------
The three executable actions in the SMR state machine.

Two REFINE strategies are available:

  refine_prf()      — Pseudo-Relevance Feedback (Rocchio, 1971)
                      Borrows terms from top retrieved docs.
                      Fast. Fails when initial retrieval is poor.

  refine_synonyms() — WordNet synonym expansion
                      Expands query words using a lexical database.
                      Independent of retrieval quality.
                      Safer for scientific/domain corpora.

  rerank()          — BM25 rescoring with current query (unchanged)

Experiment finding: on SciFact, refine_synonyms outperforms refine_prf
because PRF propagates noise from irrelevant top docs, while synonym
expansion stays grounded in the original query's semantics.
"""

import re
from collections import Counter
from typing import List, Dict
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


def _retrieve_and_merge(
    state: RetrievalState,
    new_query: str,
    retriever,
    top_k: int,
) -> RetrievalState:
    """
    Shared merge logic used by both REFINE strategies.

    Retrieves docs for new_query and appends any new doc_ids to the
    existing list. Never discards old docs (paper Appendix A.1).
    """
    new_results  = retriever.retrieve(new_query, k=top_k)
    merged_ids   = list(state.documents)
    merged_scores = dict(state.scores)

    for doc_id, score in new_results:
        if doc_id not in merged_scores:
            merged_ids.append(doc_id)
            merged_scores[doc_id] = score

    return RetrievalState(
        query=new_query,
        documents=merged_ids,
        scores=merged_scores,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REFINE strategy A — Pseudo-Relevance Feedback
# ─────────────────────────────────────────────────────────────────────────────

def refine_prf(
    state: RetrievalState,
    retriever,
    corpus: Dict[str, dict],
    n_feedback_docs: int = 3,
    n_expansion_terms: int = 5,
    top_k: int = 10,
) -> RetrievalState:
    """
    Expand query by borrowing frequent terms from top retrieved docs.

    Weakness: if top docs are irrelevant, we borrow irrelevant terms.
    This is the "garbage in, garbage out" problem on SciFact.
    """
    term_counts = Counter()
    for doc_id in state.documents[:n_feedback_docs]:
        doc = corpus.get(doc_id, {})
        text = doc.get("title", "") + " " + doc.get("text", "")
        term_counts.update(_tokenize(text))

    existing  = set(_tokenize(state.query))
    new_terms = [t for t, _ in term_counts.most_common(30)
                 if t not in existing][:n_expansion_terms]

    new_query = state.query + " " + " ".join(new_terms)
    return _retrieve_and_merge(state, new_query, retriever, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# REFINE strategy B — WordNet synonym expansion
# ─────────────────────────────────────────────────────────────────────────────

def refine_synonyms(
    state: RetrievalState,
    retriever,
    corpus: Dict[str, dict],      # kept for API compatibility, not used here
    n_synonyms_per_word: int = 1,
    n_total_synonyms: int = 5,
    top_k: int = 10,
) -> RetrievalState:
    """
    Expand query using WordNet synonyms of the query words.

    For each content word in the query:
      1. Look up its WordNet synsets
      2. Take only synsets[0] — the most common meaning (avoids semantic drift)
      3. Pick the first synonym that isn't already in the query

    Advantage over PRF: expansion is based on the query itself, not on
    potentially-irrelevant retrieved documents. Safer for domain corpora
    where BM25 initial retrieval may be noisy.

    Limitation: WordNet has limited coverage of biomedical terms
    (e.g. "metformin", "prostaglandin" may not be found).
    General words like "reduce", "mortality", "inhibit" work well.

    Requires: nltk + wordnet data
        pip install nltk
        python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        raise ImportError(
            "NLTK not installed. Run: pip install nltk\n"
            "Then: python -c \"import nltk; nltk.download('wordnet')\""
        )

    tokens   = _tokenize(state.query)
    existing = set(tokens)
    new_terms = []

    for token in tokens:
        if len(new_terms) >= n_total_synonyms:
            break

        synsets = wn.synsets(token)
        if not synsets:
            continue                       # word not in WordNet — skip

        # Only use the first synset (most common meaning)
        # Using all synsets risks "drug" → "drug someone (verb)" drift
        first_synset = synsets[0]
        added = 0

        for lemma in first_synset.lemmas():
            if added >= n_synonyms_per_word:
                break
            synonym = lemma.name().replace("_", " ").lower()

            # Skip: same as original, multi-word (noisy), already present
            if (synonym == token
                    or " " in synonym
                    or synonym in existing
                    or synonym in new_terms):
                continue

            new_terms.append(synonym)
            existing.add(synonym)
            added += 1

    if not new_terms:
        # WordNet found nothing useful — fall back to PRF
        return refine_prf(state, retriever, corpus,
                          n_feedback_docs=3,
                          n_expansion_terms=3,
                          top_k=top_k)

    new_query = state.query + " " + " ".join(new_terms)
    return _retrieve_and_merge(state, new_query, retriever, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# RERANK — BM25 rescoring
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    state: RetrievalState,
    retriever,
) -> RetrievalState:
    """
    Re-score all documents against the current (possibly refined) query
    and sort by new score.

    This corrects the ranking when the query has changed since initial
    retrieval — old scores were computed against a different query string.
    """
    new_scores = {
        doc_id: retriever.score(state.query, doc_id)
        for doc_id in state.documents
    }
    reranked = sorted(state.documents,
                      key=lambda d: new_scores[d], reverse=True)

    return RetrievalState(
        query=state.query,
        documents=reranked,
        scores=new_scores,
    )
