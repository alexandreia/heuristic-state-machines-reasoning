"""
llm/prompt.py
-------------
The exact policy prompt from the SMR paper (Table 5).

This prompt is the entire "intelligence" of the LLM-based SMR — the LLM
reads it, sees the current state, and outputs a JSON action. No fine-tuning,
no RL, no gradient updates. Just structured prompting of an off-the-shelf model.

Key design choices in the prompt:
  1. Decision policy is ordered (REFINE first, then RERANK, then STOP)
     — this ordering matters for action balance
  2. Output is strict JSON — reduces hallucination compared to free text
  3. The "reason" field is kept for interpretability but not used by the system
"""

import json
from typing import List, Tuple

# ── Paper's exact prompt (Table 5) ────────────────────────────────────────

POLICY_PROMPT = """You are a highly intelligent artificial agent responsible for managing a search system. Your role is to either refine the given query or re-rank retrieved search results, thereby enhancing both recall and precision of the search. You can output exactly one of the following operations, after which another agent will execute it and return the results to you.

## Input Format
The input provided to you will have the following structure:
{
  "query": "<current version of a query>",
  "retrieved": [
    ("<docid>", "document contents"),
    ...
  ]
}

### Decision policy (check in order):
1. Query Refinement
   Choose "refine query" if any of the following are met:
   - The query is ambiguous or generic
   - The retrieved search results are unsatisfactory
   - The query is short
   - Key domain terms are missing in the query

2. Reranking
   Only if the query already looks good and at least one retrieved document seems on topic.

3. Stop
   Only when you are certain that no further improvement is possible.

## Possible Outputs (select exactly one)

### Query Refinement
Output format:
```json
{
  "action": "refine query",
  "refined_query": "<refined version of a query>",
  "reason": "<reason for this action>"
}
```

### Re-ranking
Output format:
```json
{
  "action": "re-rank",
  "reranked": ["<docid>", "<docid>", ...],
  "reason": "<reason for this action>"
}
```

### Stop
Output format:
```json
{
  "action": "stop"
}
```

## Current State
"""


def build_prompt(query: str,
                 doc_ids: List[str],
                 corpus: dict,
                 max_docs: int = 10,
                 max_chars_per_doc: int = 300) -> str:
    """
    Format the current (query, documents) state into the full prompt.

    Args:
        query            : current query string
        doc_ids          : ranked list of doc IDs
        corpus           : {doc_id: {"title": str, "text": str}}
        max_docs         : how many docs to include (context length limit)
        max_chars_per_doc: truncate doc text to avoid hitting token limits
    """
    retrieved = []
    for doc_id in doc_ids[:max_docs]:
        doc  = corpus.get(doc_id, {})
        text = doc.get("title", "") + " " + doc.get("text", "")
        retrieved.append((doc_id, text[:max_chars_per_doc]))

    state_json = json.dumps({
        "query": query,
        "retrieved": retrieved,
    }, indent=2)

    return POLICY_PROMPT + state_json
