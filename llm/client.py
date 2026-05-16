"""
llm/client.py
-------------
Unified LLM client that works with both OpenAI and Ollama.

Ollama uses the same OpenAI API spec intentionally — so the exact same
code works locally (free) or via the cloud (paid) by just changing
base_url. This is a great example of API compatibility design.

Setup:
  OpenAI:  export OPENAI_API_KEY=sk-...
  Ollama:  brew install ollama && ollama serve && ollama pull qwen2.5:7b
"""

import os
from openai import OpenAI

# ── Pre-configured clients ─────────────────────────────────────────────────

_openai_client = None
_ollama_client = None


def _get_client(backend: str) -> OpenAI:
    global _openai_client, _ollama_client

    if backend == "openai":
        if _openai_client is None:
            _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return _openai_client

    elif backend == "ollama":
        if _ollama_client is None:
            _ollama_client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",   # Ollama doesn't need a real key
            )
        return _ollama_client

    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'openai' or 'ollama'.")


def llm_call(
    prompt: str,
    backend: str = "ollama",
    model: str = "qwen2.5:7b",
    temperature: float = 0.0,
) -> str:
    """
    Send a prompt to the chosen LLM backend and return the raw text response.

    Args:
        prompt      : full prompt string (already formatted)
        backend     : "ollama" (local, free) or "openai" (cloud)
        model       : model name — "qwen2.5:7b", "llama3.2", "gpt-4o-mini", etc.
        temperature : 0.0 = deterministic (paper default, Appendix A.3)

    Returns:
        raw text response from the LLM
    """
    client = _get_client(backend)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── Model recommendations ──────────────────────────────────────────────────
#
#   Backend   Model              Notes
#   ────────  ─────────────────  ──────────────────────────────────────────
#   ollama    qwen2.5:7b         Closest to paper's Qwen2.5-32B, ~5GB
#   ollama    llama3.2           Good general alternative, ~2GB
#   ollama    qwen2.5:14b        Better quality, needs 16GB RAM
#   openai    gpt-4o-mini        Cheapest cloud option (~$0.002/query)
#   openai    gpt-4o             Best quality
