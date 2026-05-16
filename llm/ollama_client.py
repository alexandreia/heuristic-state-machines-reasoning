"""
llm/ollama_client.py
---------------------
Trimite un query + documente la Mistral via Ollama și returnează răspunsul.

Setup:
    brew install ollama
    ollama serve                   # într-un terminal separat
    ollama pull mistral:7b-instruct
"""

import requests

MODEL   = "mistral:7b-instruct"
URL     = "http://localhost:11434/api/generate"


def build_prompt(query: str, docs: list[dict]) -> str:
    """Construiește promptul din query + documentele recuperate."""

    if not docs:
        return f"Answer the following question based on your knowledge.\n\nQuestion: {query}\nAnswer:"

    context = "\n\n".join(
        f"[Document {i}]\nTitle: {doc['title']}\n{doc['text']}"
        for i, doc in enumerate(docs, 1)
    )

    return (
        f"Answer the following question based only on the provided documents.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def ask(query: str, docs: list[dict]) -> dict:
    """
    Trimite query + documente la Mistral.

    Returns:
        {"answer": str, "prompt_tokens": int, "output_tokens": int}
    """
    prompt   = build_prompt(query, docs)
    response = requests.post(URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": 0, "temperature": 0, "num_predict": 150}
    })
    data = response.json()

    return {
        "answer":        data["response"].strip(),
        "prompt_tokens": data["prompt_eval_count"],
        "output_tokens": data["eval_count"]
    }


# ── Test rapid ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "Does aspirin reduce fever?"
    docs  = [{"title": "Aspirin study", "text": "Aspirin reduces fever by inhibiting prostaglandins."}]

    result = ask(query, docs)
    print(f"Answer: {result['answer']}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
