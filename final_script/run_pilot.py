"""
scripts/run_pilot.py
---------------------
Rulează experimentul complet și salvează rezultatele.

Utilizare:
    python scripts/run_pilot.py

Rezultate salvate în:
    results/final.json
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # adaugă rădăcina proiectului

from data.scifact_loader import load_queries, load_corpus, load_qrels
from retrieval.dense_retriever import DenseRetriever
from budget.heuristic import HeuristicController, FixedController, NoRetrievalController
from llm.ollama_client import ask
from eval.metrics import token_f1, macro_f1, ndcg_at_k, mean_ndcg


def run_config(name: str, controller, retriever, queries: list, qrels: dict, corpus_index: dict) -> dict:
    """
    Rulează o singură configurație pe toate query-urile.

    Returns:
        dict cu rezultatele agregate + per query
    """
    print(f"\n── {name} ──")
    per_query = []

    for query in queries:
        # întâi ia scorul top documentului (cu k=1) pentru decizia controller-ului
        _, top_score = retriever.retrieve(query["text"], k=1)
        k            = controller.decide(query["text"], top_score=top_score)
        docs, _      = retriever.retrieve(query["text"], k=k) if k > 0 else ([], 0.0)
        result       = ask(query["text"], docs)

        # gold = textul primului document relevant din qrels
        gold_ids  = qrels.get(query["query_id"], [])
        gold_text = corpus_index[gold_ids[0]]["text"] if gold_ids else ""

        f1            = token_f1(result["answer"], gold_text)
        retrieved_ids = [doc["doc_id"] for doc in docs]
        ndcg          = ndcg_at_k(retrieved_ids, gold_ids)

        per_query.append({
            "query_id":      query["query_id"],
            "query":         query["text"],
            "k":             k,
            "prompt_tokens": result["prompt_tokens"],
            "answer":        result["answer"],
            "gold":          gold_text,
            "f1":            f1,
            "ndcg":          ndcg
        })

        print(f"  [{query['query_id']}] k={k} | tokens={result['prompt_tokens']} | f1={f1:.2f} | ndcg={ndcg:.2f}")

    predictions = [r["answer"] for r in per_query]
    golds       = [r["gold"]   for r in per_query]

    retrieved_ids_list = [[doc["doc_id"] for doc in retriever.retrieve(q["text"], k=controller.decide(q["text"]))] for q in queries]
    gold_ids_list      = [qrels.get(q["query_id"], []) for q in queries]

    return {
        "config":            name,
        "macro_f1":          macro_f1(predictions, golds),
        "mean_ndcg":         mean_ndcg(retrieved_ids_list, gold_ids_list),
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in per_query) / len(per_query),
        "avg_k":             sum(r["k"] for r in per_query) / len(per_query),
        "per_query":         per_query
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rulează experimentul RAG")
    parser.add_argument("--fixed_k1",   type=int, default=5,  help="k pentru fixed_k1 (default: 5)")
    parser.add_argument("--fixed_k2",   type=int, default=10, help="k pentru fixed_k2 (default: 10)")
    parser.add_argument("--k_short",    type=int, default=3,  help="k pentru query scurt în heuristică (default: 3)")
    parser.add_argument("--k_long",     type=int, default=8,  help="k pentru query lung în heuristică (default: 8)")
    parser.add_argument("--threshold",  type=int, default=6,  help="prag de cuvinte în heuristică (default: 6)")
    parser.add_argument("--n",          type=int, default=None, help="număr de queries de rulat (default: toate)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Încarcă datele ──────────────────────────────────────────────────────
    print("Încarc date...")
    queries  = load_queries()[:args.n]
    corpus   = load_corpus()
    qrels    = load_qrels()

    # ── Indexează corpus (din cache dacă există) ────────────────────────────
    retriever = DenseRetriever()
    retriever.index(corpus, cache_path="data/embeddings.pkl")

    # ── Index rapid doc_id → doc (pentru gold lookup) ───────────────────────
    corpus_index = {doc["doc_id"]: doc for doc in corpus}

    # ── Definește configurațiile ────────────────────────────────────────────
    configs = {
        "no_retrieval":          NoRetrievalController(),
        f"fixed_{args.fixed_k1}": FixedController(k=args.fixed_k1),
        f"fixed_{args.fixed_k2}": FixedController(k=args.fixed_k2),
        "heuristic":             HeuristicController(
                                     k_short=args.k_short,
                                     k_long=args.k_long,
                                     threshold=args.threshold
                                 ),
    }

    # ── Rulează fiecare configurație ────────────────────────────────────────
    results = {}
    for name, controller in configs.items():
        results[name] = run_config(name, controller, retriever, queries, qrels, corpus_index)

    # ── Afișează tabelul de rezultate ───────────────────────────────────────
    print("\n\n── Rezultate finale ──")
    print(f"{'Config':<15} {'Macro F1':>10} {'nDCG@10':>10} {'Avg Tokens':>12} {'Avg k':>7}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<15} {r['macro_f1']:>10.3f} {r['mean_ndcg']:>10.3f} {r['avg_prompt_tokens']:>12.0f} {r['avg_k']:>7.1f}")

    # ── Salvează rezultatele ─────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    with open("results/final.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Rezultate salvate în results/final.json")


if __name__ == "__main__":
    main()
