# scripts/run_pilot.py
from data.scifact_loader import load_queries, load_corpus, load_qrels
from retrieval.dense_retriever import DenseRetriever
from budget.heuristic import HeuristicController, FixedController, NoRetrievalController

# 1. Încarcă datele
queries = load_queries()   # 150 queries
corpus  = load_corpus()    # 5183 documente
qrels   = load_qrels()     # etichetele gold

# 2. Indexează corpus (o dată, apoi din cache)
retriever = DenseRetriever()
retriever.index(corpus)

# 3. Rulează un controller
controller = HeuristicController()

for query in queries:
    k    = controller.decide(query["text"])
    docs = retriever.retrieve(query["text"], k=k)
    # → trimiți docs la LLM → evaluezi răspunsul