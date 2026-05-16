"""
budget/heuristic.py
--------------------
Controller-ele care decid câte documente să recupereze per query.

Trei variante:
- HeuristicController : contribuția voastră — adaptiv pe baza lungimii query-ului
- FixedController     : baseline fix (fixed_5, fixed_10)
- NoRetrievalController: baseline closed-book (niciun document)
"""


class HeuristicController:
    """
    Decide k pe baza numărului de cuvinte din query și a scorului de relevanță.

    Regula 1 (lungime): query scurt → mai puține documente
    Regula 2 (scor):    dacă cel mai bun document are scor mic → k=0 (inspirat din Stop în SMR)
    """

    def __init__(self, k_short: int = 3, k_long: int = 8, threshold: int = 6, min_score: float = 0.4):
        """
        Args:
            k_short   : documente pentru query scurt (≤ threshold cuvinte)
            k_long    : documente pentru query lung (> threshold cuvinte)
            threshold : granița în număr de cuvinte
            min_score : scor cosinus minim — sub acest prag, k=0 (retriever-ul nu a găsit nimic util)
        """
        self.k_short    = k_short
        self.k_long     = k_long
        self.threshold  = threshold
        self.min_score  = min_score

    def decide(self, query: str, top_score: float = None) -> int:
        """
        Args:
            query     : textul întrebării
            top_score : scorul cosinus al celui mai bun document (din retriever)

        Returns:
            k: câte documente să recupereze
        """
        # Regula 2: scor prea mic → closed-book (inspirat din Stop în SMR)
        if top_score is not None and top_score < self.min_score:
            return 0

        # Regula 1: lungime query
        word_count = len(query.split())
        return self.k_short if word_count <= self.threshold else self.k_long


class FixedController:
    """Recuperează întotdeauna exact k documente. Folosit pentru fixed_5 și fixed_10."""

    def __init__(self, k: int):
        self.k = k

    def decide(self, query: str) -> int:
        return self.k


class NoRetrievalController:
    """Baseline closed-book — nu recuperează niciun document."""

    def decide(self, query: str) -> int:
        return 0


# ── Test rapid ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "Does aspirin reduce fever?",                                        # 4 cuvinte → k_short
        "What are the molecular mechanisms by which statins reduce LDL?",   # 13 cuvinte → k_long
    ]

    heuristic = HeuristicController(k_short=3, k_long=8, threshold=6)
    fixed10   = FixedController(k=10)
    no_ret    = NoRetrievalController()

    print(f"{'Query':<60} {'Heuristic':>10} {'Fixed10':>10} {'NoRet':>7}")
    print("-" * 90)
    for q in queries:
        print(f"{q:<60} {heuristic.decide(q):>10} {fixed10.decide(q):>10} {no_ret.decide(q):>7}")

    print("\n✅ Controller OK!")
