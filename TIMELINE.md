# Project Timeline — Adaptive RAG for Token-Efficiency

---

## Week 1 — Learn & Setup (Days 1–5)

### Already done ✅
- Dataset uploaded to HuggingFace (150 queries, seed=0)
- `scifact_loader.py` — loads corpus, queries, qrels from HuggingFace
- `dense_retriever.py` — MiniLM embeddings + cosine search + disk cache
- `heuristic.py` — query length rule + SMR-inspired stop rule (min_score)
- `ollama_client.py` — Mistral 7B via Ollama REST API
- `metrics.py` — Token F1 + nDCG@10
- `run_pilot.py` — full experiment runner with argparse (`--n`, `--fixed_k1`, etc.)

### Days 2–3 (everyone)
- [ ] Read: Liu (Lost in the Middle) + Lewis (RAG) + Jeong (Adaptive-RAG)
- [ ] **P4:** Install Ollama, pull `mistral:7b-instruct`, confirm `prompt_eval_count > 0`
- [ ] Clone repo, run `python run_pilot.py --n 3` — pipeline must not crash

### Days 4–5
- [ ] Read: Asai (Self-RAG) + Jiang (FLARE) — for Related Work contrast
- [ ] Set up GitHub repo, assign branches per person
- [ ] 🔒 Lock Day-1 decisions: model tag, seed=0, meeting cadence

---

## Week 2 — Build & Run (Days 6–10)

| Day | Who | Task |
|-----|-----|------|
| 6 | P1 | Verify `scifact_loader.py` end-to-end on own laptop |
| 6 | P4 | Run pipeline on 5 queries with EchoLLM → then OllamaLLM |
| 7 | P1 | Build corpus embeddings, confirm cache saves correctly |
| 7 | P2 | Tune heuristic threshold if needed (add 2nd rule only if data supports it) |
| 8 | ALL | First end-to-end run on 5 queries — goal: no crashes |
| 8 | P3 | Read: Xu (RECOMP) + Jiang (LongLLMLingua) — compression stretch goal |
| 9 | P4 | Run full eval on 150 queries, save `results/final.json` |
| 9 | P2 | Tune `min_score` threshold for SMR-inspired stop rule |
| 10 | ALL | ⏱️ **MID-PROJECT REVIEW (60 min)** — look at the table together. Are these the final numbers? |

---

## Week 3 — Write & Polish (Days 11–15)

| Day | Who | Task |
|-----|-----|------|
| 11 | ALL | 🔒 **LOCK NUMBERS** — no more controller changes. Commit `results/final.json` |
| 12 | P1 | Draft: Background & Related Work (RAG + Lost-in-Middle + Adaptive-RAG + SMR) |
| 12 | P2 | Draft: Method — controller rules, pipeline figure |
| 12 | P3 | Draft: Method — compression (passthrough) + stitch Introduction |
| 12 | P4 | Draft: Experiments & Results — table + metrics explanation |
| 13–14 | ALL | Cross-review drafts + co-write Introduction, Conclusion, Limitations |
| 15–16 | ALL | Slide deck (2 slides/person) + full dry-run timed |

---

## Submission Checklist

- [ ] `git clone` on fresh machine → `pytest` passes → `run_pilot.py --n 3` runs
- [ ] `results/final.json` matches numbers in the paper's table
- [ ] All 4 authors named in report and slide deck
- [ ] `docs/reflections.md` — one paragraph per person
- [ ] Slide deck PDF exported
- [ ] GitHub repo link in paper footnote
- [ ] HuggingFace dataset link in paper footnote

---

## Report Sections

| Section | Owner | Key content |
|---------|-------|-------------|
| Abstract | ALL | Problem, method, main finding — 5–7 sentences |
| Introduction | ALL (P3 stitches) | Problem → contribution → paper map |
| Background & Related Work | P1 | RAG basics + Lost-in-Middle + Adaptive-RAG + SMR |
| Method | P2 + P3 | Pipeline, controller rules, compression |
| Experiments | P4 | Dataset, model, configs, results table |
| Discussion | P2 | What the numbers mean, where heuristic fails |
| Limitations | P1 | Honest weaknesses — examiners look for this |
| Conclusion | ALL | 1 paragraph — what we learned, not what we did |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Ollama won't install on someone's laptop | Use `EchoLLM` for dev; share `results/final.json` over repo |
| Heuristic worse than fixed_10 | Report it honestly — negative result + honest discussion = valid paper |
| One person disappears for an exam | Backup person ships minimum version of their slice |
| Writing left to last day | Day 11 = lock numbers. Day 12 = drafting starts. Non-negotiable. |
