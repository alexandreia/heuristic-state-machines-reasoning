# Adaptive Retrieval for Token-Efficient RAG: A Heuristic Approach on SciFact

**Authors:** [Name 1] · [Name 2] · [Name 3] · [Name 4]
**Course:** Language Technology / Information Retrieval
**Repository:** [GitHub link]
**Dataset:** [HuggingFace link]

---

## Abstract

*~5–7 sentences. Write this last.*

Retrieval-Augmented Generation (RAG) systems typically retrieve a fixed number of documents per query, regardless of complexity. We investigate whether a lightweight heuristic controller can adaptively select the number of retrieved documents, reducing prompt token usage without significantly hurting answer quality. We evaluate four configurations — closed-book, fixed-5, fixed-10, and our heuristic — on a 150-query sample of SciFact using Mistral 7B. Our heuristic reduces average prompt tokens by [X]% compared to fixed-10, while achieving a Token F1 of [X] and nDCG@10 of [X]. These results suggest that [positive/negative finding in one sentence].

---

## 1. Introduction

*~1 page. Write this last, after results are in.*

### Problem

Retrieval-Augmented Generation (RAG) has become a standard approach for grounding large language model (LLM) outputs in external knowledge [Lewis et al., 2020]. However, most RAG pipelines retrieve a fixed number of documents — typically the top-k — regardless of the complexity of the input query. This leads to unnecessary token consumption: simple factual queries receive the same context budget as complex multi-hop questions.

Liu et al. [2023] demonstrate that LLMs struggle to effectively use long contexts, often ignoring documents positioned in the middle of the prompt. This suggests that retrieving more documents does not always improve answer quality, and may even hurt it.

### Contribution

We build a small RAG system that *adaptively* selects how many documents to retrieve per query, based on (1) query length as a proxy for complexity, and (2) retrieval confidence as a proxy for corpus relevance — inspired by the Stop action in State Machine Reasoning [Lee et al., 2025]. We compare this heuristic controller against three baselines on the SciFact biomedical fact-checking dataset.

### Summary of findings

*[Fill in after results are complete.]*

In this paper, we show that [summary of main result in 1–2 sentences].

---

## 2. Related Work

*~¾ page. Owner: P1.*

### Retrieval-Augmented Generation

[Lewis et al., 2020] introduced RAG as a framework for combining parametric and non-parametric memory in LLMs. A dense retriever encodes documents and queries into a shared vector space; at inference time, the top-k most similar documents are prepended to the prompt. We adopt this architecture as our baseline.

### The Cost of Long Contexts

Liu et al. [2023] show that LLMs exhibit a "lost in the middle" phenomenon: performance degrades when relevant information appears in the middle of long contexts, with models attending primarily to the beginning and end. This motivates *reducing* the number of retrieved documents when possible, rather than always maximising context.

### Adaptive Retrieval

Jeong et al. [2024] propose Adaptive-RAG, which learns to route queries to different retrieval strategies based on predicted complexity. Our work is simpler and training-free: rather than routing between retrieval strategies, we adapt the *number* of documents retrieved per query using a hand-written heuristic.

### Token-Efficient Reasoning for IR

Lee et al. [2025] propose State Machine Reasoning (SMR), which decomposes retrieval reasoning into discrete actions (Refine, Rerank, Stop), achieving a 74.4% reduction in token usage on BRIGHT. We draw inspiration from their Stop action, which terminates retrieval when no further gain is expected. Our heuristic approximates this with a simple cosine similarity threshold, requiring no LLM calls for the control decision.

---

## 3. Method

*~¾ page. Owner: P2 + P3.*

### Pipeline Overview

Our pipeline consists of four components: a data loader, a dense retriever, a heuristic controller, and a local LLM. At inference time, the controller first inspects the query and the top retrieval score, then decides how many documents to pass to the LLM.

```
Query → Controller → Retriever (k docs) → Mistral 7B → Answer
```

### Dense Retriever

We use `all-MiniLM-L6-v2` from sentence-transformers [CITATION] to encode both queries and documents into 384-dimensional vectors. Cosine similarity is computed as a dot product after L2 normalisation. Document embeddings are computed once and cached to disk.

### Heuristic Controller

Our controller applies two rules in order:

**Rule 1 — Relevance score:** If the cosine similarity of the top retrieved document falls below a threshold τ = 0.4, we set k = 0 (closed-book). This is inspired by the Stop action of SMR [Lee et al., 2025]: if the corpus is unlikely to contain a useful answer, retrieving documents wastes tokens without benefit.

**Rule 2 — Query length:** For queries with ≤ 6 words, we set k = 3 (short queries tend to be factual lookups). For longer queries, k = 8 (complex queries may require broader evidence).

### Language Model

We use Mistral 7B Instruct (`mistral:7b-instruct`) served locally via Ollama [CITATION]. We fix `temperature = 0` and `seed = 0` for reproducibility. Token counts are taken directly from Ollama's `prompt_eval_count` field.

### Compression

*[If implemented: describe extractive compressor. Otherwise:]*
We pass retrieved documents to the LLM without compression. Context compression remains a direction for future work [Xu et al., 2024].

---

## 4. Data

*~½ page. Owner: P4.*

### Dataset

We evaluate on **SciFact** [Wadden et al., 2020], a biomedical fact-checking dataset from the BEIR benchmark [Thakur et al., 2021]. SciFact consists of 5,183 scientific abstracts and 300 test queries, each paired with relevance judgements (qrels) indicating which abstracts support or refute each claim.

### Evaluation Sample

Running a full local LLM pipeline over all 300 queries is computationally expensive. We sample a fixed subset of **150 queries** from the test set using `random.seed(0)`, ensuring every team member works with the same queries. The sample is hosted at [HuggingFace link].

### Data Split

We use only the test split of SciFact, as our pipeline involves no training. The full 5,183-document corpus is used as the retrieval pool for all configurations.

---

## 5. Experiments

*~½ page. Owner: P4.*

### Configurations

We compare four configurations:

| Config | Description | k |
|--------|-------------|---|
| `no_retrieval` | Closed-book baseline — Mistral answers alone | 0 |
| `fixed_5` | Always retrieve 5 documents | 5 |
| `fixed_10` | Always retrieve 10 documents | 10 |
| `heuristic` | Adaptive: length rule + confidence threshold | 0–8 |

### Evaluation Metrics

We report three metrics: **Token F1** (token overlap between model answer and gold document text), **nDCG@10** (retrieval quality), and **average prompt tokens** (token efficiency, from Ollama's `prompt_eval_count`).

### Results

*[Fill in after running the experiment.]*

| Config | Token F1 | nDCG@10 | Avg Prompt Tokens | Avg k |
|--------|----------|---------|-------------------|-------|
| `no_retrieval` | — | — | — | 0 |
| `fixed_5` | — | — | — | 5 |
| `fixed_10` | — | — | — | 10 |
| `heuristic` | — | — | — | — |

---

## 6. Discussion

*~½ page. Owner: P2.*

*[Fill in after results are complete. Answer these questions:]*

- Does the heuristic save tokens compared to fixed_10? By how much?
- Does it hurt answer quality (Token F1)?
- Does it hurt retrieval quality (nDCG@10)?
- Where does it fail? (Look at individual queries where heuristic chose k=0 but should have retrieved.)
- Was the result expected? What surprised you?

---

## 7. Limitations

*~¼ page. Owner: P1.*

Our approach has several limitations. First, query length is a noisy proxy for complexity — a short query can be highly ambiguous, and a long query can be straightforward. Second, the cosine similarity threshold was set heuristically and may not generalise to other datasets. Third, we evaluate on a single domain (biomedical), which limits the generalisability of our findings. Fourth, Token F1 against the gold document text is an imperfect answer quality metric; a model answer may be correct without overlapping with the exact wording of the gold document. Finally, we do not explore learned controllers, context compression, or multi-turn conversation, which remain open directions for future work.

---

## 8. Ethical Considerations

*~2–3 sentences.*

SciFact contains scientific claims about biomedical topics. A system that retrieves and presents scientific evidence incorrectly could mislead users about medical facts. Our system is intended for research purposes only and should not be used for medical decision-making. We are not aware of privacy concerns in the dataset, as all documents are published scientific abstracts.

---

## 9. Author Contributions

*Fill in after project is complete.*

- **[Name 1] (P1):** Implemented `scifact_loader.py` and `dense_retriever.py`. Wrote Background & Related Work and Limitations sections.
- **[Name 2] (P2):** Implemented `heuristic.py`. Wrote Method (controller) and Discussion sections.
- **[Name 3] (P3):** Implemented compression module. Stitched Introduction. Wrote Method (compression) section.
- **[Name 4] (P4):** Implemented `ollama_client.py`, `metrics.py`, and `run_pilot.py`. Ran all experiments. Wrote Experiments & Results section.
- All authors co-wrote the Abstract and Conclusion, and contributed to the slide deck.

---

## 10. AI Contributions

Claude (Anthropic) was used to scaffold the initial code structure for `scifact_loader.py`, `dense_retriever.py`, `heuristic.py`, `ollama_client.py`, `metrics.py`, and `run_pilot.py`. All generated code was reviewed, tested, and modified by the team. Claude was also used to explain paper concepts and debug errors. The report text, analysis, and conclusions were written by the authors.

---

## 11. Conclusions

*1 short paragraph. Write this last.*

We investigated whether a simple, training-free heuristic controller can reduce token usage in RAG without significantly hurting answer quality on SciFact. Our results show that [finding]. This suggests that [broader implication]. Future work could explore learned controllers [Jeong et al., 2024], context compression [Xu et al., 2024], or richer query complexity signals beyond length.

---

## References

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Liu, N. F., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.
- Jeong, S., et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. *NAACL*.
- Asai, A., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR*.
- Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE). *EMNLP*.
- Lee, D., et al. (2025). From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval.
- Xu, F., et al. (2024). RECOMP: Improving Retrieval-Augmented LMs with Context Compression. *ICLR*.
- Wadden, D., et al. (2020). Fact or Fiction: Verifying Scientific Claims. *EMNLP*.
- Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS*.
