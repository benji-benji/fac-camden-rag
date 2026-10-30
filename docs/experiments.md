# RAG Experiment Results

Summary of retrieval evaluation experiments run against the Camden FOI corpus.

- **Ground truth:** 17 hand-labelled queries in `eval_data/hand_labelled_queries.csv`, each scoring 2-5 chunks as 0 (irrelevant), 2 (medium), or 4 (very relevant). Relevant threshold = 2.
- **Metrics:** Precision@k, Recall@k, MRR (Mean Reciprocal Rank).
- **Two corpus sizes:** original 40 FOIs (97 chunks) and expanded 1,040 FOIs (3,171 chunks).

---

## Experiment 1 — Vary k (corpus scaling)

How does retrieval quality change when we return more results, and how does it change when the corpus grows 32×?

**Setup:** MiniLM-L6-v2 embeddings, chunk-level eval (matching chunk UUIDs).

### 97-chunk corpus (40 FOIs)

|   k |   P@k |   R@k |   MRR |
|-----|-------|-------|-------|
|   1 | 0.941 | 0.575 | 0.941 |
|   3 | 0.471 | 0.813 | 0.971 |
|   5 | 0.306 | 0.836 | 0.971 |
|  10 | 0.171 | 0.892 | 0.971 |
|  20 | 0.088 | 0.907 | 0.971 |

### 3,171-chunk corpus (1,040 FOIs)

|   k |   P@k |   R@k |   MRR |
|-----|-------|-------|-------|
|   1 | 0.588 | 0.426 | 0.588 |
|   3 | 0.275 | 0.517 | 0.696 |
|   5 | 0.200 | 0.595 | 0.711 |
|  10 | 0.124 | 0.742 | 0.727 |
|  20 | 0.065 | 0.772 | 0.727 |

### Corpus scaling impact (k=1 and k=5)

| Metric | 97 chunks | 3,171 chunks | Change |
|--------|-----------|--------------|--------|
| P@1    | 0.941     | 0.588        | **−37%** |
| R@5    | 0.836     | 0.595        | **−29%** |
| MRR@5  | 0.971     | 0.711        | **−27%** |

**Read:** At 97 chunks, the relevant chunk was at rank 1 for 94% of queries. At 3,171 chunks it drops to 59%. The queries and labels are identical — only the haystack changed. Scaling up exposes a retrieval quality issue that was hidden on the small corpus.

---

## Experiment 2 — Vary chunk size

How does chunk size affect retrieval? Re-chunked the original 40 PDFs at token-based sizes with 20% overlap.

**Setup:** OpenAI text-embedding-3-small, k=5, doc-level eval (match via `foi_reference` because chunk IDs differ per chunk size).

| Chunk size | Total chunks | P@5   | R@5   | MRR   |
|------------|--------------|-------|-------|-------|
| 128        | 345          | 0.480 | 0.941 | 0.912 |
| 256        | 176          | 0.333 | 0.941 | 0.941 |
| 512        | 95           | 0.340 | 1.000 | 0.961 |
| 1024       | 49           | 0.242 | 1.000 | 0.971 |

**Read:** Larger chunks produce better MRR and recall. Makes sense for FOI responses — short letters that answer a single question. A 1024-token chunk captures the whole response, so *any* query about that document produces a strong similarity signal. Splitting into 128-token chunks fragments the answer and weakens the match.

The current production chunker uses 800-token max with sentence boundaries — close to the sweet spot indicated here.

---

## Experiment 3 — Vary embedding model

Compare OpenAI's hosted embeddings vs a free local model.

**Setup:** chunk_size=256, k=5, doc-level eval.

| Model                         | Dimensions | P@5   | R@5   | MRR   |
|-------------------------------|------------|-------|-------|-------|
| text-embedding-3-small (OpenAI) | 1536       | 0.333 | 0.941 | 0.941 |
| all-MiniLM-L6-v2 (local)      | 384        | 0.375 | 1.000 | **1.000** |

**Read:** The smaller, free, local model outperformed OpenAI on this corpus. MRR = 1.0 means a relevant chunk was at rank 1 for every single query.

Counterintuitive but explainable — MiniLM is purpose-built for sentence similarity, while `text-embedding-3-small` is a general-purpose model. For small specialised corpora, purpose-built often wins.

**Caveat:** 17 queries is a small sample. One or two re-labelled queries would knock MRR below 1.0. Do not over-interpret this.

**Practical implication:** Swapping the production system from OpenAI to MiniLM for retrieval would:
- Remove a paid API call from every query
- Remove network latency (~100-300ms per query)
- Remove the API key dependency for retrieval
- Deliver equal-or-better retrieval on this corpus

Worth testing with the expanded corpus and more labelled queries before making the switch.

---

## Methodology notes

**Why two eval styles?**
- Chunk-level (UUID matching) — strict, requires exact chunk match. Used for production corpus comparisons where chunk IDs are stable.
- Doc-level (FOI reference matching) — coarser, counts any chunk from the relevant document. Used when comparing different chunking strategies (chunk IDs change when you re-chunk).

**Why both MiniLM and OpenAI?**
- Production system uses OpenAI (ingestion pipeline embeddings stored in pgvector).
- Izaak harness uses MiniLM (re-embeds DB chunk text locally).
- Experiment 3 compares them head-to-head on the same chunks.

**Known limitations of these results**
1. Only 17 queries. Single outliers materially move averages.
2. Hand-labelled ground truth covers 17 of 40 original FOIs. The expanded 1,040-doc corpus has labels for <2% of documents.
3. Queries were written after seeing the documents (though before seeing chunks), so inherent "answerable" bias.
4. PDF extraction quality affects retrieval — known issue with tables being scrambled (see `docs/project.md`).

## Next steps

- Expand hand-labelled ground truth to 20-40 queries before drawing stronger conclusions.
- Build an augmented eval set: LLM-generated queries calibrated on the hand-labelled examples, spot-checked by hand.
- Re-run Experiment 2 and 3 on the 1,040-doc corpus once ground truth has wider coverage.
