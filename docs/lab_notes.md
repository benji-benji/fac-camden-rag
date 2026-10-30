# Lab Notes

Running log of changes made to the RAG system and their measured impact.

**Format for each entry:** date, change, rationale, metrics before/after, observations, next.

For detailed baseline experiment results see [experiments.md](experiments.md).

---

## Baseline (2026-04-17)

**System config at the start of this log:**

| Component | Setting |
|-----------|---------|
| Corpus | 1,040 FOI PDFs → 3,171 chunks |
| Chunker | 800 tokens max, 1-sentence overlap, sentence boundary |
| Embedder | OpenAI `text-embedding-3-small` (1,536 dim) |
| Vector index | pgvector HNSW, cosine |
| Retrieval top_k | 20 |
| Reranker | Claude Haiku 4.5 |
| Rerank top_k | 5 |
| Generator | Claude Sonnet 4.6 |

**Baseline metrics** (Izaak harness, MiniLM re-embedded for comparison, 17 hand-labelled queries, chunk-level eval):

| k  | P@k   | R@k   | MRR   |
|----|-------|-------|-------|
| 1  | 0.588 | 0.426 | 0.588 |
| 5  | 0.200 | 0.595 | 0.711 |
| 10 | 0.124 | 0.742 | 0.727 |

---

## Planned changes (priority order)

Levers we've identified for improving retrieval quality, ordered by effort vs expected impact:

- [ ] **Bump retrieval top_k 20 → 50** — give the reranker more to choose from. One-line edit.
- [ ] **Hybrid search (BM25 + vector)** — add PostgreSQL full-text, blend scores. Biggest expected win.
- [ ] **Try MiniLM embeddings in production** — beat OpenAI in Experiment 3 on the small corpus. Needs re-embedding.
- [ ] **Query rewriting** — Claude rewrites query into variants, retrieve for each, merge.
- [ ] **Upgrade reranker to Sonnet** — low effort, moderate cost.
- [ ] **Chunk size change** — experiments favoured 512-1024; current is 800, likely marginal.

---

## Entries

### Template

```markdown
2026-04-17 13:58:08 — changing top k

**Change:** increase top k 
**Why:** improve retrieval 
**Before:** 
======================================================================
k at 20 | Aggregate metrics over 28 queries
======================================================================
  MAP     0.505
  MAR     0.512
  NDCG@5  0.580
  P@1     0.607
  R@1     0.378
  P@5     0.207
  R@5     0.562
  F1@5    0.292

**After:** 
======================================================================
k at 50 | Aggregate metrics over 28 queries
======================================================================
  MAP     0.503
  MAR     0.499
  NDCG@5  0.564
  P@1     0.643
  R@1     0.396
  P@5     0.193
  R@5     0.539
  F1@5    0.275


**Observations:** 



Metric	Before	After	Delta
P@1	0.607	0.643	+5.9% ✓
R@1	0.378	0.396	+4.8% ✓
MAP	0.505	0.503	-0.4%
NDCG@5	0.580	0.564	-2.8% ✗
P@5	0.207	0.193	-6.8% ✗
R@5	0.562	0.539	-4.1% ✗
F1@5	0.292	0.275	-5.8% ✗


P@1 went up 
but P@5 went down 

f1 at 5 went down 

increasing top k didnt help 

P@1 improved (+5.9%) — reranker has more candidates. 
But P@5/R@5/NDCG@5 all dropped slightly. 
Reranker is making more errors in slots 2-5 with 50 candidates. 
Net: slightly worse for the full top-5 used by generator. 28 queries is small — may be noise.

**Next:** what this unlocked or suggests trying next
```

---

<!-- Add new entries below this line, newest first -->

2026-04-17 14:39:27
**Change:** added hybrid search 
**Why:** to imporve precision 
**Before:** 
======================================================================
k at 20 | Aggregate metrics over 28 queries
======================================================================
  MAP     0.505
  MAR     0.512
  NDCG@5  0.580
  P@1     0.607
  R@1     0.378
  P@5     0.207
  R@5     0.562
  F1@5    0.292

**After:**
======================================================================
Aggregate metrics over 28 queries
======================================================================
  MAP     0.469
  MAR     0.490
  NDCG@5  0.535
  P@1     0.536
  R@1     0.342
  P@5     0.200
  R@5     0.557
  F1@5    0.285

  **Observations:** 

  results much worse on all metrics 
  too much similarity for key word search (FOI documents share tons of vocabulary )
  Natural-language queries aren't keyword-heavy 
  RRF with equal weighting - dragged vector search influence down, when actually vector is better signal 


 ### 2026-04-17 14:45:57 — reranker changed to sonnet 

**Change:** changed reranker to better model 
**Why:** improve precision 
**Before:** 
======================================================================
k at 20 | Aggregate metrics over 28 queries
======================================================================
  MAP     0.505
  MAR     0.512
  NDCG@5  0.580
  P@1     0.607
  R@1     0.378
  P@5     0.207
  R@5     0.562
  F1@5    0.292

**After:** 
======================================================================
Reranker changed to Sonnet: Aggregate metrics over 28 queries
======================================================================
  MAP     0.668
  MAR     0.655
  NDCG@5  0.730
  P@1     0.750
  R@1     0.467
  P@5     0.279
  R@5     0.744
  F1@5    0.387


**Observations:** 
All 8 metrics improved 24-35%. Biggest single gain in any experiment so far.
Sonnet 5-10x more expensive than Haiku and ~2-3x slower per call. Worth it.

**Next:** what this unlocked or suggests trying next
 
 ### 2026-04-17 14:52:27— try top-k higher

**Change:** inrease top-k
**Why:** now ive switched reranker, should improve 
**Before:** key metric(s) before the change
**After:** key metric(s) after the change
**Observations:** what surprised you / didn't work / caveats
**Next:** what this unlocked or suggests trying next