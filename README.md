# Camden FOI RAG

A RAG system with eval dashboard. 

A tool for earching Freedom of Information (FOI) responses. Trained on real requests ( publically available) with hand written golden dataset, and a dashboard to evaluate performance. 

User Input: natural-language question (+ optional date range)
Output: answer with inline [SOURCE N] citations + list of cited FOI references, titles, page numbers, chunk IDs

Every query is logged to Postgres with a full trace (retrieval candidates, reranked candidates, prompt, response, models)

## To run
```
# .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL
docker compose up -d db
docker compose exec -T db psql -U foi -d foi < src/db/schema.sql
uv run python scripts/ingest_all.py camden_foi_random_pdfs
uv run uvicorn src.api.main:app --port 8000
uv run streamlit run scripts/app.py
uv run python3 scripts/run_eval.py
```


## Pipeline
| Stage | What | File |
|---|---|---|
| Extract | PyMuPDF, page-level text, empty pages dropped | `src/ingestion/extractor.py` |
| Metadata | FOI reference, date, title, response text from CSV | `src/ingestion/metadata.py` |
| Chunk | sentence boundaries, max 800 tokens (tiktoken cl100k_base), 1-sentence overlap, per page | `src/ingestion/chunker.py` |
| Embed | OpenAI `text-embedding-3-small`, 1,536 dims, batches of 128 | `src/ingestion/embedder.py` |
| Store | Postgres 16 + pgvector, HNSW index (cosine), GIN full-text index | `src/db/schema.sql` |
| Ingest | SHA-256 file hash, skips unchanged files, ingestion log table | `src/ingestion/pipeline.py`, `scripts/ingest_all.py` |
| Retrieve | vector search, top 50 candidates, optional date filter | `src/retrieval/search.py` |
| Rerank | Claude Sonnet 4.6 (`claude-sonnet-4-6`) returns ordered indices, keep top 5 | `src/retrieval/reranker.py` |
| Generate | Claude Sonnet 4.6, "use only the sources", inline [SOURCE N] | `src/retrieval/generator.py` |
| Hybrid search | vector + Postgres full-text, reciprocal rank fusion (k=60); built and tested, not used in the API | `src/retrieval/search.py` |

## Stack
Python 3.13, uv, FastAPI, asyncpg, Postgres 16 + pgvector (Docker Compose), OpenAI embeddings, Anthropic API, PyMuPDF, tiktoken, pandas, Streamlit, sentence-transformers (MiniLM comparison)

## API endpoints
- `POST /query`: `{query, top_k=5, date_from, date_to}` → `{answer, citations, query_id}`
- `GET /logs/{query_id}`: full trace for one query
- `GET /documents`: indexed documents
- `GET /stats`: document count, chunk count, last ingest time, average chunk tokens
- `GET /health`

## Dashboard
`scripts/app.py` (Streamlit)
4 tabs: 
1. Ask
2. Observability
3. Evals (batch eval against a golden CSV) 
4. Performance

## Corpus
- Repo: 40 FOI PDFs in `camden_foi_random_pdfs/pdfs/` (+ metadata CSV)
- DB as evaluated: 1,040 FOI PDFs → 3,171 chunks, average 370 tokens per chunk, max 1,381
- The extra 1,000 PDFs: `new_batch/pdfs` is a symlink to a local folder outside the repo (`rag-foi`), so it is broken for anyone cloning the repo

## Evaluation data
| File | Rows | Unique queries | Labels | Source |
|---|---|---|---|---|
| `eval_data/hand_labelled_queries.csv` | 91 | 28 | 0 / 2 / 4 (20 / 46 / 25 rows) | hand labelled (`scripts/cli_human_evaluator.py`) |
| `eval_data/synthetic_chunk_query_pairs.csv` | 291 | 287 | 0 / 2 / 4 (97 each) | Claude-generated, 40 FOIs |
| `eval_data/augmented_queries.csv` | 4,370 | 1,684 | 0 / 2 / 4 (1,661 / 838 / 1,871) | LLM-augmented (`scripts/bootstrap_augmented_dataset.py`) |

- Relevant = score ≥ 2. Chunk-level matching on chunk UUID.
- Metrics (`src/retrieval/metrics.py`): MAP, NDCG@k (graded gain 2^rel − 1), precision/recall/F1 @ 1, 3, 5, 10. "MAR" here = mean of recall@1, 3, 5, 10 per query, averaged over queries.
- Runner: `scripts/run_eval.py` calls the live API, scores the top 5 reranked chunks.
- All recorded results below use the 28-query hand-labelled set.

## Results: 28 hand-labelled queries, 1,040 FOIs / 3,171 chunks
| Run | Retrieval top-k | Reranker | MAP | MAR | NDCG@5 | P@1 | R@1 | P@5 | R@5 | F1@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (2026-04-17) | 20 | Haiku 4.5 | 0.505 | 0.512 | 0.580 | 0.607 | 0.378 | 0.207 | 0.562 | 0.292 |
| More candidates | 50 | Haiku 4.5 | 0.503 | 0.499 | 0.564 | 0.643 | 0.396 | 0.193 | 0.539 | 0.275 |
| Hybrid search (RRF) | 20 | Haiku 4.5 | 0.469 | 0.490 | 0.535 | 0.536 | 0.342 | 0.200 | 0.557 | 0.285 |
| Sonnet reranker | 20 | Sonnet 4.6 (first 400 chars of each chunk) | 0.668 | 0.655 | 0.730 | 0.750 | 0.467 | 0.279 | 0.744 | 0.387 |
| Current (2026-10-30) | 50 | Sonnet 4.6 (full chunk) | **0.823** | **0.768** | **0.903** | **0.964** | **0.594** | **0.307** | **0.827** | **0.428** |

- Current run: 3 min 53 s for 28 queries (~8.3 s/query), 0 failures. Output: `docs/eval_runs/2026-10-30_sonnet-rerank_k50_full-chunk.txt`
- Last row changed two things at once (top-k 20 → 50, reranker input 400 chars → full chunk)
- Max possible P@5 is about 0.5: queries average 2.5 relevant chunks (71 rows scored ≥ 2 across 28 queries)
- Full log of changes: `docs/lab_notes.md`

## Earlier experiments ( from `docs/experiments.md`; 17 queries at the time)
- Corpus scaling (MiniLM, chunk-level): 97 → 3,171 chunks: P@1 0.941 → 0.588, R@5 0.836 → 0.595, MRR 0.971 → 0.711
- Chunk size (fixed-size token chunker, 20% overlap, doc-level eval, 40 FOIs): 128 / 256 / 512 / 1024 tokens → MRR 0.912 / 0.941 / 0.961 / 0.971
- Embedding model (chunk 256, doc-level): text-embedding-3-small MRR 0.941 vs all-MiniLM-L6-v2 MRR 1.000

## Known limitations
- 28 labelled queries; covers 28 of 1,040 FOIs
- Queries written after reading the documents
- PDF tables scramble on extraction
- Scanned PDFs (no text layer) produce no chunks
- Gold labels keyed on chunk UUIDs, which change on re-ingest
- OpenAI / Anthropic calls are synchronous inside async endpoints
- No unit tests
- `docker-compose.yml` `api` service has no Dockerfile; only `db` runs in Docker

## To do
- per-step timing, token costs, and unit tests
