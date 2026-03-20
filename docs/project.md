# FOI RAG — Project Documentation

A RAG (Retrieval-Augmented Generation) system for querying Camden Council Freedom of Information request PDFs. Built as a learning project.

---

## Scope

A local tool that lets you ask natural-language questions against 40 Camden Council FOI PDFs and get cited, grounded answers. Includes an evaluation harness to measure and track quality over time.

**What it does:**
- Ingests FOI PDFs → extracts text → chunks → embeds → stores in pgvector
- Answers queries using vector search + LLM reranking + Claude Sonnet generation with inline citations
- Evaluates itself: performance (latency/cost), quality (precision/recall/F1/groundedness), and regression testing

**What it is not:**
- A production service
- A multi-user system
- A general-purpose search engine

---

## Tech stack

| Layer | Tool |
|-------|------|
| Language | Python 3.13 |
| Package manager | uv |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1536d) |
| Reranker | Claude Haiku |
| Generator | Claude Sonnet |
| API | FastAPI + uvicorn |
| UI | Streamlit (local only) |
| Infrastructure | Docker Compose |

---

## Project structure

```
foi-rag/
├── camden_foi_random_pdfs/
│   ├── downloaded_pdf_metadata.csv
│   └── pdfs/               # 40 FOI PDFs named NN_REFID_title.pdf
├── docs/
│   ├── workflow.md         # build log for the core RAG pipeline
│   └── project.md          # this file
├── logs/                   # eval run logs (CSV + baseline JSON)
├── scripts/
│   ├── download_pdfs.py
│   ├── ingest_all.py
│   ├── generate_ground_truth.py   # TO BUILD
│   ├── eval_pipeline.py           # TO BUILD
│   └── app.py                     # TO BUILD (Streamlit)
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── schema.sql
│   ├── eval/                      # TO BUILD
│   │   ├── __init__.py
│   │   ├── performance.py
│   │   ├── quality.py
│   │   ├── ground_truth.py
│   │   └── regression.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── extractor.py
│   │   ├── metadata.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── reranker.py
│   │   └── search.py
│   └── tests/                     # TO BUILD
│       ├── test_chunker.py
│       ├── test_quality.py
│       └── test_metadata.py
├── .env                    # not committed
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

---

## What's already built

### Corpus
40 FOI PDFs from Camden Council. Metadata in `downloaded_pdf_metadata.csv`.

### Database (pgvector on Docker)
Four tables: `documents`, `chunks`, `ingestion_log`, `query_logs`.
HNSW index on `chunks.embedding` for fast cosine similarity search.

Start the database:
```bash
docker compose up -d db
```

### Ingestion pipeline
`scripts/ingest_all.py` → runs all 40 PDFs through:
1. `metadata.py` — loads CSV metadata keyed by filename
2. `extractor.py` — PyMuPDF page-by-page text extraction
3. `chunker.py` — splits pages at sentence boundaries, max 800 tokens, 1-sentence overlap
4. `embedder.py` — OpenAI `text-embedding-3-small` in batches of 128
5. `pipeline.py` — SHA-256 idempotency, upserts documents, inserts chunks in a transaction

Result: 40 documents, 97 chunks, avg 350 tokens per chunk.

Re-run safely — skips already-indexed files:
```bash
uv run python3 scripts/ingest_all.py
```

### Retrieval pipeline
Query → embed → vector search (cosine, pgvector) → Claude Haiku rerank → Claude Sonnet generate with `[SOURCE N]` citations.

Supports optional `date_from` / `date_to` filters.

### FastAPI
`src/api/main.py` — two endpoints:
- `GET /health`
- `POST /query` → `{query, top_k, date_from, date_to}` → `{answer, citations, query_id}`

All queries logged to `query_logs`.

Run:
```bash
uv run uvicorn src.api.main:app --reload
```

---

## What to build next: Eval Pipeline + Streamlit UI

### Build order (do not skip ahead)

```
1.  schema.sql        → add ground_truth table
2.  generate_ground_truth.py  → LLM-generated golden dataset
3.  src/eval/quality.py       → precision, recall, F1 (pure functions first)
4.  src/eval/performance.py   → timing + cost measurement
5.  src/eval/ground_truth.py  → loads golden data from db
6.  src/tests/                → unit tests for chunker, metadata, quality functions
7.  src/eval/regression.py    → baseline save + compare
8.  scripts/eval_pipeline.py  → CLI runner
9.  scripts/app.py            → Streamlit UI last
```

---

## Step-by-step: Golden Dataset

The golden dataset is the foundation of all quality eval. Build it first.

### What it is

For each of the 97 chunks in the database:
- **1 relevant query** — a question this chunk directly answers
- **2 irrelevant queries** — plausible Camden Council questions this chunk does NOT answer

Total: 291 query–chunk pairs (97 relevant, 194 irrelevant).

### Step 1 — add the table

Add to `src/db/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS ground_truth (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id     UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    query        TEXT NOT NULL,
    is_relevant  BOOLEAN NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);
```

Apply:
```bash
docker compose exec -T db psql -U foi -d foi < src/db/schema.sql
```

**Test — table exists:**
```bash
docker compose exec db psql -U foi -d foi -c "\dt"
```
Expected: `ground_truth` appears in the list.

### Step 2 — explore your chunks before generating

Always read what you're working with first:

```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv; load_dotenv()
import asyncpg

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    async with pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT c.id, d.foi_reference, d.title, c.page_number, c.token_count,
                   LEFT(c.content, 150) as preview
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY d.foi_reference, c.page_number
            LIMIT 10
        ''')
    for r in rows:
        print(f'{r[\"foi_reference\"]} | p{r[\"page_number\"]} | {r[\"token_count\"]}t')
        print(f'  {r[\"preview\"]}')
        print()
    await pool.close()

asyncio.run(main())
"
```

### Step 3 — write `scripts/generate_ground_truth.py`

For each chunk, call Claude Haiku with a prompt like:

> "Here is a chunk from a Camden Council FOI document. Generate exactly 3 queries: one that this chunk directly answers, and two plausible questions about Camden Council that this chunk does NOT answer. Return only JSON: `{\"relevant\": \"...\", \"irrelevant\": [\"...\", \"...\"]}`"

Add `--limit N` and `--dry-run` flags. Always test on 5 before running on all 97.

**Test — dry run:**
```bash
uv run python3 scripts/generate_ground_truth.py --limit 5 --dry-run
```
Expected: prints 5 chunk previews, makes no API calls.

**Test — real run on 5:**
```bash
uv run python3 scripts/generate_ground_truth.py --limit 5
```

**Test — verify database contents:**
```bash
docker compose exec db psql -U foi -d foi -c "
SELECT is_relevant, COUNT(*) FROM ground_truth GROUP BY is_relevant;
"
```
Expected after 5 chunks:
```
 is_relevant | count
-------------+-------
 t           |     5
 f           |    10
```

Expected after all 97 chunks:
```
 is_relevant | count
-------------+-------
 t           |    97
 f           |   194
```

**Important:** Read the generated queries before running on all 97. The irrelevant queries must be genuinely irrelevant — not just vague. Fix any that look wrong.

---

## Step-by-step: Performance Eval

### Step 4 — modify embedder to return token count

The OpenAI response already has `response.usage.total_tokens`. Change `embed_texts()` to return `(embeddings, token_count)`. Update the caller in `src/api/main.py` to unpack it.

**Test:**
```bash
uv run python3 -c "
from dotenv import load_dotenv; load_dotenv()
from src.ingestion.embedder import embed_texts
embeddings, tokens = embed_texts(['temporary accommodation in Camden'])
print(f'dim={len(embeddings[0])}, tokens={tokens}')
"
```
Expected: `dim=1536, tokens=6` (approx)

### Step 5 — write `src/eval/performance.py`

Write a `run_timed_query(query, pool)` function. Use `time.perf_counter()` to time each step separately. Return a dataclass:

```
@dataclass
class PerfResult:
    query: str
    embed_ms: float
    search_ms: float
    rerank_ms: float
    generate_ms: float
    total_ms: float
    embed_tokens: int
    rerank_input_tokens: int
    rerank_output_tokens: int
    gen_input_tokens: int
    gen_output_tokens: int
    estimated_cost_usd: float
```

Hardcode current pricing:
- OpenAI `text-embedding-3-small`: $0.02 / 1M tokens
- Claude Haiku: check current Anthropic pricing page
- Claude Sonnet: check current Anthropic pricing page

**Test a timing helper in isolation first:**
```bash
uv run python3 -c "
import time

def timed(fn, *args, **kwargs):
    # write this yourself
    pass

result, elapsed = timed(time.sleep, 0.1)
print(f'elapsed: {elapsed:.3f}s')  # should be ~0.1
"
```

**Test the full function:**
```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv; load_dotenv()
import asyncpg
from src.eval.performance import run_timed_query

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    result = await run_timed_query('how many households are in temporary accommodation?', pool)
    print(result)
    await pool.close()

asyncio.run(main())
"
```

---

## Step-by-step: Quality Eval

### Step 6 — precision, recall, F1 (pure functions)

Write these in `src/eval/quality.py`. No LLM, no database — just maths.

```
precision = relevant_retrieved / total_retrieved
recall    = relevant_retrieved / total_relevant_in_golden
F1        = 2 × (precision × recall) / (precision + recall)
```

Handle edge cases: division by zero when retrieved or relevant set is empty.

**Test with hardcoded values:**
```bash
uv run python3 -c "
from src.eval.quality import precision, recall, f1

retrieved = {'a', 'b', 'c'}
relevant  = {'a', 'd'}

p = precision(retrieved, relevant)
r = recall(retrieved, relevant)
f = f1(p, r)
print(p, r, f)
# expect: 0.333, 0.5, 0.4
"
```

**Test edge cases:**
```bash
uv run python3 -c "
from src.eval.quality import precision, recall, f1

# empty retrieved
print(precision(set(), {'a', 'b'}))  # expect: 0.0

# perfect match
print(recall({'a', 'b'}, {'a', 'b'}))  # expect: 1.0

# no overlap
print(f1(0.0, 0.5))  # expect: 0.0
"
```

### Step 7 — write `run_quality_eval(query, pool)`

This should:
1. Look up `ground_truth` for the query to get expected relevant chunk IDs
2. Run embed → search → rerank
3. Compare retrieved chunk IDs to ground truth
4. Return precision, recall, F1

**Test using a query from your golden dataset:**
```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv; load_dotenv()
import asyncpg
from src.eval.quality import run_quality_eval

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    result = await run_quality_eval('YOUR GOLDEN QUERY HERE', pool)
    print(result)
    await pool.close()

asyncio.run(main())
"
```

### Step 8 — citation accuracy checker (LLM judge)

Write `check_citation_accuracy(answer, sources) -> dict` using Claude Haiku.

Prompt: "Given this answer and these source chunks, does each [SOURCE N] citation accurately reflect what the source says? Return JSON: `{1: true, 2: false, ...}`"

**Test:**
```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv; load_dotenv()
import asyncpg
from src.ingestion.embedder import embed_texts
from src.retrieval.search import vector_search
from src.retrieval.reranker import rerank
from src.retrieval.generator import generate_answer
from src.eval.quality import check_citation_accuracy

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    query = 'what enforcement action has Camden taken in the private rented sector?'
    emb, _ = embed_texts([query])
    results = await vector_search(emb[0], pool, top_k=10)
    reranked = rerank(query, results, top_k=5)
    answer = generate_answer(query, reranked)
    scores = check_citation_accuracy(answer, reranked)
    print(scores)
    await pool.close()

asyncio.run(main())
"
```

### Step 9 — groundedness score

Write `score_groundedness(answer, sources) -> float` using Claude Haiku.

Prompt: given these source chunks and this answer, what fraction of claims are supported by the sources? Return a float 0.0–1.0.

---

## Step-by-step: Regression

### Step 10 — unit tests

Create `src/tests/` with:

| Test file | What to test |
|-----------|-------------|
| `test_chunker.py` | page ≤800 tokens → single chunk; long page → multiple chunks; 1-sentence overlap preserved |
| `test_quality.py` | precision/recall/F1 edge cases: empty sets, perfect match, zero overlap |
| `test_metadata.py` | bad dates don't crash; missing fields return empty strings |

**Test:**
```bash
uv run python3 -m pytest src/tests/ -v
```

### Step 11 — baseline save and compare

Write `save_baseline(results, path)` and `compare_to_baseline(current, baseline, threshold=0.05)` in `src/eval/regression.py`.

Save to `logs/baseline.json`. Flag metrics that drop more than the threshold.

**Test:**
```bash
uv run python3 scripts/eval_pipeline.py --phase regression
```
Expected:
```
Comparing to baseline from 2026-03-13...
precision:    0.72 → 0.71  OK
recall:       0.61 → 0.60  OK
groundedness: 0.88 → 0.72  ⚠ REGRESSED
```

---

## Step-by-step: Streamlit UI

Build this last. All eval modules must work from the CLI first.

### Step 12 — add streamlit dependency

Add `streamlit` to `pyproject.toml` dependencies, then:
```bash
uv sync
```

### Step 13 — `scripts/app.py` structure

Four tabs:

```python
tab_ask, tab_perf, tab_quality, tab_regression = st.tabs([
    "Ask", "Performance", "Quality", "Regression"
])
```

#### Tab: Ask

- Text input + Submit button
- On submit: run full pipeline, display answer
- Expandable Sources section: FOI ref, title, page, chunk preview for each citation
- Show total latency and cost (reuse `run_timed_query`)

**Test:**
```bash
uv run streamlit run scripts/app.py
```
Type a query. Verify answer and citations appear.

#### Tab: Performance

- "Run Performance Eval" button
- Table: Query | Embed(ms) | Search(ms) | Rerank(ms) | Gen(ms) | Total(ms) | Cost($)
- Summary row with averages
- "Save to log" button → appends to `logs/eval_YYYY-MM-DD.csv`

#### Tab: Quality

- "Run Quality Eval" button
- `st.metric()` cards at top: avg Precision, avg Recall, avg F1
- Table: Query | Precision | Recall | F1 | Citations OK | Grounded
- Colour-coded F1: green ≥ 0.7, amber 0.4–0.7, red < 0.4

#### Tab: Regression

- `st.metric()` with `delta=` param for each metric (shows change vs baseline)
- "Set as new baseline" button
- Unit test results (pass/fail per test)

---

## Product roadmap

### In scope (current)

- [x] PDF ingestion pipeline
- [x] Vector search with pgvector HNSW index
- [x] Claude Haiku reranker
- [x] Claude Sonnet generator with inline citations
- [x] FastAPI query endpoint with query logging
- [ ] Golden dataset (97 chunks × 3 queries)
- [ ] Performance eval (latency, tokens, cost)
- [ ] Quality eval (precision, recall, F1, citation accuracy, groundedness)
- [ ] Unit test suite
- [ ] Regression baseline comparison
- [ ] Streamlit UI (Ask + 3 eval tabs)

### Stretch goals (next)

- **Dockerfile** — containerise the API so `docker compose up` starts everything
- **`/logs/{query_id}` endpoint** — retrieve full prompt and chunks for any past query
- **Hybrid search** — combine pgvector similarity with PostgreSQL full-text search (`tsvector`)
- **Query expansion** — use Claude to rewrite the question into multiple variants before retrieval
- **Conversational retrieval** — maintain history so follow-up questions resolve references correctly
- **ANN vs brute-force comparison** — compare HNSW approximate search against exact vector search for accuracy/speed tradeoff
- **Corpus expansion** — ingest more PDFs, test scaling behaviour

### Future / out of scope for now

- Multi-user authentication
- Cloud deployment
- Real-time ingestion of new FOI requests
- Fine-tuned embedding model on Camden data
