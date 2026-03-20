# FOI RAG — Build Workflow

A step-by-step record of everything built in this session, including every file created and every test run to verify it.

---

## Corpus

40 Freedom of Information request PDFs from Camden Council, downloaded using `scripts/download_pdfs.py`.

The corpus lives at `camden_foi_random_pdfs/`:
- `pdfs/` — 40 PDF files named `NN_REFID_title.pdf`
- `downloaded_pdf_metadata.csv` — metadata for each PDF with columns: `Identifier`, `Document Date`, `Document Title`, `Document Text`, `saved_filename`, and others

---

## Phase 1: Project setup

### `pyproject.toml`

```toml
[project]
name = "foi-rag"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "asyncpg",
    "openai",
    "anthropic",
    "pymupdf",
    "tiktoken",
    "python-dotenv",
    "pandas",
]
```

Install with:
```bash
uv sync
```

---

### `.env`

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DATABASE_URL=postgresql://foi:foi@localhost:5432/foi
```

Note: `localhost` is used here for running scripts locally. The API container uses `db` (Docker's internal hostname) via an override in `docker-compose.yml`.

---

### `docker-compose.yml`

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: foi
      POSTGRES_PASSWORD: foi
      POSTGRES_DB: foi
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: postgresql://foi:foi@db:5432/foi
    depends_on:
      - db

volumes:
  pgdata:
```

Start the database:
```bash
docker compose up -d db
```

**Test — verify pgvector is installed:**
```bash
docker compose exec db psql -U foi -d foi -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```
```
 extname | extversion
---------+------------
 vector  | 0.8.2
```

---

### `src/db/schema.sql`

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      TEXT NOT NULL UNIQUE,
    foi_reference TEXT,
    date          DATE,
    title         TEXT,
    response_text TEXT,
    total_pages   INT,
    indexed_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INT NOT NULL,
    chunk_index  INT NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    embedding    VECTOR(1536),
    token_count  INT,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ingestion_log (
    filename     TEXT PRIMARY KEY,
    file_hash    TEXT NOT NULL,
    status       TEXT NOT NULL,
    error        TEXT,
    ingested_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query               TEXT NOT NULL,
    filters             JSONB,
    retrieved_chunk_ids UUID[],
    prompt_sent         TEXT,
    response            TEXT,
    model               TEXT,
    queried_at          TIMESTAMPTZ DEFAULT now()
);
```

Apply the schema:
```bash
docker compose exec -T db psql -U foi -d foi < src/db/schema.sql
```

**Test — verify tables exist:**
```bash
docker compose exec db psql -U foi -d foi -c "\dt"
```
```
           List of relations
 Schema |     Name      | Type  | Owner
--------+---------------+-------+-------
 public | chunks        | table | foi
 public | documents     | table | foi
 public | ingestion_log | table | foi
 public | query_logs    | table | foi
```

**Test — verify HNSW index exists:**
```bash
docker compose exec db psql -U foi -d foi -c "\di"
```
```
 public | chunks_embedding_idx   | index | foi   | chunks
```

---

## Phase 2: Ingestion pipeline

Also create empty `src/ingestion/__init__.py` and `src/db/__init__.py`.

### `src/ingestion/metadata.py`

Loads the CSV once and returns a dict keyed by filename.

```python
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass
class MetadataRow:
    foi_reference: str
    date: date | None
    title: str
    response_text: str


def load_metadata(csv_path: Path) -> dict[str, MetadataRow]:
    result = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            filename = row["saved_filename"]
            raw_date = row.get("Document Date", "").strip()
            parsed_date = None
            if raw_date:
                try:
                    parsed_date = datetime.fromisoformat(raw_date).date()
                except ValueError:
                    pass
            result[filename] = MetadataRow(
                foi_reference=row.get("Identifier", "").strip(),
                date=parsed_date,
                title=row.get("Document Title", "").strip(),
                response_text=row.get("Document Text", "").strip(),
            )
    return result
```

**Test:**
```bash
uv run python3 -c "
from pathlib import Path
from src.ingestion.metadata import load_metadata
meta = load_metadata(Path('camden_foi_random_pdfs/downloaded_pdf_metadata.csv'))
print(f'Loaded {len(meta)} rows')
first = next(iter(meta.items()))
print(first[0])
print(first[1])
"
```
```
Loaded 40 rows
01_CAM6551_asylum seeker accommodation complaints in private sector housing.pdf
MetadataRow(foi_reference='CAM6551', date=datetime.date(2023, 12, 20), title='asylum seeker accommodation complaints in private sector housing', response_text='Date: 20/12/2023 Ref: CAM6551 ...')
```

---

### `src/ingestion/extractor.py`

Uses PyMuPDF to extract text page by page, returning only pages with content.

```python
from pathlib import Path
import fitz  # PyMuPDF


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append((page.number + 1, text))  # 1-indexed
    return pages
```

**Test:**
```bash
uv run python3 -c "
from pathlib import Path
from src.ingestion.extractor import extract_pages
pages = extract_pages(Path('camden_foi_random_pdfs/pdfs/01_CAM6551_asylum seeker accommodation complaints in private sector housing.pdf'))
print(f'{len(pages)} pages extracted')
for num, text in pages:
    print(f'--- Page {num} ({len(text)} chars) ---')
    print(text[:300])
"
```
```
2 pages extracted
--- Page 1 (1778 chars) ---
Date: 20/12/2023
Ref: CAM6551
...
--- Page 2 (944 chars) ---
Why not check our Portal Open Data Camden...
```

---

### `src/ingestion/chunker.py`

Strips whitespace, counts tokens using tiktoken, and splits long pages at sentence boundaries with 1-sentence overlap. Pages under 800 tokens are kept as a single chunk.

```python
import re
from dataclasses import dataclass
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 800


@dataclass
class Chunk:
    page_number: int
    chunk_index: int
    content: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    chunks = []
    for page_number, raw_text in pages:
        text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
        if not text:
            continue
        tokens = count_tokens(text)
        if tokens <= MAX_TOKENS:
            chunks.append(Chunk(page_number, 0, text, tokens))
        else:
            sentences = split_sentences(text)
            current, current_tokens, idx = [], 0, 1
            for sentence in sentences:
                st = count_tokens(sentence)
                if current_tokens + st > MAX_TOKENS and current:
                    content = " ".join(current)
                    chunks.append(Chunk(page_number, idx, content, count_tokens(content)))
                    idx += 1
                    current = current[-1:]  # 1-sentence overlap
                    current_tokens = count_tokens(current[0])
                current.append(sentence)
                current_tokens += st
            if current:
                content = " ".join(current)
                chunks.append(Chunk(page_number, idx, content, count_tokens(content)))
    return chunks
```

**Test:**
```bash
uv run python3 -c "
from pathlib import Path
from src.ingestion.extractor import extract_pages
from src.ingestion.chunker import chunk_pages
pages = extract_pages(Path('camden_foi_random_pdfs/pdfs/01_CAM6551_asylum seeker accommodation complaints in private sector housing.pdf'))
chunks = chunk_pages(pages)
for c in chunks:
    print(f'Page {c.page_number}, chunk {c.chunk_index}: {c.token_count} tokens')
    print(c.content[:200])
"
```
```
Page 1, chunk 0: 396 tokens
Date: 20/12/2023 ...
Page 2, chunk 0: 212 tokens
Why not check our Portal...
```

---

### `src/ingestion/embedder.py`

Calls OpenAI's `text-embedding-3-small` in batches of 128. Returns 1536-dimensional vectors.

```python
from openai import OpenAI

client = OpenAI()
BATCH_SIZE = 128
MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=MODEL)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings
```

**Test** (requires `OPENAI_API_KEY` in env):
```bash
uv run python3 -c "
from dotenv import load_dotenv
load_dotenv()
from src.ingestion.embedder import embed_texts
vecs = embed_texts(['Camden Council FOI request about housing', 'temporary accommodation policy'])
print(f'{len(vecs)} vectors, dimension {len(vecs[0])}')
print(f'First value: {vecs[0][0]:.6f}')
"
```
```
2 vectors, dimension 1536
First value: 0.007484
```

Note: `load_dotenv()` must be called before importing modules that use API keys. The ingestion script and API handle this at their entry points so individual modules don't need to.

---

### `src/ingestion/pipeline.py`

Orchestrates the full ingestion flow with idempotency via SHA-256 file hashing. Safe to re-run — skips already-indexed files, upserts on conflict, wraps chunk inserts in a transaction.

```python
import hashlib
from pathlib import Path

import asyncpg

from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_texts
from src.ingestion.extractor import extract_pages
from src.ingestion.metadata import MetadataRow


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def ingest_file(
    pdf_path: Path,
    metadata: MetadataRow | None,
    pool: asyncpg.Pool,
) -> str:
    """Returns 'ingested', 'skipped', or 'failed'."""
    filename = pdf_path.name
    file_hash = hash_file(pdf_path)

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT file_hash, status FROM ingestion_log WHERE filename = $1",
            filename,
        )
        if existing and existing["file_hash"] == file_hash and existing["status"] == "success":
            return "skipped"

        await conn.execute(
            """INSERT INTO ingestion_log (filename, file_hash, status)
               VALUES ($1, $2, 'processing')
               ON CONFLICT (filename) DO UPDATE SET file_hash=$2, status='processing', error=NULL""",
            filename, file_hash,
        )

    try:
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        texts = [c.content for c in chunks]
        embeddings = embed_texts(texts)

        async with pool.acquire() as conn:
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """INSERT INTO documents (filename, foi_reference, date, title, response_text, total_pages)
                       VALUES ($1, $2, $3, $4, $5, $6)
                       ON CONFLICT (filename) DO UPDATE
                           SET foi_reference=$2, date=$3, title=$4, response_text=$5, total_pages=$6
                       RETURNING id""",
                    filename,
                    metadata.foi_reference if metadata else None,
                    metadata.date if metadata else None,
                    metadata.title if metadata else None,
                    metadata.response_text if metadata else None,
                    len(pages),
                )

                await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)

                await conn.executemany(
                    """INSERT INTO chunks (document_id, page_number, chunk_index, content, embedding, token_count)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    [
                        (doc_id, chunk.page_number, chunk.chunk_index,
                         chunk.content, str(embedding), chunk.token_count)
                        for chunk, embedding in zip(chunks, embeddings)
                    ],
                )

                await conn.execute(
                    """UPDATE ingestion_log SET status='success' WHERE filename=$1""",
                    filename,
                )

        return "ingested"

    except Exception as e:
        async with pool.acquire() as conn:
            await conn.execute(
                """UPDATE ingestion_log SET status='failed', error=$2 WHERE filename=$1""",
                filename, str(e),
            )
        raise
```

---

### `scripts/ingest_all.py`

CLI entry point. Loads env, connects to the database, runs `ingest_file` for every PDF in the corpus.

```python
import asyncio
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.metadata import load_metadata
from src.ingestion.pipeline import ingest_file


async def main(corpus_dir: Path) -> None:
    csv_path = corpus_dir / "downloaded_pdf_metadata.csv"
    pdfs_dir = corpus_dir / "pdfs"

    metadata = load_metadata(csv_path)

    import os
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])

    results = {"ingested": 0, "skipped": 0, "failed": 0}

    for pdf_path in sorted(pdfs_dir.glob("*.pdf")):
        meta = metadata.get(pdf_path.name)
        print(f"Processing {pdf_path.name}...", end=" ", flush=True)
        try:
            outcome = await ingest_file(pdf_path, meta, pool)
            results[outcome] += 1
            print(outcome)
        except Exception as e:
            results["failed"] += 1
            print(f"FAILED: {e}")

    await pool.close()
    print(f"\nDone. {results}")


if __name__ == "__main__":
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("camden_foi_random_pdfs")
    asyncio.run(main(corpus_dir))
```

**Run first ingestion:**
```bash
uv run python3 scripts/ingest_all.py
```
```
Processing 01_CAM6551_...pdf... ingested
...
Processing 40_CAM12105_...pdf... ingested

Done. {'ingested': 40, 'skipped': 0, 'failed': 0}
```

**Test — verify database contents:**
```bash
docker compose exec db psql -U foi -d foi -c "
SELECT COUNT(*) AS documents FROM documents;
SELECT COUNT(*) AS chunks FROM chunks;
SELECT AVG(token_count)::int AS avg_tokens, MIN(token_count) AS min_tokens, MAX(token_count) AS max_tokens FROM chunks;
"
```
```
 documents
-----------
        40

 chunks
--------
     97

 avg_tokens | min_tokens | max_tokens
------------+------------+------------
        350 |          6 |        721
```

**Test — idempotency (run again, all should be skipped):**
```bash
uv run python3 scripts/ingest_all.py
```
```
Done. {'ingested': 0, 'skipped': 40, 'failed': 0}
```

---

## Phase 3: Retrieval and generation

Also create `src/retrieval/__init__.py` (empty).

### `src/retrieval/search.py`

Cosine similarity search using pgvector's `<=>` operator. Supports optional date range filters.

```python
import os
import asyncpg
from dataclasses import dataclass
from datetime import date


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    foi_reference: str
    title: str
    date: date | None
    page_number: int
    chunk_index: int
    content: str
    score: float


async def vector_search(
    query_embedding: list[float],
    pool: asyncpg.Pool,
    top_k: int = 10,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[SearchResult]:
    embedding_str = str(query_embedding)

    filters = []
    params = [embedding_str, top_k]

    if date_from:
        params.append(date_from)
        filters.append(f"d.date >= ${len(params)}")
    if date_to:
        params.append(date_to)
        filters.append(f"d.date <= ${len(params)}")

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = f"""
        SELECT
            c.id AS chunk_id,
            d.id AS document_id,
            d.filename,
            d.foi_reference,
            d.title,
            d.date,
            c.page_number,
            c.chunk_index,
            c.content,
            1 - (c.embedding <=> $1::vector) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where_clause}
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return [
        SearchResult(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            filename=row["filename"],
            foi_reference=row["foi_reference"] or "",
            title=row["title"] or "",
            date=row["date"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            score=float(row["score"]),
        )
        for row in rows
    ]
```

**Test:**
```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv
load_dotenv()
import asyncpg
from src.ingestion.embedder import embed_texts
from src.retrieval.search import vector_search

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    embedding = embed_texts(['how many people are in temporary accommodation'])[0]
    results = await vector_search(embedding, pool, top_k=3)
    for r in results:
        print(f'{r.score:.3f} | {r.foi_reference} | {r.title[:50]}')
        print(f'  {r.content[:150]}')
    await pool.close()

asyncio.run(main())
"
```
```
0.560 | CAM6854 | the number of temporary accommodation properties,...
0.542 | CAM6854 | the number of temporary accommodation properties,...
0.516 | FOI11091 | Homelessness presentations, assessments and eligib...
```

---

### `src/retrieval/reranker.py`

Uses Claude Haiku to score and reorder results by relevance. Haiku is used here (not Sonnet) because reranking is a simple classification task — fast and cheap.

```python
import anthropic
import json

client = anthropic.Anthropic()


def rerank(query: str, results: list, top_k: int = 5) -> list:
    if not results:
        return results

    candidates = "\n\n".join(
        f"[{i}] {r.content[:400]}" for i, r in enumerate(results)
    )

    prompt = f"""You are ranking search results for relevance to a query about Freedom of Information requests to Camden Council.

Query: {query}

Candidates:
{candidates}

Return a JSON array of indices ordered from most to least relevant. Only include indices of results that are genuinely relevant. Example: [2, 0, 4]

Return only the JSON array, nothing else."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        indices = json.loads(message.content[0].text.strip())
        seen = set()
        reranked = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(results) and i not in seen:
                seen.add(i)
                reranked.append(results[i])
        return reranked[:top_k]
    except (json.JSONDecodeError, KeyError):
        return results[:top_k]
```

**Test:**
```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv
load_dotenv()
import asyncpg
from src.ingestion.embedder import embed_texts
from src.retrieval.search import vector_search
from src.retrieval.reranker import rerank

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    query = 'how many people are in temporary accommodation'
    embedding = embed_texts([query])[0]
    results = await vector_search(embedding, pool, top_k=10)
    print('Before rerank:')
    for r in results[:5]:
        print(f'  {r.score:.3f} | {r.foi_reference} | {r.content[:80]}')
    print()
    reranked = rerank(query, results, top_k=5)
    print('After rerank:')
    for r in reranked:
        print(f'  {r.foi_reference} | {r.content[:80]}')
    await pool.close()

asyncio.run(main())
"
```
```
Before rerank:
  0.560 | CAM6854 | 10.  Please can you advise us on the MAXIMUM...
  ...

After rerank:
  CAM6854 | In response to Q4 on 31 January 2023, there were 7,035 households waiting...
  ...
```

The reranker promoted the chunk directly answering the question ("7,035 households") from further down the vector results to position 1.

---

### `src/retrieval/generator.py`

Formats retrieved chunks as numbered sources, calls Claude Sonnet to generate an answer with inline `[SOURCE N]` citations.

```python
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()


@dataclass
class Citation:
    foi_reference: str
    title: str
    page_number: int
    chunk_id: str


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    prompt_sent: str


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[SOURCE {i+1}] FOI {r.foi_reference} — {r.title}\n"
            f"Page {r.page_number}\n"
            f"{r.content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are answering questions about Freedom of Information requests made to Camden Council.

Use only the sources provided. Cite sources using [SOURCE N] inline. If the sources do not contain enough information to answer, say so.

Question: {query}

Sources:
{context}

Answer with inline citations:"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    citations = [
        Citation(
            foi_reference=r.foi_reference,
            title=r.title,
            page_number=r.page_number,
            chunk_id=r.chunk_id,
        )
        for i, r in enumerate(results)
    ]

    return GeneratedAnswer(answer=answer, citations=citations, prompt_sent=prompt)
```

**Test — full pipeline end to end:**
```bash
uv run python3 -c "
import asyncio, os
from dotenv import load_dotenv
load_dotenv()
import asyncpg
from src.ingestion.embedder import embed_texts
from src.retrieval.search import vector_search
from src.retrieval.reranker import rerank
from src.retrieval.generator import generate_answer

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    query = 'how many households are in temporary accommodation in Camden?'
    embedding = embed_texts([query])[0]
    results = await vector_search(embedding, pool, top_k=10)
    reranked = rerank(query, results, top_k=5)
    answer = generate_answer(query, reranked)
    print(answer.answer)
    print()
    print('--- Citations ---')
    for c in answer.citations:
        print(f'  {c.foi_reference} p{c.page_number} — {c.title[:60]}')
    await pool.close()

asyncio.run(main())
"
```
```
Based on the available sources, the question about how many households are currently in
temporary accommodation in Camden cannot be directly answered. While FOI CAM6854 requested
this information, Camden Council withheld all of the information requested [SOURCE 2].

The only related figure available is that as of 31 January 2023, there were 7,035 households
waiting to be housed by the London Borough of Camden [SOURCE 1].

--- Citations ---
  CAM6854 p5 — the number of temporary accommodation properties, average le
  CAM6854 p1 — the number of temporary accommodation properties, average le
  CAM6854 p2 — the number of temporary accommodation properties, average le
```

---

## Phase 4: API

Also create `src/api/__init__.py` (empty).

### `src/api/main.py`

FastAPI app with lifespan-managed connection pool, `/health` and `/query` endpoints, and query logging.

```python
import json
import os
from contextlib import asynccontextmanager
from datetime import date

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from src.ingestion.embedder import embed_texts
from src.retrieval.generator import generate_answer
from src.retrieval.reranker import rerank
from src.retrieval.search import vector_search


pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    yield
    await pool.close()


app = FastAPI(title="Camden FOI RAG", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    date_from: date | None = None
    date_to: date | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    query_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    embedding = embed_texts([request.query])[0]
    results = await vector_search(
        embedding, pool, top_k=20,
        date_from=request.date_from,
        date_to=request.date_to,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    reranked = rerank(request.query, results, top_k=request.top_k)
    answer = generate_answer(request.query, reranked)

    async with pool.acquire() as conn:
        query_id = await conn.fetchval(
            """INSERT INTO query_logs
               (query, filters, retrieved_chunk_ids, prompt_sent, response, model)
               VALUES ($1, $2, $3, $4, $5, $6)
               RETURNING id""",
            request.query,
            json.dumps({"date_from": str(request.date_from), "date_to": str(request.date_to)}),
            [c.chunk_id for c in answer.citations],
            answer.prompt_sent,
            answer.answer,
            "claude-sonnet-4-6",
        )

    return QueryResponse(
        answer=answer.answer,
        citations=[
            {
                "foi_reference": c.foi_reference,
                "title": c.title,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
            }
            for c in answer.citations
        ],
        query_id=str(query_id),
    )
```

**Run the API:**
```bash
uv run uvicorn src.api.main:app --reload
```

**Test — health check:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

**Test — query endpoint:**
```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what enforcement action has Camden taken in the private rented sector?"}' \
  | python3 -m json.tool
```
```json
{
    "answer": "## Camden Council Enforcement Action in the Private Rented Sector\n\n### Civil Penalties Issued [SOURCE 1]:\n- 2018/2019: 128\n- 2019/2020: 122\n- 2020/2021: 85\n\n### Average Fine Per Case [SOURCE 1]:\n- 2018/2019: £3,225\n- 2019/2020: £4,784\n- 2020/2021: £4,422\n\n### Banning Orders [SOURCE 2]: 2 secured against landlords.",
    "citations": [
        {"foi_reference": "CAM1788", "title": "Enforcement in the private rented sector", "page_number": 1, "chunk_id": "..."},
        {"foi_reference": "CAM1788", "title": "Enforcement in the private rented sector", "page_number": 2, "chunk_id": "..."}
    ],
    "query_id": "3a884bf9-b858-4fb4-8f38-eb62d5bec84e"
}
```

**Test — verify query was logged:**
```bash
docker compose exec db psql -U foi -d foi -c "SELECT query, queried_at FROM query_logs ORDER BY queried_at DESC LIMIT 1;"
```
```
 query                                                                   | queried_at
-------------------------------------------------------------------------+-------------------------------
 what enforcement action has Camden taken in the private rented sector?  | 2026-03-13 17:52:56.872903+00
```

---

## Final file structure

```
foi-rag/
├── camden_foi_random_pdfs/
│   ├── downloaded_pdf_metadata.csv
│   └── pdfs/               # 40 FOI PDFs
├── docs/
│   ├── design.md
│   └── workflow.md         # this file
├── scripts/
│   └── ingest_all.py
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── schema.sql
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── extractor.py
│   │   ├── metadata.py
│   │   └── pipeline.py
│   └── retrieval/
│       ├── __init__.py
│       ├── generator.py
│       ├── reranker.py
│       └── search.py
├── .env                    # not committed
├── .env.example
├── docker-compose.yml
└── pyproject.toml
```

---

## What's next (stretch goals)

- **Dockerfile** — containerise the API so `docker compose up` starts everything
- **Hybrid search** — combine pgvector similarity with PostgreSQL full-text search (`tsvector`)
- **Query expansion** — use Claude to rewrite the question into multiple variants before retrieval
- **`/logs/{query_id}` endpoint** — retrieve the full prompt and chunks for any past query
- **Conversational retrieval** — maintain history so follow-up questions resolve references correctly
