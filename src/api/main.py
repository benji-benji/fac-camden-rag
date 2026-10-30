import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from src.ingestion.embedder import embed_texts
from src.retrieval.generator import MODEL as GENERATOR_MODEL, generate_answer
from src.retrieval.reranker import MODEL as RERANKER_MODEL, rerank
from src.retrieval.search import vector_search

RETRIEVAL_TOP_K = 50  # candidates passed to the reranker

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


def _preview(text: str, length: int = 120) -> str:
    compact = " ".join(text.split())
    return compact[:length]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    embedding = embed_texts([request.query])[0]
    results = await vector_search(
        embedding,
        pool,
        top_k=RETRIEVAL_TOP_K,
        date_from=request.date_from,
        date_to=request.date_to,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # retrieval trace
    retrieval_trace = []
    for idx, r in enumerate(results):
        retrieval_trace.append(
            {
                "rank": idx + 1,
                "chunk_id": r.chunk_id,
                "foi_reference": r.foi_reference,
                "page_number": r.page_number,
                "score": r.score,
                "content_preview": _preview(r.content),
            }
        )

    reranked = rerank(request.query, results, top_k=request.top_k)

    rerank_trace = []
    for idx, r in enumerate(reranked):
        rerank_trace.append(
            {
                "rank": idx + 1,
                "chunk_id": r.chunk_id,
                "foi_reference": r.foi_reference,
                "page_number": r.page_number,
                "content_preview": _preview(r.content),
            }
        )

    answer = generate_answer(request.query, reranked)

    # assemble full trace_payload dictionary to add to json dump
    trace_payload = {
        "date_from": request.date_from.isoformat() if request.date_from else None,
        "date_to": request.date_to.isoformat() if request.date_to else None,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "answer_top_k": request.top_k,
        "query_embedding": embedding,
        "retrieval_candidates": retrieval_trace,
        "reranker_model": RERANKER_MODEL,
        "generator_model": GENERATOR_MODEL,
        "rerank_input_count": len(results),
        "rerank_output_count": len(reranked),
        "reranked_chunk_ids": [r.chunk_id for r in reranked],
        "reranked_candidates": rerank_trace,
    }

    async with pool.acquire() as conn:
        query_id = await conn.fetchval(
            """INSERT INTO query_logs
               (query, filters, retrieved_chunk_ids, prompt_sent, response, model)
               VALUES ($1, $2, $3, $4, $5, $6)
               RETURNING id""",
            request.query,
            json.dumps(trace_payload),
            [c.chunk_id for c in answer.citations],
            answer.prompt_sent,
            answer.answer,
            GENERATOR_MODEL,
        )

    return QueryResponse(
        answer=answer.answer,
        citations=[
            {
                "source_number": c.source_number,
                "foi_reference": c.foi_reference,
                "title": c.title,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
            }
            for c in answer.citations
        ],
        query_id=str(query_id),
    )


"""
# GET /logs/{query_id}
Take a query_id as a path parameter (type it as uuid.UUID — add import uuid at the top)
Fetch the matching row from query_logs
Parse filters from JSON string to dict
Return the full row as a dict
Raise a 404 if not found

"""


@app.get("/logs/{query_id}")
async def get_log(query_id: uuid.UUID):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, query, filters, retrieved_chunk_ids, prompt_sent, response, model, queried_at
               FROM query_logs
               WHERE id = $1""",
            query_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Log entry not found")

    return {
        "id": str(row["id"]),
        "query": row["query"],
        "filters": json.loads(row["filters"]) if isinstance(row["filters"], str) else row["filters"],
        "retrieved_chunk_ids": row["retrieved_chunk_ids"],
        "prompt_sent": row["prompt_sent"],
        "response": row["response"],
        "model": row["model"],
        "queried_at": row["queried_at"].isoformat(),
    }

'''
GET /documents 

queries the documents table and returns a list.

SELECT id, filename, foi_reference, date, title, total_pages, indexed_at
FROM documents
ORDER BY indexed_at DESC
For each row, return a dict. date and indexed_at are date/datetime objects — convert them with .isoformat() (check for None on date since it's nullable).


'''

@app.get("/stats")
async def stats():
    async with pool.acquire() as conn:
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        last_ingested = await conn.fetchval("SELECT MAX(indexed_at) FROM documents")
        avg_tokens = await conn.fetchval("SELECT AVG(token_count) FROM chunks")
    return {
        "document_count": doc_count,
        "chunk_count": chunk_count,
        "last_ingested_at": last_ingested.isoformat() if last_ingested else None,
        "avg_chunk_tokens": round(float(avg_tokens), 1) if avg_tokens else None,
    }


@app.get("/documents")
async def list_documents():
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, filename, foi_reference, date, title, total_pages, indexed_at
               FROM documents
               ORDER BY indexed_at DESC"""
        )
    documents = []
    for row in rows:
        documents.append(
            {
                "id": str(row["id"]),
                "filename": row["filename"],
                "foi_reference": row["foi_reference"],
                "date": row["date"].isoformat() if row["date"] else None,
                "title": row["title"],
                "total_pages": row["total_pages"],
                "indexed_at": row["indexed_at"].isoformat(),
            }
        )
    return documents

