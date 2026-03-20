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