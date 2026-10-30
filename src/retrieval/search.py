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


async def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    pool: asyncpg.Pool,
    top_k: int = 20,
    candidates_per_method: int = 50,
    rrf_k: int = 60,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion (RRF) of vector similarity + PostgreSQL full-text search.

    For each method, take top `candidates_per_method` results.
    Score(d) = sum over methods of 1 / (rrf_k + rank_in_method).
    Return top `top_k` by combined score.
    """
    embedding_str = str(query_embedding)

    filters = []
    params = [embedding_str, query_text, candidates_per_method, top_k, rrf_k]

    if date_from:
        params.append(date_from)
        filters.append(f"d.date >= ${len(params)}")
    if date_to:
        params.append(date_to)
        filters.append(f"d.date <= ${len(params)}")

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = f"""
        WITH vector_hits AS (
            SELECT c.id AS chunk_id,
                   ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector) AS rnk
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.embedding <=> $1::vector
            LIMIT $3
        ),
        text_hits AS (
            SELECT c.id AS chunk_id,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(c.content_tsv, plainto_tsquery('english', $2)) DESC
                   ) AS rnk
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where_clause}
                  {"AND" if where_clause else "WHERE"} c.content_tsv @@ plainto_tsquery('english', $2)
            LIMIT $3
        ),
        merged AS (
            SELECT chunk_id,
                   SUM(1.0 / ($5 + rnk)) AS rrf_score
            FROM (
                SELECT chunk_id, rnk FROM vector_hits
                UNION ALL
                SELECT chunk_id, rnk FROM text_hits
            ) u
            GROUP BY chunk_id
            ORDER BY rrf_score DESC
            LIMIT $4
        )
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
            m.rrf_score AS score
        FROM merged m
        JOIN chunks c ON c.id = m.chunk_id
        JOIN documents d ON d.id = c.document_id
        ORDER BY m.rrf_score DESC
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