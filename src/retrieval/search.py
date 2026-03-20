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