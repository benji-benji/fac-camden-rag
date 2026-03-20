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