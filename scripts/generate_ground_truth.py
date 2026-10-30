import argparse
import asyncio
import csv
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import anthropic
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OUTPUT_PATH = Path("eval_data/synthetic_chunk_query_pairs.csv")
#DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class ChunkRecord:
    chunk_id: str
    filename: str
    foi_reference: str
    title: str
    page_number: int
    chunk_index: int
    content: str


async def fetch_chunks(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT c.id AS chunk_id, d.filename, COALESCE(d.foi_reference, '') AS foi_reference,
                      COALESCE(d.title, '') AS title, c.page_number, c.chunk_index, c.content
               FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY d.filename, c.page_number, c.chunk_index""",
        )
    return [ChunkRecord(**dict(row)) for row in rows]


def generate_queries_for_chunk(client, chunk, model):
    prompt = """You are an expert at generating FOI queries based on document content.
Given the following chunk of a Camden document, generate 3 FOI queries that a user might ask:
1. A very relevant query (score 4) that is directly answered by this chunk.
2. A medium relevant query (score 2) that is related but broader or only partially answered by this chunk.
3. A low relevant query (score 0) that is a plausible Camden FOI question but not answered by this chunk."""
    prompt += f"\n\nChunk information:\nFOI Reference: {chunk.foi_reference}\nTitle: {chunk.title}\nPage Number: {chunk.page_number}\nContent: {chunk.content}\n\n"
    prompt += 'Return strict JSON only, no markdown:\n{\n  "very_relevant_query": "...",\n  "medium_relevant_query": "...",\n  "low_relevant_query": "..."\n}'

    for _ in range(3):
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    # fallback if all attempts fail
    return {
        "very_relevant_query": f"What does FOI {chunk.foi_reference} say about {chunk.title}?",
        "medium_relevant_query": f"How does Camden handle issues related to {chunk.title}?",
        "low_relevant_query": "How many parking fines were issued in Camden in 2024?",
    }


def write_dataset(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "query_id",
        "query_text",
        "relevance_score",
        "relevance_label",
        "chunk_id",
        "filename",
        "foi_reference",
        "title",
        "page_number",
        "chunk_index",
        "chunk_preview",
        "generation_method",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def main(args):
    """
    Parse CLI args: --limit (int, optional) and --output (Path, default DEFAULT_OUTPUT_PATH)
    Create an anthropic.Anthropic() client
    Create an asyncpg pool using os.environ["DATABASE_URL"]
    Call fetch_chunks(pool), apply --limit if set
    Loop through chunks — for each one call generate_queries_for_chunk(), then build 3 rows
    (one per query type) and append to a dataset_rows list
    Call write_dataset(dataset_rows, args.output)
    Print a summary: how many chunks processed, how many rows written
    Each row needs all the CSV columns — use str(uuid.uuid4()) for pair_id and query_id,
    chunk.content[:160] for chunk_preview.
    """
    client = anthropic.Anthropic()
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])

    chunks = await fetch_chunks(pool)
    await pool.close()

    if args.limit:
        chunks = chunks[:args.limit]

    dataset_rows = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[{idx}/{len(chunks)}] {chunk.filename} p{chunk.page_number}")
        queries = generate_queries_for_chunk(client, chunk, DEFAULT_MODEL)
        
        for key, score, label in [
            ("very_relevant_query", 4, "very_relevant"),
            ("medium_relevant_query", 2, "medium_relevant"),
            ("low_relevant_query", 0, "low_relevant"),
        ]:
            dataset_rows.append({
                "pair_id": str(uuid.uuid4()),
                "query_id": str(uuid.uuid4()),
                "query_text": queries[key],
                "relevance_score": score,
                "relevance_label": label,
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "foi_reference": chunk.foi_reference,
                "title": chunk.title,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "chunk_preview": chunk.content[:160],
                "generation_method": "claude",
            })

    write_dataset(dataset_rows, args.output)
    print(f"Chunks processed: {len(chunks)}")
    print(f"Rows written: {len(dataset_rows)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic eval dataset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    asyncio.run(main(args))
