"""
Bootstrap an augmented eval dataset using Claude.

Uses the user's hand-labelled queries as few-shot examples to calibrate style,
then generates queries + chunk relevance scores for FOIs that haven't been
hand-labelled yet. Writes to eval_data/augmented_queries.csv.

The user can then review this with the review pipeline (step 3) and promote
accepted rows into hand_labelled_queries.csv.
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

HAND_CSV = Path("eval_data/hand_labelled_queries.csv")
OUTPUT_CSV = Path("eval_data/augmented_queries.csv")
FIELDNAMES = ["query", "chunk_id", "relevance_score", "foi_reference", "labelled_at"]

DEFAULT_MODEL = "claude-sonnet-4-6"
QUERIES_PER_FOI = 2
SEED_EXAMPLES = 6


def load_seed_examples(csv_path: Path, n: int) -> list[dict]:
    """Group hand-labelled rows by query, pick n random queries as few-shot."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Hand-labelled CSV not found: {csv_path}")
    by_query = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_query.setdefault(row["query"], []).append(row)
    queries = list(by_query.keys())
    random.shuffle(queries)
    sampled = queries[:n]
    return [
        {
            "query": q,
            "foi_reference": by_query[q][0]["foi_reference"],
            "scores": [(r["chunk_id"], int(r["relevance_score"])) for r in by_query[q]],
        }
        for q in sampled
    ]


def load_done_refs(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {row["foi_reference"] for row in csv.DictReader(f)}


async def fetch_unlabelled_fois(conn, skip_refs: set) -> list[dict]:
    rows = await conn.fetch("""
        SELECT d.foi_reference, d.title, d.filename,
               c.id AS chunk_id, c.page_number, c.chunk_index, c.content
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.foi_reference IS NOT NULL AND d.foi_reference != ''
        ORDER BY d.foi_reference, c.page_number, c.chunk_index
    """)
    fois = {}
    for row in rows:
        ref = row["foi_reference"]
        if ref in skip_refs:
            continue
        if ref not in fois:
            fois[ref] = {
                "foi_reference": ref,
                "title": row["title"] or "",
                "filename": row["filename"] or "",
                "chunks": [],
            }
        fois[ref]["chunks"].append({
            "chunk_id": str(row["chunk_id"]),
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
        })
    return list(fois.values())


def build_prompt(foi: dict, seed_examples: list[dict]) -> str:
    example_block = ""
    for ex in seed_examples:
        scores_fmt = ", ".join(f"{cid[:8]}={s}" for cid, s in ex["scores"])
        example_block += (
            f"\n  FOI: {ex['foi_reference']}"
            f"\n  Query: {ex['query']}"
            f"\n  Scores: {scores_fmt}\n"
        )

    chunks_block = ""
    for chunk in foi["chunks"]:
        content = chunk["content"][:1000]
        chunks_block += (
            f"\n  chunk_id: {chunk['chunk_id']}"
            f"\n  page: {chunk['page_number']}  chunk_index: {chunk['chunk_index']}"
            f"\n  content: {content}\n"
        )

    return f"""You are building an evaluation dataset for a Camden Council FOI (Freedom of Information) search system.

TASK
For the FOI document below, write {QUERIES_PER_FOI} realistic questions a member of the public might ask about this topic, then for each question score every chunk 0 (irrelevant), 2 (partially relevant), or 4 (directly relevant).

RULES
1. Write the question based on the document's topic FIRST, then score chunks. Do NOT write questions that paraphrase specific chunk wording — they must sound like what a real user would type.
2. Questions should be natural, like casual typed questions. Occasional typos or imprecise wording are fine (that's realistic).
3. Cover a mix: factual questions (counts, amounts, names, dates) AND process/policy questions (why, how, does the council...).
4. A score-4 chunk contains the direct answer. A score-2 chunk is related context but doesn't directly answer. A score-0 chunk is unrelated.
5. Every chunk must have a score for every query.

HUMAN-LABELLED EXAMPLES (for style calibration)
{example_block}

FOI TO LABEL
FOI reference: {foi["foi_reference"]}
Title: {foi["title"]}
Chunks:
{chunks_block}

OUTPUT FORMAT
Respond with valid JSON only, no markdown, no prose:
{{
  "queries": [
    {{"query": "...", "scores": {{"<chunk_id>": 4, "<chunk_id>": 0}}}},
    {{"query": "...", "scores": {{"<chunk_id>": 2, "<chunk_id>": 4}}}}
  ]
}}
"""


def strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def generate_for_foi(client: Anthropic, foi: dict, seed: list[dict], model: str) -> dict | None:
    prompt = build_prompt(foi, seed)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = strip_code_fences(response.content[0].text)
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"    API error on attempt {attempt+1}: {e}")
            continue
    return None


def save_rows(csv_path: Path, rows: list[dict]) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process N FOIs (for testing)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for example selection")
    args = parser.parse_args()

    random.seed(args.seed)

    seed_examples = load_seed_examples(HAND_CSV, SEED_EXAMPLES)
    print(f"Loaded {len(seed_examples)} seed examples from {HAND_CSV}")

    skip = load_done_refs(OUTPUT_CSV) | load_done_refs(HAND_CSV)
    print(f"Skipping {len(skip)} already-labelled FOIs")

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        fois = await fetch_unlabelled_fois(conn, skip)
    finally:
        await conn.close()

    if args.limit:
        fois = fois[:args.limit]
    print(f"{len(fois)} FOIs to process with {args.model}\n")

    client = Anthropic()
    ok_count = 0
    fail_count = 0

    for i, foi in enumerate(fois, start=1):
        print(f"[{i}/{len(fois)}] {foi['foi_reference']} — {foi['title'][:60]}")
        result = generate_for_foi(client, foi, seed_examples, args.model)
        if not result or "queries" not in result:
            fail_count += 1
            print("  FAILED — no valid JSON after retries")
            continue

        valid_chunk_ids = {c["chunk_id"] for c in foi["chunks"]}
        rows_to_save = []
        for q in result["queries"]:
            query = (q.get("query") or "").strip()
            scores = q.get("scores") or {}
            if not query or not isinstance(scores, dict):
                continue
            for chunk_id, score in scores.items():
                if chunk_id not in valid_chunk_ids:
                    continue
                try:
                    score_int = int(score)
                except (TypeError, ValueError):
                    continue
                if score_int not in (0, 2, 4):
                    continue
                rows_to_save.append({
                    "query": query,
                    "chunk_id": chunk_id,
                    "relevance_score": score_int,
                    "foi_reference": foi["foi_reference"],
                    "labelled_at": datetime.now(timezone.utc).isoformat(),
                })

        if rows_to_save:
            save_rows(OUTPUT_CSV, rows_to_save)
            ok_count += 1
            print(f"  saved {len(rows_to_save)} rows")
        else:
            fail_count += 1
            print("  no valid rows produced")

    print(f"\nDone. ok={ok_count}, failed={fail_count}, output={OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
