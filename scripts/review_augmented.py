"""
Review Claude-generated augmented queries and promote the good ones
into the hand-labelled dataset.

For each (query, FOI) pair in augmented_queries.csv:
  - Shows the FOI's chunks with Claude's scores
  - Prompts: [a]ccept / [r]eject / [e]dit / [s]kip / [q]uit
  - Accepted rows get appended to hand_labelled_queries.csv
  - Rejected rows are logged so they don't come up again
  - Skip leaves the row for a later review session
"""

import asyncio
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

AUGMENTED_CSV = Path("eval_data/augmented_queries.csv")
HAND_CSV = Path("eval_data/hand_labelled_queries.csv")
REVIEW_LOG_CSV = Path("eval_data/augmented_review_log.csv")

HAND_FIELDS = ["query", "chunk_id", "relevance_score", "foi_reference", "labelled_at"]
REVIEW_LOG_FIELDS = ["query", "foi_reference", "decision", "reviewed_at"]


def load_augmented():
    if not AUGMENTED_CSV.exists():
        return []
    with open(AUGMENTED_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_reviewed_keys():
    if not REVIEW_LOG_CSV.exists():
        return set()
    with open(REVIEW_LOG_CSV, newline="") as f:
        return {(row["query"], row["foi_reference"]) for row in csv.DictReader(f)}


def log_review(query, foi_ref, decision):
    write_header = not REVIEW_LOG_CSV.exists()
    REVIEW_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "query": query,
            "foi_reference": foi_ref,
            "decision": decision,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        })


def append_to_hand(rows):
    write_header = not HAND_CSV.exists()
    HAND_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(HAND_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HAND_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


async def fetch_chunks_for_refs(conn, foi_refs):
    if not foi_refs:
        return {}
    rows = await conn.fetch(
        """SELECT d.foi_reference, d.title, c.id AS chunk_id,
                  c.page_number, c.chunk_index, c.token_count, c.content
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE d.foi_reference = ANY($1::text[])
           ORDER BY d.foi_reference, c.page_number, c.chunk_index""",
        list(foi_refs),
    )
    result = {}
    for row in rows:
        ref = row["foi_reference"]
        if ref not in result:
            result[ref] = {"title": row["title"] or "", "chunks": []}
        result[ref]["chunks"].append({
            "chunk_id": str(row["chunk_id"]),
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "token_count": row["token_count"],
            "content": row["content"],
        })
    return result


def show_review(foi_ref, foi_data, query, score_by_chunk):
    print(f"\n{'='*80}")
    print(f"FOI: {foi_ref} — {foi_data['title']}")
    print(f"Query: {query}")
    print(f"{'='*80}")

    rows = []
    for chunk in foi_data["chunks"]:
        score = score_by_chunk.get(chunk["chunk_id"], "-")
        rows.append([
            chunk["chunk_id"][:8],
            chunk["page_number"],
            chunk["chunk_index"],
            chunk["token_count"],
            score,
            chunk["content"][:200],
        ])
    print(tabulate(
        rows,
        headers=["id", "page", "cidx", "tokens", "score", "preview"],
        tablefmt="psql",
        colalign=("left", "right", "right", "right", "right", "left"),
    ))


def prompt_decision():
    while True:
        raw = input("\n[a]ccept / [r]eject / [e]dit / [s]kip / [q]uit: ").strip().lower()
        if raw in ("a", "r", "e", "s", "q"):
            return raw
        print("  invalid — try again")


def edit_scores(score_by_chunk, chunks):
    print("\nEdit scores. Press Enter to keep current score, or type 0/2/4.")
    new_scores = {}
    for i, chunk in enumerate(chunks, start=1):
        current = score_by_chunk.get(chunk["chunk_id"], 0)
        while True:
            raw = input(f"  [{i}] id={chunk['chunk_id'][:8]} p{chunk['page_number']} [{current}]: ").strip()
            if raw == "":
                new_scores[chunk["chunk_id"]] = int(current)
                break
            if raw in ("0", "2", "4"):
                new_scores[chunk["chunk_id"]] = int(raw)
                break
            print("    invalid — 0, 2, 4, or Enter")
    return new_scores


async def main():
    augmented = load_augmented()
    if not augmented:
        print(f"Nothing to review — {AUGMENTED_CSV} is empty or missing")
        return

    already_reviewed = load_reviewed_keys()

    groups = {}
    for row in augmented:
        key = (row["query"], row["foi_reference"])
        if key in already_reviewed:
            continue
        groups.setdefault(key, []).append(row)

    total = len(augmented)
    reviewed_count = len([r for r in augmented if (r["query"], r["foi_reference"]) in already_reviewed])
    print(f"{total} augmented rows total")
    print(f"{reviewed_count} already reviewed")
    print(f"{len(groups)} (query, FOI) pairs to review\n")

    if not groups:
        print("Nothing left to review.")
        return

    foi_refs = {foi for _, foi in groups.keys()}
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        chunks_by_ref = await fetch_chunks_for_refs(conn, foi_refs)
    finally:
        await conn.close()

    accepted = 0
    rejected = 0
    skipped = 0

    for i, ((query, foi_ref), rows) in enumerate(groups.items(), start=1):
        print(f"\n### {i}/{len(groups)} ###")
        foi_data = chunks_by_ref.get(foi_ref)
        if not foi_data:
            print(f"  FOI {foi_ref} not in DB — skipping")
            continue

        score_by_chunk = {r["chunk_id"]: int(r["relevance_score"]) for r in rows}
        show_review(foi_ref, foi_data, query, score_by_chunk)

        choice = prompt_decision()
        if choice == "q":
            print("\nQuitting.")
            break
        if choice == "s":
            skipped += 1
            continue
        if choice == "r":
            log_review(query, foi_ref, "rejected")
            rejected += 1
            continue

        final_scores = score_by_chunk
        if choice == "e":
            final_scores = edit_scores(score_by_chunk, foi_data["chunks"])

        timestamp = datetime.now(timezone.utc).isoformat()
        to_save = [
            {
                "query": query,
                "chunk_id": chunk_id,
                "relevance_score": score,
                "foi_reference": foi_ref,
                "labelled_at": timestamp,
            }
            for chunk_id, score in final_scores.items()
        ]
        append_to_hand(to_save)
        log_review(query, foi_ref, "accepted")
        accepted += 1
        print(f"  → promoted {len(to_save)} rows to {HAND_CSV}")

    print(f"\nDone. accepted={accepted}, rejected={rejected}, skipped={skipped}")


if __name__ == "__main__":
    asyncio.run(main())
