import asyncio
import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from tabulate import tabulate

OUTPUT_CSV = Path("eval_data/hand_labelled_queries.csv")
FIELDNAMES = ["query", "chunk_id", "relevance_score", "foi_reference", "labelled_at"]


def load_labelled_refs(csv_path: Path) -> set:
    """Return the set of foi_references already saved in the CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["foi_reference"] for row in reader}


def append_labels(csv_path: Path, rows: list) -> None:
    """Append scored rows to the CSV, writing the header if the file is new."""
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def show_foi(foi_reference: str, title: str, chunks: list[dict]) -> None:
    """
    Print the chunks for a single FOI document to stdout.

    Output format:
      ═══ CAM6551 — Asylum seeker accommodation ═══
      [1] chunk_id=e530fd81  page=1  chunk_index=0
          Date: 20/12/2023 Ref: CAM6551 Dear Requester...

    Args:
        foi_reference: e.g. "CAM6551"
        title: document title from the documents table
        chunks: list of dicts with keys chunk_id, page_number, chunk_index, content
    """
    print(f"\n═══ {foi_reference} — {title} ═══\n")
    rows = [
        [
            chunk["chunk_id"][:8],
            chunk["page_number"],
            chunk["chunk_index"],
            chunk["token_count"],
            chunk["content"][:200],
        ]
        for chunk in chunks
    ]
    print(tabulate(
        rows,
        headers=["id", "page_number", "chunk_index", "token_count", "preview"],
        tablefmt="psql",
        colalign=("left", "right", "right", "right", "left"),
    ))
    print()


async def fetch_fois(conn):
    """
    Returns a list of dicts, one per distinct document, ordered by filename prefix.

    [{"foi_reference": "CAM6551", "title": "...", "filename": "...", "chunks": [...]}, ...]
    """
    rows = await conn.fetch("""
        SELECT d.foi_reference, d.title, d.filename,
            c.id AS chunk_id, c.page_number, c.chunk_index, c.token_count, c.content
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY d.foi_reference, c.page_number, c.chunk_index
    """)

    fois = {}
    for row in rows:
        ref = row["foi_reference"]
        if ref not in fois:
            fois[ref] = {"foi_reference": ref, "title": row["title"], "filename": row["filename"], "chunks": []}
        fois[ref]["chunks"].append({
            "chunk_id": str(row["chunk_id"]),
            "page_number": row["page_number"],
            "chunk_index": row["chunk_index"],
            "token_count": row["token_count"],
            "content": row["content"],
        })

    def _file_order(foi):
        m = re.match(r'^(\d+)_', foi.get("filename", ""))
        return int(m.group(1)) if m else 999

    return sorted(fois.values(), key=_file_order)


async def label_session(conn):
    fois = await fetch_fois(conn)
    already_done = load_labelled_refs(OUTPUT_CSV)

    skipped = [f["foi_reference"] for f in fois if f["foi_reference"] in already_done]
    if skipped:
        print(f"Resuming — skipping {len(skipped)} already labelled: {', '.join(skipped)}\n")

    for foi in fois:
        if foi["foi_reference"] in already_done:
            continue

        show_foi(foi["foi_reference"], foi["title"], foi["chunks"])

        query = input("Enter your question (or 'skip' / 'quit'): ").strip()
        if query == "quit":
            break
        if query == "skip":
            continue

        scored_rows = []
        for i, chunk in enumerate(foi["chunks"], start=1):
            raw = input(f"  Score chunk [{i}] (0/2/4, s=skip): ").strip()
            if raw == "s":
                continue
            if raw not in ("0", "2", "4"):
                print("  Invalid — skipping chunk.")
                continue
            scored_rows.append({
                "query": query,
                "chunk_id": chunk["chunk_id"],
                "relevance_score": int(raw),
                "foi_reference": foi["foi_reference"],
                "labelled_at": datetime.now(timezone.utc).isoformat(),
            })

        if scored_rows:
            append_labels(OUTPUT_CSV, scored_rows)
            print(f"  Saved {len(scored_rows)} row(s) to {OUTPUT_CSV}\n")


async def main():
    load_dotenv()
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await label_session(conn)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
