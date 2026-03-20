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