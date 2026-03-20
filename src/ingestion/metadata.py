import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass
class MetadataRow:
    foi_reference: str
    date: date | None
    title: str
    response_text: str


def load_metadata(csv_path: Path) -> dict[str, MetadataRow]:
    result = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            filename = row["saved_filename"]
            raw_date = row.get("Document Date", "").strip()
            parsed_date = None
            if raw_date:
                try:
                    parsed_date = datetime.fromisoformat(raw_date).date()
                except ValueError:
                    pass
            result[filename] = MetadataRow(
                foi_reference=row.get("Identifier", "").strip(),
                date=parsed_date,
                title=row.get("Document Title", "").strip(),
                response_text=row.get("Document Text", "").strip(),
            )
    return result