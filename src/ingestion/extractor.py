from pathlib import Path
import fitz  # PyMuPDF


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append((page.number + 1, text))  # 1-indexed
    return pages