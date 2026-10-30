import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import tiktoken
from src.ingestion.extractor import extract_pages
from src.ingestion.embedder import embed_texts

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, paper_id, chunk_size=256, overlap_ratio=0.2):
    """Split text into token-based chunks with overlap."""
    overlap = int(chunk_size * overlap_ratio)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        chunks.append({
            "chunk_id": f"{paper_id}_{chunk_size}_{idx}",
            "paper_id": paper_id,
            "text": enc.decode(chunk_tokens),
        })

        idx += 1
        start += chunk_size - overlap
        if end >= len(tokens):
            break

    return chunks

def build_corpus(pdf_dir, chunk_size):
    all_chunks = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        # derive foi_reference from filename "04_FOI11091_title.pdf"
        foi_ref = pdf_path.stem.split("_")[1]
        pages = extract_pages(pdf_path)
        text = "\n".join(t for _, t in pages)
        chunks = chunk_text(text, foi_ref, chunk_size)
        all_chunks.extend(chunks)
    return all_chunks


def main():
    chunks = build_corpus("camden_foi_random_pdfs/pdfs", chunk_size=256)
    print(f"Built {len(chunks)} chunks at size 256")

    texts = [c["text"] for c in chunks]
    embeddings = np.array(embed_texts(texts))
    print(f"Embedded {embeddings.shape[0]} chunks, dim={embeddings.shape[1]}")

    # quick sanity check — embed a query and find the top match
    query = "How many asylum seeker accommodation complaints?"
    query_vec = np.array(embed_texts([query])[0])
    scores = embeddings @ query_vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))
    top_idx = np.argsort(scores)[::-1][:5]
    print(f"\nTop 5 chunks for: {query!r}")
    for i in top_idx:
        print(f"  {scores[i]:.3f}  {chunks[i]['paper_id']}  {chunks[i]['text'][:80]}...")


if __name__ == "__main__":
    main()
