"""
Run all three Izaak harness experiments against the original 40-FOI corpus.

Experiment 1: vary k
Experiment 2: vary chunk size
Experiment 3: vary embedding model (OpenAI vs sentence-transformers)

Evaluation uses doc-level matching (foi_reference) because chunk IDs differ
across chunk sizes.
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import tiktoken
from openai import OpenAI

from src.ingestion.extractor import extract_pages

PDF_DIR = "camden_foi_random_pdfs/pdfs"
EVAL_CSV = "eval_data/hand_labelled_queries.csv"
ENC = tiktoken.get_encoding("cl100k_base")
OPENAI_CLIENT = OpenAI()


# ---------- chunking ----------

def chunk_text(text, foi_ref, chunk_size, overlap_ratio=0.2):
    overlap = int(chunk_size * overlap_ratio)
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append({
            "chunk_id": f"{foi_ref}_{chunk_size}_{idx}",
            "foi_reference": foi_ref,
            "text": ENC.decode(tokens[start:end]),
        })
        idx += 1
        start += chunk_size - overlap
        if end >= len(tokens):
            break
    return chunks


def build_corpus(pdf_dir, chunk_size):
    all_chunks = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        foi_ref = pdf_path.stem.split("_")[1]
        pages = extract_pages(pdf_path)
        text = "\n".join(t for _, t in pages)
        all_chunks.extend(chunk_text(text, foi_ref, chunk_size))
    return all_chunks


# ---------- embedding backends ----------

def embed_openai(texts, model="text-embedding-3-small", batch_size=128):
    out = []
    for i in range(0, len(texts), batch_size):
        resp = OPENAI_CLIENT.embeddings.create(input=texts[i:i+batch_size], model=model)
        out.extend([d.embedding for d in resp.data])
    arr = np.array(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


_ST_MODEL_CACHE = {}

def embed_sentence_transformers(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    if model_name not in _ST_MODEL_CACHE:
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name].encode(texts, normalize_embeddings=True)


# ---------- retrieval ----------

def make_retrieve(embed_fn, chunks, embeddings):
    def retrieve(query, k=5):
        qv = embed_fn([query])[0]
        scores = embeddings @ qv
        top = np.argsort(scores)[::-1][:k]
        # return foi_references (doc-level eval)
        return [chunks[i]["foi_reference"] for i in top]
    return retrieve


# ---------- eval ----------

def load_eval_set(csv_path, threshold=2):
    grouped = defaultdict(set)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["relevance_score"]) >= threshold:
                grouped[row["query"]].add(row["foi_reference"])
    return [{"query": q, "relevant": list(refs)} for q, refs in grouped.items()]


def precision_at_k(retrieved, relevant):
    r, g = set(retrieved), set(relevant)
    return len(r & g) / len(r) if r else 0.0


def recall_at_k(retrieved, relevant):
    r, g = set(retrieved), set(relevant)
    return len(r & g) / len(g) if g else 0.0


def reciprocal_rank(retrieved, relevant):
    g = set(relevant)
    for i, rid in enumerate(retrieved):
        if rid in g:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(eval_set, retrieve_fn, k=5):
    rows = [
        (
            precision_at_k(r := retrieve_fn(q["query"], k), q["relevant"]),
            recall_at_k(r, q["relevant"]),
            reciprocal_rank(r, q["relevant"]),
        )
        for q in eval_set
    ]
    return {
        "avg_precision": float(np.mean([x[0] for x in rows])),
        "avg_recall": float(np.mean([x[1] for x in rows])),
        "mrr": float(np.mean([x[2] for x in rows])),
        "k": k,
    }


# ---------- experiments ----------

def experiment_1_vary_k(eval_set, chunks, embeddings):
    print("\n=== Experiment 1: vary k (chunk_size=256, OpenAI small) ===")
    print(f"\n  {'k':>4} | {'P@k':>8} | {'R@k':>8} | {'MRR':>6}")
    print(f"  {'-'*36}")
    retrieve = make_retrieve(embed_openai, chunks, embeddings)
    for k in [1, 3, 5, 10, 20]:
        r = evaluate(eval_set, retrieve, k=k)
        print(f"  {k:>4} | {r['avg_precision']:>8.3f} | {r['avg_recall']:>8.3f} | {r['mrr']:>6.3f}")


def experiment_2_vary_chunk_size(eval_set):
    print("\n=== Experiment 2: vary chunk size (k=5, OpenAI small) ===")
    print(f"\n  {'size':>5} | {'chunks':>7} | {'P@5':>8} | {'R@5':>8} | {'MRR':>6}")
    print(f"  {'-'*48}")
    for size in [128, 256, 512, 1024]:
        chunks = build_corpus(PDF_DIR, size)
        texts = [c["text"] for c in chunks]
        emb = embed_openai(texts)
        retrieve = make_retrieve(embed_openai, chunks, emb)
        r = evaluate(eval_set, retrieve, k=5)
        print(f"  {size:>5} | {len(chunks):>7} | {r['avg_precision']:>8.3f} | {r['avg_recall']:>8.3f} | {r['mrr']:>6.3f}")


def experiment_3_vary_model(eval_set, chunks):
    print("\n=== Experiment 3: vary embedding model (chunk_size=256, k=5) ===")
    print(f"\n  {'model':<28} | {'dim':>5} | {'P@5':>8} | {'R@5':>8} | {'MRR':>6}")
    print(f"  {'-'*64}")
    texts = [c["text"] for c in chunks]

    openai_emb = embed_openai(texts)
    r = evaluate(eval_set, make_retrieve(embed_openai, chunks, openai_emb), k=5)
    print(f"  {'text-embedding-3-small':<28} | {openai_emb.shape[1]:>5} | {r['avg_precision']:>8.3f} | {r['avg_recall']:>8.3f} | {r['mrr']:>6.3f}")

    st_emb = embed_sentence_transformers(texts)
    def st_embed_one(q):
        return embed_sentence_transformers(q)
    r = evaluate(eval_set, make_retrieve(st_embed_one, chunks, st_emb), k=5)
    print(f"  {'all-MiniLM-L6-v2':<28} | {st_emb.shape[1]:>5} | {r['avg_precision']:>8.3f} | {r['avg_recall']:>8.3f} | {r['mrr']:>6.3f}")


# ---------- main ----------

def main():
    eval_set = load_eval_set(EVAL_CSV, threshold=2)
    print(f"Loaded {len(eval_set)} eval queries")

    chunks_256 = build_corpus(PDF_DIR, chunk_size=256)
    emb_256 = embed_openai([c["text"] for c in chunks_256])
    print(f"Built {len(chunks_256)} chunks at size 256, dim={emb_256.shape[1]}")

    experiment_1_vary_k(eval_set, chunks_256, emb_256)
    experiment_2_vary_chunk_size(eval_set)
    experiment_3_vary_model(eval_set, chunks_256)


if __name__ == "__main__":
    main()
