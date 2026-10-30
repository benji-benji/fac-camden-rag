"""
Run batch retrieval eval against the currently-running API.

Hits /query + /logs for each unique query in the hand-labelled CSV,
computes aggregate metrics, and prints a compact report suitable for
pasting into lab_notes.md.

Usage:
    uv run python3 scripts/run_eval.py
    uv run python3 scripts/run_eval.py --limit 5
    uv run python3 scripts/run_eval.py --dataset eval_data/hand_labelled_queries.csv
"""

import argparse
import sys
from pathlib import Path
from statistics import mean

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.metrics import evaluate_query_ranking, load_golden_dataset

DEFAULT_DATASET = "eval_data/hand_labelled_queries.csv"
DEFAULT_API = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()

    golden_df = load_golden_dataset(args.dataset)
    queries = sorted(golden_df["query_text"].unique().tolist())
    if args.limit:
        queries = queries[: args.limit]

    print(f"Running batch eval: {len(queries)} queries against {args.api}")
    print(f"Dataset: {args.dataset}, threshold={args.threshold}, top_k={args.top_k}\n")

    series = {k: [] for k in ["map", "mar", "ndcg@5", "p@5", "r@5", "f1@5", "p@1", "r@1"]}
    per_query = []

    for i, q in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {q[:70]}", flush=True)
        try:
            qr = requests.post(
                f"{args.api}/query", json={"query": q, "top_k": args.top_k}, timeout=180
            )
            qr.raise_for_status()
            query_id = qr.json()["query_id"]
            log = requests.get(f"{args.api}/logs/{query_id}", timeout=60).json()
            reranked = log.get("filters", {}).get("reranked_candidates", [])
            predicted_ids = [r["chunk_id"] for r in reranked]
            if not predicted_ids:
                predicted_ids = [str(c) for c in (log.get("retrieved_chunk_ids") or [])]

            res = evaluate_query_ranking(
                query_text=q,
                predicted_chunk_ids=predicted_ids,
                golden_df=golden_df,
                ks=[1, 3, 5, 10],
                relevant_threshold=args.threshold,
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        if res.matched_gold_rows == 0:
            print("    no gold rows matched — skipped")
            continue

        series["map"].append(res.average_precision)
        series["mar"].append(res.mar)
        series["ndcg@5"].append(res.ndcg_at_k.get(5, 0.0))
        series["p@1"].append(res.precision_at_k.get(1, 0.0))
        series["r@1"].append(res.recall_at_k.get(1, 0.0))
        series["p@5"].append(res.precision_at_k.get(5, 0.0))
        series["r@5"].append(res.recall_at_k.get(5, 0.0))
        series["f1@5"].append(res.f1_at_k.get(5, 0.0))
        per_query.append((q[:60], res))

    if not series["map"]:
        print("\nNo results — nothing matched the golden dataset.")
        return

    print(f"\n{'='*70}")
    print(f"Aggregate metrics over {len(series['map'])} queries")
    print(f"{'='*70}")
    print(f"  MAP     {mean(series['map']):.3f}")
    print(f"  MAR     {mean(series['mar']):.3f}")
    print(f"  NDCG@5  {mean(series['ndcg@5']):.3f}")
    print(f"  P@1     {mean(series['p@1']):.3f}")
    print(f"  R@1     {mean(series['r@1']):.3f}")
    print(f"  P@5     {mean(series['p@5']):.3f}")
    print(f"  R@5     {mean(series['r@5']):.3f}")
    print(f"  F1@5    {mean(series['f1@5']):.3f}")


if __name__ == "__main__":
    main()
