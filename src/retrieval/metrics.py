from __future__ import annotations

from dataclasses import dataclass
from math import log2
from pathlib import Path

import pandas as pd


@dataclass
class QueryEvalResult:
    query_text: str
    matched_gold_rows: int
    relevant_count: int
    predicted_count: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    f1_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    average_precision: float
    mar: float


def load_golden_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "query_text" not in df.columns and "query" in df.columns:
        df = df.rename(columns={"query": "query_text"})
    required = {"query_text", "chunk_id", "relevance_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in golden dataset: {sorted(missing)}")
    df["query_text_norm"] = df["query_text"].astype(str).str.strip().str.lower()
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce").fillna(0.0)
    return df


def _precision_recall_f1_at_k(
    predicted: list[str],
    relevant_binary: set[str],
    k: int,
) -> tuple[float, float, float]:
    top_k = predicted[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_binary)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_binary) if relevant_binary else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _average_precision(predicted: list[str], relevant_binary: set[str]) -> float:
    if not relevant_binary:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, chunk_id in enumerate(predicted, start=1):
        if chunk_id in relevant_binary:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / len(relevant_binary)


def _dcg(scores: list[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(scores[:k], start=1):
        gain = (2**rel) - 1
        total += gain / log2(i + 1)
    return total


def _ndcg_at_k(predicted: list[str], gold_scores: dict[str, float], k: int) -> float:
    actual_scores = [gold_scores.get(chunk_id, 0.0) for chunk_id in predicted[:k]]
    ideal_scores = sorted(gold_scores.values(), reverse=True)[:k]
    actual_dcg = _dcg(actual_scores, k)
    ideal_dcg = _dcg(ideal_scores, k)
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def evaluate_query_ranking(
    query_text: str,
    predicted_chunk_ids: list[str],
    golden_df: pd.DataFrame,
    ks: list[int] | None = None,
    relevant_threshold: float = 2.0,
) -> QueryEvalResult:
    ks = ks or [1, 3, 5, 10]
    query_norm = query_text.strip().lower()
    gold_rows = golden_df[golden_df["query_text_norm"] == query_norm].copy()

    if gold_rows.empty:
        return QueryEvalResult(
            query_text=query_text,
            matched_gold_rows=0,
            relevant_count=0,
            predicted_count=len(predicted_chunk_ids),
            precision_at_k={k: 0.0 for k in ks},
            recall_at_k={k: 0.0 for k in ks},
            f1_at_k={k: 0.0 for k in ks},
            ndcg_at_k={k: 0.0 for k in ks},
            average_precision=0.0,
            mar=0.0,
        )

    gold_scores = {
        str(row["chunk_id"]): float(row["relevance_score"])
        for _, row in gold_rows.iterrows()
    }
    relevant_binary = {
        chunk_id for chunk_id, score in gold_scores.items() if score >= relevant_threshold
    }
    predicted = [str(chunk_id) for chunk_id in predicted_chunk_ids]

    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    f1_at_k: dict[int, float] = {}
    ndcg_at_k: dict[int, float] = {}

    for k in ks:
        p, r, f1 = _precision_recall_f1_at_k(predicted, relevant_binary, k)
        precision_at_k[k] = p
        recall_at_k[k] = r
        f1_at_k[k] = f1
        ndcg_at_k[k] = _ndcg_at_k(predicted, gold_scores, k)

    average_precision = _average_precision(predicted, relevant_binary)
    mar = sum(recall_at_k.values()) / len(recall_at_k) if recall_at_k else 0.0

    return QueryEvalResult(
        query_text=query_text,
        matched_gold_rows=len(gold_rows),
        relevant_count=len(relevant_binary),
        predicted_count=len(predicted),
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        f1_at_k=f1_at_k,
        ndcg_at_k=ndcg_at_k,
        average_precision=average_precision,
        mar=mar,
    )
