import numpy as np

'''
Position 1: Document 7  (relevant)
Position 2: Document 2  (not relevant)
Position 3: Document 12 (relevant)
Position 4: Document 5  (not relevant)
Position 5: Document 18 (relevant)
'''

def precision_at_k(retrieved_ids, relevant_ids):
    """What fraction of the returned results were relevant?"""
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    if len(retrieved) == 0:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)

def recall_at_k(retrieved_ids, relevant_ids):
    """What fraction of the relevant items did we return?"""
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    if len(relevant) == 0:
        return 0.0
    return len(retrieved & relevant) / len(relevant)

## Our example
retrieved = [7, 2, 12, 5, 18]
relevant = [3, 7, 12, 18]

print(f"Precision@5: {precision_at_k(retrieved, relevant):.2f}")
print(f"Recall@5:    {recall_at_k(retrieved, relevant):.2f}")

print(f"Precision@3: {precision_at_k(retrieved[:3], relevant):.2f}")
print(f"Recall@3:    {recall_at_k(retrieved[:3], relevant):.2f}")

print("k  | Precision@k | Recall@k")
print("---+-------------+---------")
for k in [1, 2, 3, 4, 5]:
    p = precision_at_k(retrieved[:k], relevant)
    r = recall_at_k(retrieved[:k], relevant)
    print(f" {k} |       {p:.2f}  |    {r:.2f}")
    
    
def reciprocal_rank(retrieved_ids, relevant_ids):
    """1 / position of the first relevant result. 0 if none found."""
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(queries):
    """Average reciprocal rank across multiple queries.
    queries is a list of (retrieved_ids, relevant_ids) pairs."""
    if len(queries) == 0:
        return 0.0
    return np.mean([reciprocal_rank(r, rel) for r, rel in queries])

## Our original example: first relevant result is at position 1
rr = reciprocal_rank(retrieved, relevant)
print(f"Reciprocal rank: {rr:.2f}")

## The worse retrieval: first relevant result is at position 3
worse_retrieved = [2, 5, 7, 12, 18]
rr_worse = reciprocal_rank(worse_retrieved, relevant)
print(f"Reciprocal rank (worse): {rr_worse:.2f}")

## MRR across both queries
mrr = mean_reciprocal_rank([
    (retrieved, relevant),
    (worse_retrieved, relevant),
])
print(f"MRR across both queries: {mrr:.2f}")

## Three queries, each with a list of retrieved results and the known relevant documents
eval_set = [
    {
        "query": "What optimiser was used?",
        "retrieved": [7, 2, 12, 5, 18],
        "relevant": [3, 7, 12, 18],
    },
    {
        "query": "What was the learning rate?",
        "retrieved": [15, 8, 3, 1, 9],
        "relevant": [3, 15],
    },
    {
        "query": "How was the model evaluated?",
        "retrieved": [10, 11, 14, 6, 20],
        "relevant": [10, 14, 19],
    },
]

print(f"{'Query':<35} {'P@5':>6} {'R@5':>6} {'RR':>6}")
print("-" * 57)

rr_scores = []
for item in eval_set:
    p = precision_at_k(item["retrieved"], item["relevant"])
    r = recall_at_k(item["retrieved"], item["relevant"])
    rr = reciprocal_rank(item["retrieved"], item["relevant"])
    rr_scores.append(rr)
    print(f"{item['query']:<35} {p:>6.2f} {r:>6.2f} {rr:>6.2f}")

print("-" * 57)
avg_p = np.mean([precision_at_k(q["retrieved"], q["relevant"]) for q in eval_set])
avg_r = np.mean([recall_at_k(q["retrieved"], q["relevant"]) for q in eval_set])
mrr = np.mean(rr_scores)
print(f"{'Average':<35} {avg_p:>6.2f} {avg_r:>6.2f} {mrr:>6.2f}")