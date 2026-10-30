import anthropic
import json

client = anthropic.Anthropic()


def rerank(query: str, results: list, top_k: int = 5) -> list:
    if not results:
        return results

    candidates = "\n\n".join(
        f"[{i}] {r.content[:400]}" for i, r in enumerate(results)
    )

    prompt = f"""You are ranking search results for relevance to a query about Freedom of Information requests to Camden Council.

Query: {query}

Candidates:
{candidates}

Return a JSON array of indices ordered from most to least relevant. Only include indices of results that are genuinely relevant. Example: [2, 0, 4]

Return only the JSON array, nothing else."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        indices = json.loads(message.content[0].text.strip())
        seen = set()
        reranked = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(results) and i not in seen:
                seen.add(i)
                reranked.append(results[i])
        return reranked[:top_k]
    except (json.JSONDecodeError, KeyError):
        return results[:top_k]