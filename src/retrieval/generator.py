import re

import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"
SOURCE_TAG = re.compile(r"\[SOURCE (\d+)\]")


@dataclass
class Citation:
    source_number: int
    foi_reference: str
    title: str
    page_number: int
    chunk_id: str


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    prompt_sent: str


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[SOURCE {i+1}] FOI {r.foi_reference} — {r.title}\n"
            f"Page {r.page_number}\n"
            f"{r.content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are answering questions about Freedom of Information requests made to Camden Council.

Use only the sources provided. Cite sources using [SOURCE N] inline. If the sources do not contain enough information to answer, say so.

Question: {query}

Sources:
{context}

Answer with inline citations:"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    # Only return sources the answer actually cites, in order of first mention.
    cited_numbers = []
    for match in SOURCE_TAG.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(results) and n not in cited_numbers:
            cited_numbers.append(n)

    citations = [
        Citation(
            source_number=n,
            foi_reference=results[n - 1].foi_reference,
            title=results[n - 1].title,
            page_number=results[n - 1].page_number,
            chunk_id=results[n - 1].chunk_id,
        )
        for n in cited_numbers
    ]

    return GeneratedAnswer(answer=answer, citations=citations, prompt_sent=prompt)