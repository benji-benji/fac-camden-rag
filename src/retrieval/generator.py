import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()


@dataclass
class Citation:
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
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    citations = [
        Citation(
            foi_reference=r.foi_reference,
            title=r.title,
            page_number=r.page_number,
            chunk_id=r.chunk_id,
        )
        for i, r in enumerate(results)
    ]

    return GeneratedAnswer(answer=answer, citations=citations, prompt_sent=prompt)