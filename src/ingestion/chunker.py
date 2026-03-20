import re
from dataclasses import dataclass
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 800


@dataclass
class Chunk:
    page_number: int
    chunk_index: int
    content: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    chunks = []
    for page_number, raw_text in pages:
        text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
        if not text:
            continue
        tokens = count_tokens(text)
        if tokens <= MAX_TOKENS:
            chunks.append(Chunk(page_number, 0, text, tokens))
        else:
            sentences = split_sentences(text)
            current, current_tokens, idx = [], 0, 1
            for sentence in sentences:
                st = count_tokens(sentence)
                if current_tokens + st > MAX_TOKENS and current:
                    content = " ".join(current)
                    chunks.append(Chunk(page_number, idx, content, count_tokens(content)))
                    idx += 1
                    current = current[-1:]  # 1-sentence overlap
                    current_tokens = count_tokens(current[0])
                current.append(sentence)
                current_tokens += st
            if current:
                content = " ".join(current)
                chunks.append(Chunk(page_number, idx, content, count_tokens(content)))
    return chunks