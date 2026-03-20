from openai import OpenAI

client = OpenAI()
BATCH_SIZE = 128
MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=MODEL)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings