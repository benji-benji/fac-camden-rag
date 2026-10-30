CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      TEXT NOT NULL UNIQUE,
    foi_reference TEXT,
    date          DATE,
    title         TEXT,
    response_text TEXT,
    total_pages   INT,
    indexed_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INT NOT NULL,
    chunk_index  INT NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding    VECTOR(1536),
    token_count  INT,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING GIN (content_tsv);

CREATE TABLE IF NOT EXISTS ingestion_log (
    filename     TEXT PRIMARY KEY,
    file_hash    TEXT NOT NULL,
    status       TEXT NOT NULL,
    error        TEXT,
    ingested_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query               TEXT NOT NULL,
    filters             JSONB,
    retrieved_chunk_ids UUID[],
    prompt_sent         TEXT,
    response            TEXT,
    model               TEXT,
    queried_at          TIMESTAMPTZ DEFAULT now()
);