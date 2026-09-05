-- Requires the pgvector extension. On a managed provider (Neon/Supabase/RDS) this is
-- usually a one-click "enable extension" toggle; on self-hosted Postgres you need the
-- pgvector server package installed first.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
    id            SERIAL PRIMARY KEY,
    product_id    TEXT UNIQUE NOT NULL,
    category      TEXT NOT NULL,
    subcategory   TEXT NOT NULL,
    brand         TEXT,
    image         BYTEA NOT NULL,      -- catalog image bytes (JPEG), served straight to the UI
    embedding     VECTOR(512) NOT NULL -- fine-tuned CLIP image embedding, L2-normalized
);

-- The ANN index (ivfflat) is deliberately NOT created here. ivfflat clusters based on
-- the data present at CREATE INDEX time, so db/ingest.py creates it after all rows are
-- loaded, which gives much better recall than building it on an empty table.
