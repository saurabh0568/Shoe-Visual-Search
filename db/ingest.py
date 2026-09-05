import argparse
import json
import os

import psycopg2
from tqdm import tqdm

from db.connection import get_connection

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def apply_schema():
    from config import DATABASE_URL
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with open(SCHEMA_PATH, "r") as f:
            schema_sql = f.read()
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    finally:
        conn.close()


def ingest(export_dir: str):
    catalog_path = os.path.join(export_dir, "catalog_export.json")
    images_dir = os.path.join(export_dir, "catalog_images")

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"{catalog_path} not found. Run Section 16 of the notebook first and copy "
            f"its 'export_for_app' output here (or point --export-dir at it)."
        )

    with open(catalog_path, "r") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} catalog records from {catalog_path}")

    print("Creating pgvector extension + products table...")
    apply_schema()

    with get_connection() as conn:
        with conn.cursor() as cur:
            for rec in tqdm(records, desc="Ingesting into Postgres"):
                image_path = os.path.join(images_dir, rec["image_filename"])
                with open(image_path, "rb") as img_f:
                    image_bytes = img_f.read()

                cur.execute(
                    """
                    INSERT INTO products (product_id, category, subcategory, brand, image, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (product_id) DO UPDATE SET
                        category    = EXCLUDED.category,
                        subcategory = EXCLUDED.subcategory,
                        brand       = EXCLUDED.brand,
                        image       = EXCLUDED.image,
                        embedding   = EXCLUDED.embedding
                    """,
                    (
                        rec["product_id"],
                        rec["category"],
                        rec["subcategory"],
                        rec["brand"],
                        psycopg2.Binary(image_bytes),
                        rec["embedding"],
                    ),
                )
        conn.commit()

        # Build the ANN index once the data is in place (see note in schema.sql).
        print("Building ivfflat index...")
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS products_embedding_idx
                ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """
            )
            cur.execute("ANALYZE products;")
        conn.commit()

    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="data/export_for_app",
                         help="Folder containing catalog_export.json + catalog_images/")
    args = parser.parse_args()
    ingest(args.export_dir)