from io import BytesIO
from PIL import Image
from config import TOP_K_DEFAULT, CATALOG_PAGE_SIZE, LOW_CONFIDENCE_SIMILARITY
from db.connection import get_dict_connection
from model.encoder import embed_image
from model.gate import is_shoe_image

def _row_image(row) -> Image.Image:
    return Image.open(BytesIO(bytes(row["image"]))).convert("RGB")


def search_similar(pil_image: Image.Image, top_k: int = TOP_K_DEFAULT) -> dict:
    passed_gate, confidence = is_shoe_image(pil_image)
    if not passed_gate:
        return {"status": "rejected", "shoe_confidence": confidence, "results": []}

    query_vec = embed_image(pil_image)

    with get_dict_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id, category, subcategory, brand, image,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM products
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, query_vec, top_k),
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            "product_id": row["product_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "brand": row["brand"],
            "similarity": float(row["similarity"]) * 100,
            "image": _row_image(row),
        })

    status = "ok"
    if results and results[0]["similarity"] < LOW_CONFIDENCE_SIMILARITY:
        status = "low_confidence"

    return {"status": status, "shoe_confidence": confidence, "results": results}


def get_catalog_page(page: int = 1, page_size: int = CATALOG_PAGE_SIZE,
                      category: str = None, subcategory: str = None, brand: str = None):
    """Returns (items, total_count) for one page of the product catalog, the images
    coming straight out of Postgres -- nothing is read from disk at request time."""
    page = max(1, int(page))
    offset = (page - 1) * page_size

    filters, params = [], []
    if category:
        filters.append("category = %s")
        params.append(category)
    if subcategory:
        filters.append("subcategory = %s")
        params.append(subcategory)
    if brand:
        filters.append("brand = %s")
        params.append(brand)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with get_dict_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM products {where_clause}", params)
            total = cur.fetchone()["cnt"]

            cur.execute(
                f"""
                SELECT product_id, category, subcategory, brand, image
                FROM products
                {where_clause}
                ORDER BY id
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    items = [{
        "product_id": row["product_id"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "brand": row["brand"],
        "image": _row_image(row),
    } for row in rows]

    return items, total


def get_filter_options():
    with get_dict_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT category FROM products ORDER BY category")
            categories = [r["category"] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT subcategory FROM products ORDER BY subcategory")
            subcategories = [r["subcategory"] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT brand FROM products WHERE brand IS NOT NULL ORDER BY brand")
            brands = [r["brand"] for r in cur.fetchall()]
    return categories, subcategories, brands
