import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shoedb")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
FINETUNED_WEIGHTS_PATH = os.getenv("FINETUNED_WEIGHTS_PATH", "model/weights/clip_visual_finetuned.pt")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

SHOE_GATE_THRESHOLD = float(os.getenv("SHOE_GATE_THRESHOLD", "0.5"))

LOW_CONFIDENCE_SIMILARITY = float(os.getenv("LOW_CONFIDENCE_SIMILARITY", "55"))

CATALOG_PAGE_SIZE = int(os.getenv("CATALOG_PAGE_SIZE", "24"))
