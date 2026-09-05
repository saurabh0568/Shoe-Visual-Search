from functools import lru_cache
import spaces  
import clip
import numpy as np
import torch
from PIL import Image

from config import CLIP_MODEL_NAME, FINETUNED_WEIGHTS_PATH


@lru_cache(maxsize=2)
def _load_model(device: str):
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model = model.float()

    state_dict = torch.load(FINETUNED_WEIGHTS_PATH, map_location=device)
    model.visual.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded fine-tuned CLIP visual encoder from {FINETUNED_WEIGHTS_PATH} on {device}")
    return model, preprocess


@spaces.GPU
def embed_image(pil_image: Image.Image) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = _load_model(device)
    image_tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    vec = embedding.cpu().numpy().flatten().astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()
