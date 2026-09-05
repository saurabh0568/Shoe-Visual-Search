from functools import lru_cache
import spaces  
import clip
import torch
from PIL import Image

from config import CLIP_MODEL_NAME, SHOE_GATE_THRESHOLD

_SHOE_PROMPTS = [
    "a photo of a shoe",
    "a photo of a sneaker",
    "a photo of a boot",
    "a photo of a sandal",
    "a photo of a slipper",
    "a photo of a pair of shoes",
]

_NON_SHOE_PROMPTS = [
    "a photo of furniture",
    "a photo of a person",
    "a photo of an animal",
    "a photo of food",
    "a photo of a vehicle",
    "a photo of a building or room",
    "a photo of an electronic device",
    "a photo of clothing that is not footwear",
    "a photo of a landscape or nature scene",
    "a photo of something unrelated to shoes",
]

_ALL_PROMPTS = _SHOE_PROMPTS + _NON_SHOE_PROMPTS


@lru_cache(maxsize=2)
def _load_gate_model(device: str):
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()

    text_tokens = clip.tokenize(_ALL_PROMPTS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features


@spaces.GPU
def shoe_confidence(pil_image: Image.Image) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, text_features = _load_gate_model(device)
    image_tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    text_features = text_features.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = similarity.cpu().numpy().flatten()
    return float(probs[:len(_SHOE_PROMPTS)].sum())


def is_shoe_image(pil_image: Image.Image, threshold: float = SHOE_GATE_THRESHOLD):
    confidence = shoe_confidence(pil_image)
    return confidence >= threshold, confidence