import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class ImageEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_image(self, path: str):
        img = Image.open(path).convert("RGBA").convert("RGB")
        emb = self.model.encode(img, normalize_embeddings=True)
        return np.array(emb, dtype="float32")

    def embed_text(self, text: str):
        emb = self.model.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype="float32")

    def embed_hybrid(self, image_path: str, metadata: dict):
        image_vec = self.embed_image(image_path)

        metadata_text = " ".join(
            f"{k}:{v}" for k, v in metadata.items()
        ) if metadata else ""

        if metadata_text:
            text_vec = self.embed_text(metadata_text)
            hybrid = (image_vec + text_vec) / 2
        else:
            hybrid = image_vec

        return hybrid.astype("float32")
