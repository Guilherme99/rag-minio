import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class ImageEmbedder:

    def __init__(self, alpha=0.7):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.alpha = alpha

    # =====================================================

    def _normalize(self, vec):
        vec = np.array(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    # =====================================================

    def embed_image(self, path: str):
        img = Image.open(path).convert("RGB")
        vec = self.model.encode(img)
        return self._normalize(vec)

    # =====================================================

    def embed_text(self, text: str):
        vec = self.model.encode(text)
        return self._normalize(vec)

    # =====================================================

    def embed_hybrid(self, image_path: str, metadata: dict | None):
        image_vec = self.embed_image(image_path)

        if not metadata:
            return image_vec

        metadata_text = " ".join(
            str(v) for v in metadata.values()
            if isinstance(v, str)
        )

        if not metadata_text.strip():
            return image_vec

        text_vec = self.embed_text(
            f"photo of {metadata_text}"
        )

        hybrid = (
            image_vec * self.alpha +
            text_vec * (1 - self.alpha)
        )

        return self._normalize(hybrid)

    # =====================================================

    def cosine_similarity(self, v1, v2):
        return float(np.dot(v1, v2))