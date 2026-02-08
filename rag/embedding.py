from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

class ImageEmbedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        embedding = self.model.encode(image)
        return np.array(embedding, dtype="float32")
