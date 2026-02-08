import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        # 🔹 índice para cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, vector, meta):
        vector = self._normalize(vector)
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, vector, k=3):
        vector = self._normalize(vector)
        scores, indices = self.index.search(vector.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "object_name": self.metadata[idx]["object_name"],
                    "metadata": self.metadata[idx]["metadata"],
                    "score": float(score)  # agora maior = melhor
                })

        return results

    def _normalize(self, v):
        v = v.astype("float32")
        return v / np.linalg.norm(v)
