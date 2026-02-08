import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, vector, meta):
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(meta)

    def search(self, vector, k=3):
        distances, indices = self.index.search(vector.reshape(1, -1), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "object_name": self.metadata[idx]["object_name"],
                    "metadata": self.metadata[idx]["metadata"],
                    "score": float(dist)
                })

        return results
