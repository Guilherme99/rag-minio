import os
import shutil
import lancedb
import numpy as np
import pyarrow as pa
from config import LANCEDB_PATH


class LanceDBStore:

    def __init__(self, dim=512, reset=False):

        self.dim = dim
        self.db_path = LANCEDB_PATH

        if reset and os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        self.db = lancedb.connect(self.db_path)

        schema = pa.schema([
            ("vector", pa.list_(pa.float32(), dim)),
            ("object_name", pa.string()),
            ("metadata", pa.string())
        ])

        self.table = self.db.create_table(
            "image_index",
            schema=schema,
            mode="overwrite"
        )

    # =====================================================

    def add(self, vector, obj):

        self.table.add([{
            "vector": vector.tolist(),
            "object_name": obj["object_name"],
            "metadata": str(obj["metadata"])
        }])

    # =====================================================

    def search(self, vector, k=5):

        results = (
            self.table.search(vector.tolist())
            .limit(k)
            .to_list()
        )

        formatted = []

        for r in results:
            # Lance retorna _distance (quanto menor melhor)
            similarity = 1 - float(r["_distance"])

            formatted.append({
                "object_name": r["object_name"],
                "metadata": r["metadata"],
                "score": similarity
            })

        # Ordena por similaridade real
        formatted.sort(key=lambda x: x["score"], reverse=True)

        return formatted

    # =====================================================

    def get_index_size(self):

        total = 0
        for root, dirs, files in os.walk(self.db_path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))

        return total / (1024 * 1024)