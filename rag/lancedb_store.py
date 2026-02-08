import lancedb
import numpy as np
import pyarrow as pa
from config import LANCEDB_PATH

class LanceDBStore:
    def __init__(self, dim=512):
        self.db = lancedb.connect(LANCEDB_PATH)

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

    def add(self, vector, obj):
        self.table.add([{
            "vector": vector.tolist(),
            "object_name": obj["object_name"],
            "metadata": str(obj["metadata"])
        }])

    def search(self, vector, k=3):
        results = (
            self.table.search(vector.tolist())
            .limit(k)
            .to_list()
        )

        return [
            {
                "object_name": r["object_name"],
                "metadata": r["metadata"],
                "score": float(r["_distance"])
            }
            for r in results
        ]
