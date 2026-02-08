from minio import Minio
from config import *

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def list_images():
    client = get_minio_client()
    objects = client.list_objects(MINIO_BUCKET)

    results = []
    for obj in objects:
        stat = client.stat_object(MINIO_BUCKET, obj.object_name)

        metadata = dict(stat.metadata) if stat.metadata else {}

        results.append({
            "object_name": obj.object_name,
            "metadata": metadata
        })

    return results
