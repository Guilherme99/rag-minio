from minio import Minio
import os

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "imagens"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Garante que o bucket existe
if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

# for com 50 imagens, use um loop para iterar sobre os arquivos e enviar cada um
for i in range(1, 51):
    image_path = f"imgs/1.png"
    object_name = f"imagem_{i}.png"

        
    metadata = {
        "tipo": f"lesao_cutanea_{i}",
        "area": f"dermatologia_{i}",
        "paciente_id": f"{i}",
        "ano": f"202{i}"
    }

    client.fput_object(
        BUCKET,
        object_name,
        image_path,
        content_type="image/png",
        metadata=metadata
    )

print("✅ Imagens enviadas com sucesso!")
