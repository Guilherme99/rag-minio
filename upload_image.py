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
# for i in range(1, 51):
image_path = f"imgs/coelho.png"
object_name = f"imagem6.png"

    
metadata = {
    "tipo": f"animal",
    "cor": f"marrom",
    "tamanho": f"medio"
}

client.fput_object(
    BUCKET,
    object_name,
    image_path,
    content_type="image/png",
    metadata=metadata
)

print("✅ Imagens enviadas com sucesso!")
