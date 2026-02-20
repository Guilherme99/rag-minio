import os
import uuid
import random
import requests
from minio import Minio

# ================================
# CONFIG MINIO
# ================================

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "imagens"

TOTAL_IMAGES = 100

# ================================
# CLIENT
# ================================

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Garante bucket
if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

# ================================
# METADADOS SIMULADOS
# ================================

tipos = ["animal", "pessoa", "paisagem", "objeto", "cidade"]
cores = ["vermelho", "azul", "verde", "preto", "branco", "amarelo"]
tamanhos = ["pequeno", "medio", "grande"]

# ================================
# DOWNLOAD + UPLOAD
# ================================

print("🚀 Baixando e enviando imagens aleatórias...")

for i in range(TOTAL_IMAGES):

    try:
        # imagem aleatória real
        url = f"https://picsum.photos/400/400?random={uuid.uuid4()}"
        response = requests.get(url)

        if response.status_code != 200:
            print("Erro ao baixar imagem")
            continue

        # salva temporário
        file_name = f"imagem_{i}.png"
        with open(file_name, "wb") as f:
            f.write(response.content)

        metadata = {
            "tipo": random.choice(tipos),
            "cor": random.choice(cores),
            "tamanho": random.choice(tamanhos)
        }

        client.fput_object(
            BUCKET,
            file_name,
            file_name,
            content_type="image/png",
            metadata=metadata
        )

        os.remove(file_name)

        print(f"✅ {file_name} enviada")

    except Exception as e:
        print("Erro:", e)

print("\n🎉 100 imagens enviadas com sucesso!")