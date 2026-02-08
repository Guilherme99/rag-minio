import tempfile
from minio_client import get_minio_client, list_images
from embedding_utils import ImageEmbedder
from lancedb_store import LanceDBStore
from rag_pipeline import RAGPipeline
from config import *

def save_temp_image(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(data)
    tmp.close()
    return tmp.name

def main():
    print("🚀 RAG MINIO + LANCEDB + OLLAMA")

    minio = get_minio_client()
    embedder = ImageEmbedder()
    store = LanceDBStore()
    rag = RAGPipeline()

    print("📦 Indexando imagens (modo híbrido)...")

    objects = list_images()

    for obj in objects:
        data = minio.get_object(MINIO_BUCKET, obj["object_name"]).read()
        path = save_temp_image(data)

        vector = embedder.embed_hybrid(path, obj["metadata"])
        store.add(vector, obj)

    print("✅ Indexação concluída:", len(objects))


    while True:
        question = input("\n❓ Pergunta: ")

        query_vector = embedder.embed_text(question)
        results = store.search(query_vector, TOP_K)

        print("\n📦 Ranking:")
        for r in results:
            print(r["object_name"], "->", round(float(r["score"]), 3))

        # menor distância = melhor
        best = min(results, key=lambda x: x["score"])

        similarity = round(float(best["score"]), 3)

        payload = {
            "existe": similarity,
            "objetos": []
        }

        payload["objetos"].append({
            "object_name": best["object_name"],
            "similarity": similarity,
            "metadata": best["metadata"]
        })

        answer = rag.format_answer(payload)

        print("\n📊 RESULTADO JSON:")
        print(answer)



if __name__ == "__main__":
    main()
