from minio_client import get_minio_client, list_images
from embedding import ImageEmbedder
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from utils import save_temp_image
from config import *

TOP_K = 3

def main():
    print("🚀 RAG IMAGEM + MINIO + OLLAMA")

    minio = get_minio_client()
    embedder = ImageEmbedder(EMBEDDING_MODEL)
    vector_store = VectorStore(dim=512)
    rag = RAGPipeline()

    print("📦 Indexando imagens do MinIO...")

    objects = list_images()

    for obj in objects:
        data = minio.get_object(MINIO_BUCKET, obj["object_name"]).read()
        img_path = save_temp_image(data)

        embedding = embedder.embed_image(img_path)
        vector_store.add(embedding, obj)

    print(f"✅ {len(objects)} imagens indexadas")

    question = input("\n❓ Pergunta: ")

    query_embedding = embedder.model.encode(question).astype("float32")

    results = vector_store.search(query_embedding, k=TOP_K)

    print("\n📦 OBJETOS RECUPERADOS:")
    for r in results:
        print(r)

    # 🔹 Construção do contexto multimodal
    context = "Resultados recuperados:\n"

    for i, r in enumerate(results, 1):
        context += f"""
Resultado {i}
Objeto: {r['object_name']}
Metadados: {r['metadata']}
Similaridade (distância L2): {r['score']}
"""

    answer = rag.answer(context, question)

    print("\n🧠 RESPOSTA DO LLM:")
    print(answer)


if __name__ == "__main__":
    main()
