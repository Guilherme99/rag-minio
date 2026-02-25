import os
import tempfile
import numpy as np

from minio_client import get_minio_client, list_images
from embedding_utils import ImageEmbedder
from lancedb_store import LanceDBStore
from config import *
from rag_pipeline import RAGPipeline


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth(root_dir="."):

    ground_truth = {}

    for pasta in os.listdir(root_dir):

        if pasta in ["cat", "dog", "person", "landscape"]:

            caminho = os.path.join(root_dir, pasta)

            if os.path.isdir(caminho):

                arquivos = [
                    f for f in os.listdir(caminho)
                    if os.path.isfile(os.path.join(caminho, f))
                ]

                if arquivos:
                    ground_truth[pasta] = arquivos

    return ground_truth


# ============================================================

def save_temp_image(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(data)
    tmp.close()
    return tmp.name


# ============================================================
# RAG EVALUATION
# ============================================================

def evaluate_rag(ground_truth, embedder, store, rag, top_k=10):

    faithfulness_scores = []
    factual_correctness_scores = []
    similarity_scores = []
    correctness_scores = []

    for label, gt_files in ground_truth.items():

        # 🔥 Prompt melhorado para CLIP
        query_vector = embedder.embed_text(
            f"photo of a {label}"
        )

        results = store.search(query_vector, top_k)

        ranked_names = [r["object_name"] for r in results]

        payload = {
            "objetos": results
        }

        generated_answer = rag.format_answer(payload)

        ground_truth_answer = f"A classe correta é {label}"

        # Faithfulness
        faithfulness = int(
            any(name in generated_answer for name in ranked_names)
        )
        faithfulness_scores.append(faithfulness)

        # Factual Correctness
        factual_correctness = int(
            label.lower() in generated_answer.lower()
        )
        factual_correctness_scores.append(factual_correctness)

        # Similaridade semântica
        gen_vec = embedder.embed_text(generated_answer)
        gt_vec = embedder.embed_text(ground_truth_answer)

        similarity = float(np.dot(gen_vec, gt_vec))
        similarity_scores.append(similarity)

        # Métrica combinada
        answer_correctness = (
            0.75 * factual_correctness +
            0.25 * similarity
        )

        correctness_scores.append(answer_correctness)

    return {
        "Faithfulness": np.mean(faithfulness_scores),
        "FactualCorrectness": np.mean(factual_correctness_scores),
        "AnswerSimilarity": np.mean(similarity_scores),
        "AnswerCorrectness": np.mean(correctness_scores),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("🚀 RAG MINIO + LANCEDB + OLLAMA (Improved Version)")

    minio = get_minio_client()
    embedder = ImageEmbedder()
    store = LanceDBStore(reset=True)
    rag = RAGPipeline()

    print("📦 Indexando imagens...")

    objects = list_images()

    for obj in objects:

        data = minio.get_object(
            MINIO_BUCKET,
            obj["object_name"]
        ).read()

        path = save_temp_image(data)

        # 🔥 Pode trocar para embed_hybrid se quiser testar
        vector = embedder.embed_image(path)

        store.add(vector, obj)

    print("✅ Indexação concluída:", len(objects))

    ground_truth = build_ground_truth(".")

    mode = input("\nDigite '1' para avaliação ou '2' para interativo: ")

    if mode == "1":

        metrics = evaluate_rag(
            ground_truth,
            embedder,
            store,
            rag,
            top_k=10
        )

        print("\n===== MÉTRICAS RAG =====")
        for k, v in metrics.items():
            print(f"{k}: {round(v, 4)}")

        return

    while True:

        question = input("\n❓ Pergunta: ")

        query_vector = embedder.embed_text(question)
        results = store.search(query_vector, 10)

        print("\n📦 Ranking:")
        for r in results:
            print(r["object_name"], "->", round(r["score"], 4))

        payload = {"objetos": results}
        answer = rag.format_answer(payload)

        print("\n📊 RESULTADO:")
        print(answer)


if __name__ == "__main__":
    main()