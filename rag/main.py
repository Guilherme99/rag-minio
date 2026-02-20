import os
import tempfile
import numpy as np

from minio_client import get_minio_client, list_images
from embedding_utils import ImageEmbedder
from lancedb_store import LanceDBStore
from config import *
from rag_pipeline import RAGPipeline


# ============================================================
# GROUND TRUTH AUTOMÁTICO
# ============================================================

def build_ground_truth(root_dir="."):
    ground_truth = {}

    for pasta in os.listdir(root_dir):
        print(pasta)

        if(pasta in ["cat", "dog", "person", "landscape"]):
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
# UTIL
# ============================================================

def save_temp_image(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(data)
    tmp.close()
    return tmp.name


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate_retrieval(ground_truth, embedder, store, top_k=5):

    recall_total = 0
    precision_total = 0
    mrr_total = 0
    ap_total = 0
    hit_rate_total = 0

    relevant_scores = []
    irrelevant_scores = []

    total_queries = len(ground_truth.keys())

    for label, gt_files in ground_truth.items():

        query_vector = embedder.embed_text(label)
        results = store.search(query_vector, top_k)

        ranked_names = [r["object_name"] for r in results]
        ranked_scores = [float(r["score"]) for r in results]

        gt_set = set(gt_files)

        hits = [name for name in ranked_names if name in gt_set]
        print(hits)
        # Recall@K
        recall_total += len(hits) / len(gt_set)

        # Precision@K
        precision_total += len(hits) / top_k

        # HitRate@K
        if hits:
            hit_rate_total += 1

        # MRR
        reciprocal_rank = 0
        for idx, name in enumerate(ranked_names):
            if name in gt_set:
                reciprocal_rank = 1 / (idx + 1)
                break
        mrr_total += reciprocal_rank

        # AP
        num_correct = 0
        precision_acc = 0
        for idx, name in enumerate(ranked_names):
            if name in gt_set:
                num_correct += 1
                precision_acc += num_correct / (idx + 1)

        ap_total += precision_acc / len(gt_set)

        # Scores
        for name, score in zip(ranked_names, ranked_scores):
            if name in gt_set:
                relevant_scores.append(score)
            else:
                irrelevant_scores.append(score)

    metrics = {
        "Recall@K": recall_total / total_queries,
        "Precision@K": precision_total / total_queries,
        "MRR": mrr_total / total_queries,
        "HitRate@K": hit_rate_total / total_queries,
        "mAP": ap_total / total_queries,
    }

    if relevant_scores and irrelevant_scores:
        mean_rel = np.mean(relevant_scores)
        mean_irrel = np.mean(irrelevant_scores)

        metrics["MeanRelevantScore"] = mean_rel
        metrics["MeanIrrelevantScore"] = mean_irrel
        metrics["SemanticGap"] = mean_rel - mean_irrel  # corrigido

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("🚀 RAG MINIO + LANCEDB + OLLAMA")

    minio = get_minio_client()
    embedder = ImageEmbedder()
    store = LanceDBStore()
    rag = RAGPipeline()

    print("📦 Indexando imagens...")

    objects = list_images()

    for obj in objects:
        data = minio.get_object(MINIO_BUCKET, obj["object_name"]).read()
        path = save_temp_image(data)

        vector = embedder.embed_image(path)
        # vector = embedder.embed_hybrid(path, obj["metadata"])
        store.add(vector, obj)

    print("✅ Indexação concluída:", len(objects))

    # GROUND TRUTH AUTOMÁTICO
    ground_truth = build_ground_truth(".")

    print("\nGround truth detectado:")
    print(ground_truth)

    mode = input("\nDigite '1' para avaliação ou '2' para interativo: ")

    if mode == "1":

        metrics = evaluate_retrieval(
            ground_truth,
            embedder,
            store,
            TOP_K
        )

        print("\n===== MÉTRICAS =====")
        for k, v in metrics.items():
            print(f"{k}: {round(v, 4)}")

        return

    while True:

        question = input("\n❓ Pergunta: ")

        query_vector = embedder.embed_text(question)
        results = store.search(query_vector, TOP_K)

        print("\n📦 Ranking:")
        for r in results:
            print(r["object_name"], "->", round(float(r["score"]), 4))

        # Se for cosine similarity use max
        best = max(results, key=lambda x: x["score"])

        similarity = round(float(best["score"]), 4)

        payload = {
            "existe": similarity,
            "objetos": [
                {
                    "object_name": best["object_name"],
                    "similarity": similarity,
                    "metadata": best["metadata"]
                }
            ]
        }

        answer = rag.format_answer(payload)

        print("\n📊 RESULTADO JSON:")
        print(answer)


if __name__ == "__main__":
    main()