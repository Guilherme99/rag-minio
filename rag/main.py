import tempfile
import numpy as np

from minio_client import get_minio_client, list_images
from embedding_utils import ImageEmbedder
from lancedb_store import LanceDBStore
from config import *
from rag_pipeline import RAGPipeline


# ============================================================
# UTIL
# ============================================================

def save_temp_image(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(data)
    tmp.close()
    return tmp.name


# ============================================================
# AVALIAÇÃO DE RECUPERAÇÃO
# ============================================================

def evaluate_retrieval(ground_truth, embedder, store, top_k=5):

    recall_total = 0
    precision_total = 0
    mrr_total = 0
    ap_total = 0
    hit_rate_total = 0

    relevant_scores = []
    irrelevant_scores = []

    questions = list(ground_truth.keys())
    total_queries = len(questions)

    for q in questions:

        query_vector = embedder.embed_text(q)
        results = store.search(query_vector, top_k)

        ranked_names = [r["object_name"] for r in results]
        ranked_scores = [float(r["score"]) for r in results]

        gt_set = set(ground_truth[q])

        # -------------------------
        # Recall@K
        # -------------------------
        hits = [name for name in ranked_names if name in gt_set]
        recall_total += len(hits) / len(gt_set)

        # -------------------------
        # Precision@K
        # -------------------------
        precision_total += len(hits) / top_k

        # -------------------------
        # HitRate@K
        # -------------------------
        if len(hits) > 0:
            hit_rate_total += 1

        # -------------------------
        # MRR
        # -------------------------
        reciprocal_rank = 0
        for idx, name in enumerate(ranked_names):
            if name in gt_set:
                reciprocal_rank = 1 / (idx + 1)
                break
        mrr_total += reciprocal_rank

        # -------------------------
        # Average Precision (AP)
        # -------------------------
        num_correct = 0
        precision_acc = 0

        for idx, name in enumerate(ranked_names):
            if name in gt_set:
                num_correct += 1
                precision_acc += num_correct / (idx + 1)

        ap_total += precision_acc / len(gt_set)

        # -------------------------
        # Similaridade relevante vs irrelevante
        # -------------------------
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
        metrics["SemanticGap"] = mean_irrel - mean_rel

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

    print("📦 Indexando imagens (modo híbrido)...")

    objects = list_images()

    for obj in objects:

        data = minio.get_object(MINIO_BUCKET, obj["object_name"]).read()
        path = save_temp_image(data)

        vector = embedder.embed_hybrid(path, obj["metadata"])
        store.add(vector, obj)

    print("✅ Indexação concluída:", len(objects))

    # ========================================================
    # GROUND TRUTH (EDITE AQUI COM SEUS NOMES REAIS)
    # ========================================================

    ground_truth = {
        "cachorro": ["1.jpg", "2.PNG"],
        "camisa laranja": ["3.PNG"],
        "camisa vermelha": ["4.PNG"],
        "presidente": ["2.jpg"]
    }

    # ========================================================
    # ESCOLHA DE MODO
    # ========================================================

    mode = input("\nDigite '1' para rodar avaliação ou 2 para modo interativo: ")

    if mode.lower() == "1":

        metrics = evaluate_retrieval(ground_truth, embedder, store, TOP_K)

        print("\n===== MÉTRICAS =====")
        for k, v in metrics.items():
            print(f"{k}: {round(v, 4)}")

        return

    # ========================================================
    # MODO INTERATIVO
    # ========================================================

    while True:

        question = input("\n❓ Pergunta: ")

        query_vector = embedder.embed_text(question)
        results = store.search(query_vector, TOP_K)

        print("\n📦 Ranking:")
        for r in results:
            print(r["object_name"], "->", round(float(r["score"]), 3))

        best = min(results, key=lambda x: x["score"])
        similarity = round(float(best["score"]), 3)

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