"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional


import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vector."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    # Thử sử dụng Jina Reranker API
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key and not jina_key.startswith("jina_xxx") and len(jina_key) > 10:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                return [
                    {**candidates[r["index"]], "score": float(r["relevance_score"])}
                    for r in reranked
                ]
        except Exception as e:
            print(f"Jina API rerank failed: {e}. Falling back to local similarity...")

    # Fallback: sử dụng cosine similarity bằng BAAI/bge-m3
    try:
        from src.task4_chunking_indexing import EMBEDDING_MODEL
    except ImportError:
        EMBEDDING_MODEL = "BAAI/bge-m3"

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_emb = model.encode(query, convert_to_numpy=True)
    doc_embs = model.encode([c["content"] for c in candidates], convert_to_numpy=True)
    
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    
    scores = np.dot(doc_embs, query_emb) / (doc_norms * query_norm + 1e-10)
    
    reranked_candidates = []
    for idx, score in enumerate(scores):
        reranked_candidates.append({
            **candidates[idx],
            "score": float(score)
        })
        
    reranked_candidates.sort(key=lambda x: x["score"], reverse=True)
    return reranked_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Tạo embedding nếu thiếu
    for idx in range(len(candidates)):
        if "embedding" not in candidates[idx] or not candidates[idx]["embedding"]:
            try:
                from src.task4_chunking_indexing import EMBEDDING_MODEL
                model = SentenceTransformer(EMBEDDING_MODEL)
                candidates[idx]["embedding"] = model.encode(candidates[idx]["content"]).tolist()
            except Exception:
                candidates[idx]["embedding"] = [0.0] * 1024

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to selected docs
            max_sim_to_selected = 0.0
            if selected:
                max_sim_to_selected = max(
                    cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                )

            # MMR score
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        try:
            from src.task4_chunking_indexing import EMBEDDING_MODEL
            model = SentenceTransformer(EMBEDDING_MODEL)
            query_embedding = model.encode(query).tolist()
        except Exception:
            query_embedding = [0.0] * 1024
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k)
        else:
            return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
