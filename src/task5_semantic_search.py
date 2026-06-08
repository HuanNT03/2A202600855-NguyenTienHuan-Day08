"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


# pyrefly: ignore [missing-import]
import weaviate
from sentence_transformers import SentenceTransformer

from task4_chunking_indexing import connect_to_weaviate_cloud, EMBEDDING_MODEL


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model ở Task 4
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    # Bước 2: Kết nối vector store (Weaviate)
    client = None
    try:
        client = connect_to_weaviate_cloud()
    except Exception:
        # Fallback to local connection
        client = weaviate.connect_to_local()

    try:
        collection_name = "DrugLawDocs"
        if not client.collections.exists(collection_name):
            print(f"Collection {collection_name} không tồn tại. Trả về kết quả rỗng.")
            return []

        collection = client.collections.get(collection_name)

        # Truy vấn near_vector
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )

        # Bước 3: Định dạng kết quả và tính score (cosine similarity = 1 - distance)
        search_results = []
        for obj in results.objects:
            distance = obj.metadata.distance
            score = 1.0 - distance if distance is not None else 0.0
            
            search_results.append({
                "content": obj.properties.get("content", ""),
                "score": score,
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0)
                }
            })

        # Sắp xếp kết quả giảm dần theo score
        search_results.sort(key=lambda x: x["score"], reverse=True)
        return search_results[:top_k]

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
