"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
from rank_bm25 import BM25Okapi

try:
    from src.task4_chunking_indexing import load_documents, chunk_documents
except ImportError:
    from task4_chunking_indexing import load_documents, chunk_documents

# Load corpus từ data/standardized/ hoặc từ vector store
try:
    docs = load_documents()
    CORPUS: list[dict] = chunk_documents(docs)
except Exception as e:
    print(f"Lỗi khi load corpus: {e}")
    CORPUS: list[dict] = []

# Khởi tạo toàn cục BM25_INDEX
BM25_INDEX = None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    # Tokenize - split() đơn giản và chuyển thành chữ thường
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# Tự động khởi tạo BM25_INDEX khi import module
if CORPUS:
    BM25_INDEX = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global BM25_INDEX
    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)
        if BM25_INDEX is None:
            return []

    # Tokenize query
    tokenized_query = query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)

    # Lọc các kết quả có điểm score > 0
    results = []
    for idx, score in enumerate(scores):
        if score > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(score),
                "metadata": CORPUS[idx]["metadata"]
            })

    # Sắp xếp theo score giảm dần
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
