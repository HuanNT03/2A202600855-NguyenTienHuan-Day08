"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex import PageIndexClient
    
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        raise ValueError("PAGEINDEX_API_KEY chưa được cấu hình hợp lệ trong file .env.")
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    if not STANDARDIZED_DIR.exists():
        print(f"Thư mục {STANDARDIZED_DIR} không tồn tại.")
        return []
    uploaded_doc_ids = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        # Upload và submit document lên PageIndex
        response = client.submit_document(file_path=str(md_file))
        doc_id = response.get("doc_id")
        print(f"  ✓ Uploaded: {md_file.name} (doc_id: {doc_id})")
        if doc_id:
            uploaded_doc_ids.append(doc_id)
    return uploaded_doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient
    import time
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        raise ValueError("PAGEINDEX_API_KEY chưa được cấu hình hợp lệ trong file .env.")
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Liệt kê tài liệu đã có
    docs_resp = client.list_documents(limit=50)
    documents = docs_resp.get("documents", [])
    results = []
    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        
        # Gửi truy vấn
        query_resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = query_resp.get("retrieval_id")
        if not retrieval_id:
            continue
        # Chờ kết quả retrieval (polling tối đa 10 lần)
        retries = 10
        retrieval_result = None
        while retries > 0:
            status_resp = client.get_retrieval(retrieval_id)
            if status_resp.get("status") == "completed":
                retrieval_result = status_resp
                break
            elif status_resp.get("status") == "failed":
                break
            time.sleep(1)
            retries -= 1
        if retrieval_result:
            nodes = retrieval_result.get("results", []) or retrieval_result.get("nodes", []) or retrieval_result.get("chunks", [])
            for node in nodes:
                results.append({
                    "content": node.get("text", "") or node.get("content", ""),
                    "score": float(node.get("score", 0.5)),
                    "metadata": node.get("metadata", {}),
                    "source": "pageindex"
                })
    # Sắp xếp kết quả theo score giảm dần
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
