"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import weaviate
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Chọn chunk_size 500 vì đây là độ dài lý tưởng (khoảng ~300-400 words), 
# đủ để chứa một ngữ cảnh trọn vẹn (ví dụ 1 khoản luật hoặc 1 đoạn tin tức) 
# mà không làm loãng thông tin khi LLM đọc.
CHUNK_OVERLAP = 50      # Overlap 50 giúp các chunk không bị đứt gãy ý một cách đột ngột.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Sử dụng bge-m3 vì đây là model hỗ trợ Multilingual (đa ngôn ngữ) cực kì tốt,
# đặc biệt hiệu quả trong việc nắm bắt ngữ nghĩa của Tiếng Việt (cả ngôn ngữ luật và báo chí).
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def connect_to_weaviate_cloud() -> weaviate.WeaviateClient:
    """
    Kết nối tới Weaviate Cloud (WCD) sử dụng URL và API key từ biến môi trường.
    """
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")
    if not url or not api_key or "xxx" in url or "xxx" in api_key:
        raise ValueError(
            "Vui lòng cấu hình WEAVIATE_URL và WEAVIATE_API_KEY trong file .env trước khi kết nối!"
        )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key)
    )


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"Thư mục {STANDARDIZED_DIR} không tồn tại.")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Xác định doc_type dựa vào thư mục cha
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type
            }
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # Khai báo các custom separators kết hợp với separators mặc định
    custom_separators = ["\n#", "\n##", "\n**", "\n***", "\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=custom_separators
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE != "weaviate":
        raise ValueError(f"Vector store '{VECTOR_STORE}' chưa được hỗ trợ.")

    from weaviate.classes.config import Configure, Property, DataType

    # Kết nối tới Weaviate
    client = None
    try:
        print("Đang thử kết nối tới Weaviate Cloud (WCD)...")
        client = connect_to_weaviate_cloud()
        print("✓ Kết nối Weaviate Cloud thành công!")
    except Exception as e:
        print(f"Không thể kết nối Weaviate Cloud: {e}")
        print("Đang thử kết nối Weaviate Local (connect_to_local)...")
        try:
            client = weaviate.connect_to_local()
            print("✓ Kết nối Weaviate Local thành công!")
        except Exception as local_err:
            print(f"Lỗi kết nối Local: {local_err}")
            raise RuntimeError("Không thể kết nối tới Weaviate Cloud lẫn Weaviate Local. Vui lòng kiểm tra lại thiết lập.")

    try:
        collection_name = "DrugLawDocs"

        # Kiểm tra và xóa collection cũ nếu tồn tại
        if client.collections.exists(collection_name):
            print(f"Đang xóa collection cũ: {collection_name}...")
            client.collections.delete(collection_name)

        print(f"Đang tạo collection mới: {collection_name}...")
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )

        print(f"Bắt đầu index {len(chunks)} chunks vào Weaviate...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )
        print("✓ Hoàn thành indexing dữ liệu thành công!")
    finally:
        if client:
            client.close()
            print("Đã đóng kết nối Weaviate.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
