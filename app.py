import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import chainlit as cl

# Load environment variables
load_dotenv()

# Add project root to sys.path to ensure relative/absolute imports work nicely
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import SYSTEM_PROMPT, TEMPERATURE, TOP_P, reorder_for_llm, format_context
except ImportError:
    from task9_retrieval_pipeline import retrieve
    from task10_generation import SYSTEM_PROMPT, TEMPERATURE, TOP_P, reorder_for_llm, format_context


def rewrite_query(query: str, chat_history: list[dict]) -> str:
    """
    Sử dụng LLM để viết lại câu hỏi tiếp nối thành câu hỏi độc lập (standalone query)
    dựa trên lịch sử hội thoại. Nếu lịch sử trống hoặc câu hỏi đã độc lập, trả về query gốc.
    """
    if not chat_history:
        return query

    # Lấy 6 lượt hội thoại gần nhất để tránh tràn token và tối ưu hóa context
    history_str = ""
    for turn in chat_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_str += f"{role}: {turn['content']}\n"

    prompt = f"""Dưới đây là lịch sử cuộc trò chuyện và một câu hỏi tiếp theo của người dùng.
Hãy viết lại câu hỏi tiếp theo này thành một câu hỏi độc lập, đầy đủ ngữ cảnh để có thể dùng tìm kiếm tài liệu chính xác về pháp luật ma tuý.
Không trả lời câu hỏi, không thêm bất kỳ lời dẫn nào khác. Chỉ trả về duy nhất câu hỏi độc lập đã được viết lại bằng tiếng Việt.

Lịch sử trò chuyện:
{history_str}

Câu hỏi tiếp theo: {query}

Câu hỏi độc lập:"""

    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            api_key = dashscope_key
            base_url = "https://ws-rcvsccl6lcfj1n1s.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
            
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="qwen3.5-122b-a10b",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo chuyên viết lại câu hỏi tiếp nối thành câu hỏi độc lập đầy đủ ngữ cảnh."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        rewritten = response.choices[0].message.content.strip()
        # Loại bỏ các dấu ngoặc kép thừa nếu mô hình tự động thêm vào
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        if rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1].strip()
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"Lỗi khi rewrite query: {e}")
    return query


@cl.on_chat_start
async def on_chat_start():
    """
    Kích hoạt khi bắt đầu phiên chat mới.
    Khởi tạo lịch sử trò chuyện và hiển thị thông điệp chào mừng cùng các câu hỏi gợi ý.
    """
    cl.user_session.set("chat_history", [])

    welcome_message = """# ⚖️ Chào mừng bạn đến với **DrugLaw RAG Chatbot**!

Tôi là trợ lý AI chuyên biệt về **Pháp luật Phòng chống ma tuý** tại Việt Nam. Tôi có thể hỗ trợ bạn giải đáp các vấn đề như:
- 🚫 **Hình phạt các hành vi tàng trữ, mua bán, sử dụng trái phép ma tuý**
- 🏥 **Quy trình và điều kiện cai nghiện bắt buộc / tự nguyện (theo Luật 2021)**
- 📰 **Tin tức cập nhật về các vụ án ma tuý liên quan đến người nổi tiếng**

---
### 💡 **Các câu hỏi gợi ý dành cho bạn:**
1. *Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam là gì?*
2. *Nghệ sĩ nào bị bắt vì liên quan tới ma tuý gần đây?*
3. *Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021 được quy định ra sao?*

Hãy nhập câu hỏi của bạn xuống bên dưới để trò chuyện cùng tôi! 👇
"""
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Xử lý câu hỏi của người dùng khi họ gửi tin nhắn.
    - Hiển thị ngay câu hỏi của người dùng (tự động bởi Chainlit).
    - Tạo một message placeholder trống cho câu trả lời và cập nhật tiến trình.
    - Thực hiện Query Rewriting trong trường hợp là câu hỏi tiếp nối.
    - Gọi Retrieval Pipeline để lấy context tài liệu (hybrid search + reranking + pageindex).
    - Sắp xếp tài liệu tránh 'lost in the middle'.
    - Gọi LLM qwen3.5-122b-a10b và stream kết quả về UI.
    - Hiển thị tài liệu tham chiếu (source documents) ở khung bên cạnh (side panel).
    - Lưu lại lịch sử trò chuyện.
    """
    # 1. Tạo tin nhắn phản hồi trống và gửi ngay để tạo loading/typing indicator
    msg = cl.Message(content="")
    await msg.send()

    # Lấy lịch sử trò chuyện hiện tại
    chat_history = cl.user_session.get("chat_history", [])

    # 2. Xử lý câu hỏi tiếp nối bằng cách rewrite câu hỏi (không chặn event loop)
    msg.content = "*Đang xử lý và làm rõ ngữ cảnh câu hỏi...*"
    await msg.update()
    
    # Chạy rewrite_query trên thread riêng tránh block async loop
    standalone_query = await asyncio.to_thread(rewrite_query, message.content, chat_history)

    # Hiển thị thông tin tìm kiếm
    if standalone_query != message.content:
        msg.content = f"*Đang tìm kiếm tài liệu tham chiếu cho câu hỏi: '{standalone_query}'...*"
    else:
        msg.content = "*Đang tìm kiếm tài liệu tham chiếu...*"
    await msg.update()

    # 3. Truy xuất tài liệu từ pipeline (không chặn event loop)
    try:
        # retrieve chạy các tác vụ nặng (như embedding sinh ra từ SentenceTransformer)
        chunks = await asyncio.to_thread(retrieve, standalone_query)
    except Exception as e:
        print(f"Lỗi khi retrieve tài liệu: {e}")
        chunks = []

    # 4. Cập nhật trạng thái trước khi gọi LLM sinh văn bản
    if not chunks:
        msg.content = "*Không tìm thấy tài liệu liên quan trong hệ thống dữ liệu. Đang chuẩn bị phản hồi...*"
    else:
        msg.content = f"*Tìm thấy {len(chunks)} tài liệu liên quan. Đang biên soạn câu trả lời...*"
    await msg.update()

    # 5. Xử lý các tài liệu tham chiếu
    reordered_chunks = reorder_for_llm(chunks)
    context = format_context(reordered_chunks)

    # 6. Chuẩn bị prompt gọi model qwen3.5-122b-a10b
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    # Nạp lịch sử cuộc trò chuyện
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    # Thêm câu hỏi hiện tại có chứa thông tin ngữ cảnh tìm kiếm được
    user_message_content = f"Context:\n{context}\n\n---\n\nQuestion: {message.content}"
    messages.append({"role": "user", "content": user_message_content})

    # Cấu hình API OpenAI compatible client
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_key:
        api_key = dashscope_key
        base_url = "https://ws-rcvsccl6lcfj1n1s.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

    # Reset tin nhắn phản hồi về rỗng chuẩn bị streaming
    msg.content = ""
    await msg.update()

    # 7. Gọi LLM ở chế độ Stream
    from openai import AsyncOpenAI
    try:
        async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await async_client.chat.completions.create(
            model="qwen3.5-122b-a10b",
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True
        )

        full_answer = ""
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            full_answer += token
            await msg.stream_token(token)

    except Exception as e:
        error_msg = f"Đã xảy ra lỗi khi gọi mô hình sinh câu trả lời: {str(e)}"
        print(error_msg)
        msg.content = error_msg
        await msg.update()
        return

    # 8. Hiển thị tài liệu tham chiếu (source documents) trên Side Panel
    elements = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Tài liệu {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        
        elements.append(
            cl.Text(
                name=source,
                content=f"Loại tài liệu: {doc_type}\nNguồn: {source}\n\nNội dung:\n{chunk['content']}",
                display="side"
            )
        )

    # Đính kèm elements vào tin nhắn và cập nhật
    if elements:
        msg.elements = elements
    await msg.update()

    # 9. Lưu trữ câu hỏi và câu trả lời vào lịch sử trò chuyện
    chat_history.append({"role": "user", "content": message.content})
    chat_history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("chat_history", chat_history)
