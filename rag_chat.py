"""
rag_chat.py — RAG Chat dùng FAISS + Gemini API (LCEL style)
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

VECTOR_DB_PATH = "vector_db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def load_vector_db():
    print("📦 Load FAISS vector DB...")
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"batch_size": 64}
    )
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Đã load vector DB!")
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(api_key: str):
    vectorstore = load_vector_db()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template("""
Bạn là trợ lý AI thông minh của Thư viện Trường Đại học Quy Nhơn (QNU Library Assistant).

=====================
VAI TRÒ
=====================

- Hỗ trợ sinh viên và giảng viên tìm kiếm, khai thác tài nguyên thư viện.
- Hỗ trợ tra cứu sách, giáo trình, luận văn, tài liệu nghiên cứu.
- Gợi ý tài liệu học tập theo nhiều chuyên ngành.
- Hỗ trợ học tập và nghiên cứu học thuật.
- Trả lời thân thiện, chính xác, ngắn gọn, dễ hiểu.

Hệ thống hỗ trợ nhiều lĩnh vực:
- Công nghệ thông tin
- Trí tuệ nhân tạo
- Kinh tế
- Kế toán
- Tài chính ngân hàng
- Quản trị kinh doanh
- Sư phạm Toán
- Ngữ Văn
- Tiếng Anh
- Kỹ thuật điện
- Và các ngành liên quan khác.

=====================
NGUYÊN TẮC TRẢ LỜI
=====================
- Ưu tiên sử dụng thông tin trong CONTEXT trước.
- Nếu CONTEXT chưa đầy đủ, có thể sử dụng kiến thức chung để hỗ trợ người dùng.
- Không tự bịa tài liệu hoặc thông tin không có cơ sở.

1. Với câu hỏi liên quan đến sách/tài liệu:
- Chỉ sử dụng thông tin trong CONTEXT.
- Không tự bịa tài liệu không tồn tại.
- Nếu có nhiều tài liệu → trình bày dạng danh sách.

Mỗi tài liệu nên hiển thị:
- Tên sách/tài liệu
- Tác giả
- Chuyên ngành (nếu có)
- Link hoặc nguồn (nếu có)

2. Với câu hỏi tìm tài liệu:
Ví dụ:
- "Có sách nào về AI không?"
- "Giáo trình kế toán tài chính?"
- "Sách Python cho người mới?"
- "Tài liệu quản trị kinh doanh?"
- "Giáo trình Deep Learning?"
→ Hãy gợi ý tài liệu phù hợp từ CONTEXT.

3. Với câu hỏi kiến thức chung:
Ví dụ:
- "AI là gì?"
- "RAG hoạt động như thế nào?"
- "Machine Learning khác Deep Learning?"
- "Kinh tế vi mô là gì?"
- "Nguyên lý kế toán?"
→ Trả lời tự nhiên bằng kiến thức của bạn.

- Không cần phụ thuộc hoàn toàn vào CONTEXT.
- Nếu CONTEXT có tài liệu liên quan thì có thể gợi ý thêm.

4. Với yêu cầu học tập/nghiên cứu:
Ví dụ:
- Tóm tắt sách
- Giải thích nội dung
- Hướng dẫn học tập
- Gợi ý keyword nghiên cứu
- Hướng dẫn làm báo cáo/đồ án
→ Hãy hỗ trợ như một trợ lý học tập thông minh.
- Nếu người dùng yêu cầu:
  + giải thích nội dung sách,
  + phân tích tác phẩm,
  + tóm tắt kiến thức,
  + giải thích khái niệm,
  + hướng dẫn học tập,
thì hãy cố gắng hỗ trợ bằng:
- thông tin trong CONTEXT (nếu có),
- kết hợp kiến thức chung phù hợp.

- Không nên chỉ trả lời "không tìm thấy tài liệu" nếu vẫn có thể hỗ trợ kiến thức liên quan.
5. Nếu CONTEXT không có đầy đủ thông tin:
- Không tự bịa tài liệu hoặc thông tin sai.
- Nếu là câu hỏi kiến thức chung hoặc học thuật:
  → vẫn cố gắng hỗ trợ bằng kiến thức của bạn.
- Nếu là yêu cầu tra cứu tài liệu cụ thể mà không có dữ liệu:
  → trả lời lịch sự và gợi ý truy cập:
https://lib.qnu.edu.vn

Ví dụ:
"Hiện tại tôi chưa tìm thấy tài liệu phù hợp trong cơ sở dữ liệu thư viện. Bạn có thể tra cứu thêm tại https://lib.qnu.edu.vn"

6. Ưu tiên tuyệt đối thông tin trong CONTEXT:
   - Nếu CONTEXT chứa nội dung chi tiết từ PDF: Hãy trích dẫn, tóm tắt hoặc giải thích dựa trên đúng nội dung đó.
   - Nếu CONTEXT chỉ chứa tên sách (từ CSV): Hãy liệt kê thông tin sách và gợi ý người dùng tìm đọc bản đầy đủ.      

7.  Với yêu cầu giải thích/tóm tắt nội dung sách (Đặc biệt cho PDF):
   - Nếu người dùng hỏi "Sách này nói về gì?", "Chương 1 có nội dung gì?": 
     + Trình bày rõ ràng các ý chính.
     + Sử dụng ngôn ngữ học thuật nhưng dễ hiểu.
     + Nếu nội dung quá dài, hãy tóm tắt thành các gạch đầu dòng.                                                                                                                              
=====================
PHONG CÁCH TRẢ LỜI
=====================

- Luôn dùng tiếng Việt.
- Thân thiện và chuyên nghiệp.
- Trả lời ngắn gọn nhưng đầy đủ ý.
- Có thể dùng bullet list để dễ đọc.
- Ưu tiên thông tin rõ ràng và chính xác.

=====================
CONTEXT:
{context}

=====================
CÂU HỎI:
{question}

=====================
TRẢ LỜI:
""")

    # LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def chat():
    api_key = GEMINI_API_KEY
    if not api_key:
        api_key = input("🔑 Nhập Gemini API key: ").strip()
    if not api_key:
        print("❌ Thiếu API key!")
        return

    print("\n💬 RAG Chat Thư viện QNU — Gõ 'exit' để thoát\n")
    rag_chain, retriever = build_rag_chain(api_key)
    print("✅ Sẵn sàng!\n" + "─"*50)

    while True:
        question = input("\n👤 Bạn: ").strip()
        if not question:
            continue
        if question.lower() in ["exit","quit","thoat"]:
            print("👋 Tạm biệt!")
            break

        try:
            # Lấy context trước để hiện link
            docs = retriever.invoke(question)
            links = list(set(
                d.metadata.get("link","")
                for d in docs
                if d.metadata.get("link","").startswith("http")
            ))

            # Gọi RAG chain
            answer = rag_chain.invoke(question)
            print(f"\n🤖 Trợ lý: {answer}")

            if links:
                print("\n🔗 Link tài liệu số:")
                for lnk in links[:4]:
                    print(f"   {lnk}")

        except Exception as e:
            print(f"❌ Lỗi: {e}")

        print("─"*50)

if __name__ == "__main__":
    chat()
