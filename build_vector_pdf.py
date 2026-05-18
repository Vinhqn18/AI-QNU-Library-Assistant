"""
build_vector_pdf.py — Tạo FAISS Vector DB riêng cho các file PDF
Chạy: python build_vector_pdf.py

Yêu cầu:
    pip install pypdf langchain-community sentence-transformers faiss-cpu
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_pdf import load_all_pdfs

VECTOR_DB_PDF_PATH = "vector_db_pdf"

def build_vector_db_pdf():
    print("=" * 50)
    print("📚 BUILD VECTOR DB CHO PDF")
    print("=" * 50)

    # 1. Load PDF
    print("\n📄 Bước 1: Load tất cả PDF...")
    docs = load_all_pdfs()
    if not docs:
        print("❌ Không có tài liệu nào để xử lý. Hãy thêm file PDF vào thư mục 'pdfs/'.")
        return

    # 2. Tách đoạn văn bản
    print(f"\n✂️  Bước 2: Tách văn bản (chunk_size=800, overlap=150)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = splitter.split_documents(docs)

    # Lọc đoạn hợp lệ
    valid_docs = []
    for doc in split_docs:
        text = doc.page_content.strip() if isinstance(doc.page_content, str) else ""
        if len(text) > 30:  # bỏ đoạn quá ngắn
            valid_docs.append(doc)

    print(f"   ✅ {len(valid_docs)} đoạn hợp lệ (từ {len(split_docs)} đoạn ban đầu)")

    # 3. Embedding
    print(f"\n🤖 Bước 3: Load embedding model...")
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"batch_size": 32}
    )

    # 4. Build FAISS
    print(f"\n📦 Bước 4: Build FAISS index ({len(valid_docs)} đoạn)...")
    print("   (Quá trình này có thể mất vài phút tùy số lượng trang PDF...)")
    vectorstore = FAISS.from_documents(valid_docs, embeddings)

    # 5. Lưu
    Path(VECTOR_DB_PDF_PATH).mkdir(exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PDF_PATH)

    print(f"\n✅ Hoàn thành! Vector DB PDF lưu tại: {VECTOR_DB_PDF_PATH}/")
    print(f"   - index.faiss")
    print(f"   - index.pkl")
    print(f"\n📊 Thống kê:")
    print(f"   - Số file PDF:    {len(list(Path('pdfs').glob('*.pdf')))} file")
    print(f"   - Số đoạn chunk:  {len(valid_docs)} đoạn")


if __name__ == "__main__":
    build_vector_db_pdf()
