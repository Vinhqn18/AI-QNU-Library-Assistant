"""
load_pdf.py — Load tất cả file PDF từ thư mục pdfs/
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

PDF_FOLDER = "pdfs"

def load_all_pdfs() -> list[Document]:
    """Load tất cả PDF trong thư mục pdfs/, trả về danh sách Document"""
    folder = Path(PDF_FOLDER)
    if not folder.exists():
        print(f"⚠️  Thư mục '{PDF_FOLDER}' không tồn tại. Tạo thư mục và thêm file PDF vào.")
        folder.mkdir(exist_ok=True)
        return []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  Không có file PDF nào trong thư mục '{PDF_FOLDER}'.")
        return []

    all_docs = []
    for pdf_path in pdf_files:
        try:
            print(f"📄 Đang load: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            docs   = loader.load()

            # Thêm metadata tên file
            for doc in docs:
                doc.metadata["source"]    = pdf_path.name
                doc.metadata["file_type"] = "pdf"

            all_docs.extend(docs)
            print(f"   ✅ {len(docs)} trang")
        except Exception as e:
            print(f"   ❌ Lỗi load {pdf_path.name}: {e}")

    print(f"\n📦 Tổng cộng: {len(all_docs)} trang từ {len(pdf_files)} file PDF")
    return all_docs


if __name__ == "__main__":
    docs = load_all_pdfs()
    for i, d in enumerate(docs[:3]):
        print(f"\n--- Trang {i+1} ---")
        print(f"Nguồn: {d.metadata.get('source')} | Trang: {d.metadata.get('page')}")
        print(d.page_content[:200] + "...")
