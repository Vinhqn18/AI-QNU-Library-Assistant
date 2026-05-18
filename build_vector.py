import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from split_docs import split_documents

VECTOR_DB_PATH = "vector_db"

def build_vector_db():
    print("📄 Load & split documents...")
    docs = split_documents()

    # 🔥 LỌC TEXT HỢP LỆ
    texts = []
    metadatas = []

    for doc in docs:
        if isinstance(doc.page_content, str):
            text = doc.page_content.strip()
            if len(text) > 0:
                texts.append(text)
                metadatas.append(doc.metadata)

    print(f"🔢 Số đoạn hợp lệ sau khi lọc: {len(texts)}")

    print("🤖 Load sentence-transformer embedding...")
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"batch_size": 64}
    )

    print("📦 Build FAISS vector store...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)

    print("✅ Build FAISS Vector DB thành công!")

if __name__ == "__main__":
    build_vector_db()
