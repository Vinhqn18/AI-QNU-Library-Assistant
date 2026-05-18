"""
split_docs.py — Load dữ liệu từ 3 file CSV và chuyển thành Documents cho LangChain
"""

import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cấu hình từng file: path + dòng header thực tế
CSV_CONFIGS = [
    {"path": "clean_BaoCao_DsTaiLieuSo.csv", "header": 1},  # dòng 0 là tiêu đề lớn
    {"path": "clean_qnu_allITEM.csv",         "header": 0},
    {"path": "clean_books.csv",               "header": 0},
]

TITLE_VARIANTS   = ["Tiêu đề","ten_sach","Tieu de","Tên sách","title"]
AUTHOR_VARIANTS  = ["Tác giả","tac_gia","Tac gia","nguoi_so_hoa","author"]
SUBJECT_VARIANTS = ["Chủ đề","chu_de","Chu de","chuyen_nganh","subject","Học phần","Hoc phan"]
LINK_VARIANTS    = ["Link URL","Link so","link","linkso","url","LinkURL"]
KEYWORD_VARIANTS = ["Từ khóa","tu_khoa","Tu khoa","keyword","Chủ đề","Chu de"]

JUNK_PATTERNS = ["tuần","tuan","m4","m3","10 ctđt","aun","nghỉ","nghi",
                 "tên sách","ten sach","stt","unnamed"]

def find_col(headers, variants):
    import unicodedata
    def norm(s):
        s = unicodedata.normalize("NFD", str(s).lower())
        return "".join(c for c in s if unicodedata.category(c) != "Mn").replace(" ","")
    norm_variants = [norm(v) for v in variants]
    for h in headers:
        if norm(h) in norm_variants:
            return h
    return None

def is_junk(title):
    t = str(title).strip().lower()
    if len(t) < 3 or t == "nan": return True
    return any(t.startswith(j) for j in JUNK_PATTERNS)

def load_csv_as_documents():
    all_docs = []

    for cfg in CSV_CONFIGS:
        csv_path = cfg["path"]
        header_row = cfg["header"]
        path = Path(csv_path)

        if not path.exists():
            print(f"⚠️  Không tìm thấy: {csv_path}")
            continue

        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig",
                             header=header_row, on_bad_lines="skip")
            if df.shape[1] <= 1:
                df = pd.read_csv(path, sep=",", encoding="utf-8-sig",
                                 header=header_row, on_bad_lines="skip")

            # Bỏ dòng đầu nếu là header lặp (chứa "STT" hoặc "Tiêu đề")
            headers = list(df.columns)
            col_title   = find_col(headers, TITLE_VARIANTS)
            col_author  = find_col(headers, AUTHOR_VARIANTS)
            col_subject = find_col(headers, SUBJECT_VARIANTS)
            col_link    = find_col(headers, LINK_VARIANTS)
            col_keyword = find_col(headers, KEYWORD_VARIANTS)

            print(f"📄 {path.name}: {len(df)} dòng")
            print(f"   Cột → title={col_title}, author={col_author}, link={col_link}")

            if not col_title:
                print(f"   ❌ Không nhận diện được cột tiêu đề! Cột hiện có: {headers[:6]}")
                continue

            count = 0
            for _, row in df.iterrows():
                title = str(row.get(col_title, "")).strip()
                if is_junk(title):
                    continue

                author  = str(row.get(col_author,  "") if col_author  else "").replace("nan","").strip()
                subject = str(row.get(col_subject, "") if col_subject else "").replace("nan","").strip()
                link    = str(row.get(col_link,    "") if col_link    else "").replace("nan","").strip()
                keyword = str(row.get(col_keyword, "") if col_keyword else "").replace("nan","").strip()

                # Bỏ dòng không có link hợp lệ (tùy chọn — bỏ comment nếu muốn)
                # if not link.startswith("http"): continue

                content = f"Tiêu đề: {title}"
                if author  and author  != "nan": content += f"\nTác giả: {author}"
                if subject and subject != "nan": content += f"\nChủ đề: {subject}"
                if keyword and keyword != "nan": content += f"\nTừ khóa: {keyword}"
                if link.startswith("http"):      content += f"\nLink: {link}"

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "title":  title,
                        "author": author,
                        "link":   link if link.startswith("http") else "",
                    }
                )
                all_docs.append(doc)
                count += 1

            print(f"   ✅ Đã tạo {count} documents")

        except Exception as e:
            print(f"❌ Lỗi {csv_path}: {e}")

    print(f"\n📦 Tổng cộng: {len(all_docs)} documents")
    return all_docs


def split_documents():
    docs = load_csv_as_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split = splitter.split_documents(docs)
    print(f"✂️  Số đoạn sau khi chia: {len(split)}")
    return split


if __name__ == "__main__":
    docs = split_documents()
    for i, d in enumerate(docs[:3]):
        print(f"\n--- Đoạn {i+1} ---")
        print(d.page_content)
        print("Metadata:", d.metadata)
