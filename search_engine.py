"""
search_engine.py — BM25 + Regex Hybrid Full-Text Search Engine
Lightweight alternative to FAISS embedding (no ML model needed)

Hỗ trợ load từ:
  - CSV (clean_BaoCao_DsTaiLieuSo.csv)
  - XLSX (tailieu_templates_AI.xlsx)
"""

import re
import os
import pickle
import pandas as pd
from pathlib import Path
from collections import Counter
from text_utils import normalize_text

DATA_DIR = Path("Data")

class BM25SearchEngine:
    """BM25 ranking algorithm for efficient full-text search"""

    def __init__(self, documents):
        self.documents = documents
        self.k1 = 1.5
        self.b = 0.75
        self.idf_cache = {}
        self._build_index()

    def _build_index(self):
        self.tokenized_docs = []
        self.doc_lengths = []
        self.vocabulary = Counter()

        for doc in self.documents:
            text = normalize_text(doc.get('text', '')) + " " + normalize_text(doc.get('title', ''))
            tokens = self._tokenize(text)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.vocabulary.update(tokens)

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
        self.num_docs = len(self.documents)
        print(f"✅ BM25 Index built: {self.num_docs} docs, {len(self.vocabulary)} unique terms")

    def _tokenize(self, text):
        tokens = re.findall(r'\w+', text.lower(), re.UNICODE)
        return [t for t in tokens if len(t) > 1]

    def _get_idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]
        doc_freq = sum(1 for tokens in self.tokenized_docs if term in tokens)
        idf = max(0.1, self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5)
        self.idf_cache[term] = idf
        return idf

    def _bm25_score(self, tokens, doc_idx):
        score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        for token in tokens:
            if token in doc_tokens:
                term_freq = doc_tokens.count(token)
                idf = self._get_idf(token)
                numerator = idf * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                score += numerator / denominator

        return score

    def search(self, query, top_k=6, use_regex=True):
        normalized_query = normalize_text(query)
        tokens = self._tokenize(normalized_query)

        if not tokens:
            return []

        scores = []
        for doc_idx in range(len(self.documents)):
            bm25_score = self._bm25_score(tokens, doc_idx)

            regex_boost = 0.0
            if use_regex and len(query) > 3:
                doc_text = normalize_text(self.documents[doc_idx].get('text', ''))
                pattern = re.escape(normalized_query)
                if re.search(pattern, doc_text, re.IGNORECASE):
                    regex_boost = 5.0

            total_score = bm25_score + regex_boost
            if total_score > 0:
                scores.append((doc_idx, total_score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scores[:top_k]:
            results.append({
                'doc': self.documents[doc_idx],
                'score': score,
                'index': doc_idx
            })

        return results


class DocumentIndexer:
    """Build and manage document index from CSV and XLSX files"""

    @staticmethod
    def load_from_data(data_dir=None):
        """
        Load documents từ tất cả file dữ liệu trong thư mục Data/

        Args:
            data_dir: đường dẫn đến thư mục Data (mặc định: "Data")

        Returns:
            list of document dicts
        """
        if data_dir is None:
            data_dir = DATA_DIR
        data_path = Path(data_dir)

        # ── Cache check ──────────────────────────────────────
        cache_file = data_path / "_documents_cache.pkl"
        source_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.xlsx"))
        use_cache = False
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            running_on_vercel = os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))
            if running_on_vercel or all(f.stat().st_mtime < cache_mtime for f in source_files):
                try:
                    with open(cache_file, "rb") as f:
                        documents = pickle.load(f)
                    print(f"📦 Loaded {len(documents)} documents from cache ({cache_file.name})")
                    return documents
                except Exception as e:
                    print(f"⚠️ Cache load failed: {e}")

        # ── Full load ────────────────────────────────────────
        documents = []

        # 1. Load CSV (clean_BaoCao_DsTaiLieuSo.csv)
        csv_file = data_path / "clean_BaoCao_DsTaiLieuSo.csv"
        if csv_file.exists():
            try:
                csv_docs = DocumentIndexer._load_csv(str(csv_file))
                documents.extend(csv_docs)
                print(f"✅ Loaded {len(csv_docs)} documents from {csv_file.name}")
            except Exception as e:
                print(f"⚠️ Error loading {csv_file.name}: {e}")
        else:
            print(f"⚠️  File not found: {csv_file}")

        # 2. Load XLSX (tailieu_templates_AI.xlsx)
        xlsx_file = data_path / "tailieu_templates_AI.xlsx"
        xlsx_docs = []
        if xlsx_file.exists():
            try:
                xlsx_docs = DocumentIndexer._load_xlsx(str(xlsx_file))
                documents.extend(xlsx_docs)
                print(f"✅ Loaded {len(xlsx_docs)} documents from {xlsx_file.name}")
            except Exception as e:
                print(f"⚠️ Error loading {xlsx_file.name}: {e}")
        else:
            print(f"⚠️  File not found: {xlsx_file}")

        # 3. Cross-reference: gán link từ CSV vào XLSX dựa trên tên sách
        csv_link_map = {}
        for doc in documents:
            if 'csv' in doc.get('source', '') and doc.get('link'):
                norm = normalize_text(doc['title'])
                if norm and len(norm) > 5:
                    csv_link_map[norm] = doc['link']

        link_count = 0
        for doc in documents:
            if 'xlsx' in doc.get('source', '') and not doc.get('link'):
                norm = normalize_text(doc['title'])
                if norm in csv_link_map:
                    doc['link'] = csv_link_map[norm]
                    link_count += 1
                else:
                    for csv_norm, csv_link in csv_link_map.items():
                        if len(csv_norm) > 10 and (csv_norm in norm or norm in csv_norm):
                            doc['link'] = csv_link
                            link_count += 1
                            break

        if link_count:
            print(f"🔗 Cross-referenced {link_count} links from CSV to XLSX documents")

        # 4. Save cache for next startup
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 Cached {len(documents)} documents to {cache_file.name}")
        except Exception as e:
            print(f"⚠️ Cache write failed: {e}")

        print(f"📦 Total documents loaded: {len(documents)}")
        return documents

    @staticmethod
    def _load_csv(filepath):
        """Load documents from CSV file"""
        documents = []

        df = pd.read_csv(filepath, encoding='utf-8', delimiter=';', skiprows=1)
        df = df.rename(columns={
            'Tiêu đề': 'title',
            'Tác giả': 'author',
            'Năm xuất bản': 'year',
            'Nhà xuất bản': 'publisher',
            'Chủ đề': 'subject',
            'Chuyên ngành': 'major',
            'Học phần': 'course',
            'Loại tài liệu': 'doc_type',
            'Kiểu tài liệu': 'format',
            'Ngôn ngữ': 'language',
            'Link URL': 'link',
            'Ghi chú': 'notes',
            'Tác giả phụ': 'co_author',
        })

        for idx, row in df.iterrows():
            title = str(row.get('title', 'Untitled')).strip() if pd.notna(row.get('title')) else 'Untitled'
            author = str(row.get('author', 'Unknown')).strip() if pd.notna(row.get('author')) else 'Unknown'
            year = str(row.get('year', 'N/A')).strip() if pd.notna(row.get('year')) else 'N/A'
            subject = str(row.get('subject', 'N/A')).strip() if pd.notna(row.get('subject')) else 'N/A'
            link = str(row.get('link', '')).strip() if pd.notna(row.get('link')) else ''
            doc_type = str(row.get('doc_type', 'N/A')).strip() if pd.notna(row.get('doc_type')) else 'N/A'
            doc_format = str(row.get('format', 'Số')).strip() if pd.notna(row.get('format')) else 'Số'
            notes = str(row.get('notes', '')).strip() if pd.notna(row.get('notes')) else ''
            publisher = str(row.get('publisher', '')).strip() if pd.notna(row.get('publisher')) else ''
            major = str(row.get('major', '')).strip() if pd.notna(row.get('major')) else ''
            course = str(row.get('course', '')).strip() if pd.notna(row.get('course')) else ''
            language = str(row.get('language', '')).strip() if pd.notna(row.get('language')) else ''

            # Build searchable text
            text_parts = [title, author, subject, doc_type, publisher, major, course, notes]
            text = ' '.join(p for p in text_parts if p)

            document = {
                'title': title,
                'author': author,
                'year': year,
                'subject': subject,
                'link': link,
                'doc_type': doc_type,
                'format': doc_format,
                'publisher': publisher,
                'major': major,
                'course': course,
                'language': language,
                'notes': notes,
                'source': filepath,
                'csv_file': str(filepath),
                'text': text,
            }
            documents.append(document)

        return documents

    @staticmethod
    def _clean_marc_value(val):
        """Remove MARC prefix markers like $a, $b, $c from values"""
        if not val or not isinstance(val, str):
            return val
        # Remove MARC subfield codes: $a, $b, $c, etc.
        cleaned = re.sub(r'\$[a-z]\s*', '', val)
        # Remove hanging punctuation at start/end
        cleaned = cleaned.strip().strip(',').strip(';').strip('/').strip(':').strip()
        return cleaned

    @staticmethod
    def _load_xlsx(filepath):
        """Load documents from XLSX file (tailieu_templates_AI.xlsx) với cấu trúc MARC"""
        documents = []

        df = pd.read_excel(filepath, dtype=str)

        # Map MARC-like columns to readable fields
        column_map = {
            'Chỉ số phân loai DDC(082$a)': 'ddc',
            'Nhan đề chính(245$a)': 'title',
            'Nhan đề khác(245$b)': 'subtitle',
            'Trách nhiệm(245$c)': 'author',
            'Tên nhà xuất bản (260$b)': 'publisher',
            'Năm xuất bản (260$c)': 'year',
            'Tóm tắt (520$a)': 'abstract',
            'Chuyên ngành đào tạo(526$a)': 'major',
            'Loại tài liệu(526$b)': 'doc_type',
        }

        df_renamed = df.rename(columns=column_map)
        cleaner = DocumentIndexer._clean_marc_value

        for idx, row in df_renamed.iterrows():
            title = cleaner(str(row.get('title', ''))) if pd.notna(row.get('title')) else ''
            subtitle = cleaner(str(row.get('subtitle', ''))) if pd.notna(row.get('subtitle')) else ''
            author = cleaner(str(row.get('author', 'Unknown'))) if pd.notna(row.get('author')) else 'Unknown'
            year = cleaner(str(row.get('year', 'N/A'))) if pd.notna(row.get('year')) else 'N/A'
            year = re.sub(r'\D', '', year)  # Extract digits only
            publisher = cleaner(str(row.get('publisher', ''))) if pd.notna(row.get('publisher')) else ''
            abstract = cleaner(str(row.get('abstract', ''))) if pd.notna(row.get('abstract')) else ''
            major = cleaner(str(row.get('major', ''))) if pd.notna(row.get('major')) else ''
            doc_type = cleaner(str(row.get('doc_type', 'Sách'))) if pd.notna(row.get('doc_type')) else 'Sách'
            ddc = cleaner(str(row.get('ddc', ''))) if pd.notna(row.get('ddc')) else ''

            if not title and not author:
                continue

            # Build full title: avoid double punctuation
            if subtitle:
                subtitle_clean = subtitle.lstrip(': ').lstrip(':').strip()
                if title.endswith(':') or title.endswith(':'):
                    full_title = f"{title} {subtitle_clean}"
                else:
                    full_title = f"{title}: {subtitle_clean}"
            else:
                full_title = title

            # Collect location information from location columns
            location_parts = []
            location_cols = [c for c in df.columns if 'phòng' in c.lower() or 'mượn' in c.lower() or 'đọc' in c.lower() or 'thiếu' in c.lower() or 'thừa' in c.lower()]
            for col in location_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    clean_val = str(val).strip().lstrip('|').strip()
                    if clean_val not in ('nan', ''):
                        short_col = col.replace('(526$b)', '').replace('(082$a)', '').strip()
                        # Remove duplicate room name prefix from value
                        if clean_val.lower().startswith(short_col.lower()):
                            clean_val = clean_val[len(short_col):].lstrip(': ').lstrip('|').strip()
                        location_parts.append(f"{short_col}: {clean_val[:60]}")

            location_str = ' ; '.join(location_parts) if location_parts else ''

            # Build searchable text
            text_parts = [full_title, author, publisher, major, abstract, doc_type]
            text = ' '.join(p for p in text_parts if p and p not in ('Unknown', 'N/A'))
            if abstract:
                text = f"{full_title} {author} {abstract}"

            document = {
                'title': full_title,
                'author': author,
                'year': year if year else 'N/A',
                'subject': major if major else publisher,
                'link': '',
                'doc_type': doc_type,
                'format': 'Sách',
                'publisher': publisher,
                'major': major,
                'ddc': ddc,
                'abstract': abstract,
                'location': location_str,
                'source': str(filepath),
                'csv_file': str(filepath),
                'text': text,
            }
            documents.append(document)

        return documents


class HybridSearchEngine:
    """Wrapper combining BM25 + regex + filtering"""

    def __init__(self, bm25_engine):
        self.bm25 = bm25_engine

    def search_by_query(self, query, top_k=6):
        return self.bm25.search(query, top_k=top_k, use_regex=True)

    def search_by_author(self, author, query, top_k=5):
        results = self.bm25.search(query, top_k=top_k*2)
        author_results = [r for r in results if author.lower() in r['doc'].get('author', '').lower()]
        return author_results[:top_k]

    def search_by_subject(self, subject, query, top_k=5):
        results = self.bm25.search(query, top_k=top_k*2)
        subject_results = [r for r in results if subject.lower() in r['doc'].get('subject', '').lower()]
        return subject_results[:top_k]

    def get_related_documents(self, title, author=None, top_k=5):
        query = f"{title} {author if author else ''}"
        results = self.bm25.search(query, top_k=top_k*3)
        related = [r for r in results if r['doc']['title'].lower() != title.lower()]
        return related[:top_k]

    def search_with_filters(self, query, filters=None, top_k=6):
        results = self.bm25.search(query, top_k=top_k*2)

        if filters:
            for result in results[:]:
                doc = result['doc']

                if 'author' in filters and filters['author'].lower() not in doc.get('author', '').lower():
                    results.remove(result)
                    continue

                if 'subject' in filters and filters['subject'].lower() not in doc.get('subject', '').lower():
                    results.remove(result)
                    continue

                if 'year' in filters:
                    doc_year = str(doc.get('year', ''))
                    if str(filters['year']) not in doc_year:
                        results.remove(result)
                        continue

        return results[:top_k]
