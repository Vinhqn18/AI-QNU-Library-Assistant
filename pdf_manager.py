"""
pdf_manager.py — Quản lý PDF: metadata, chapters, preview
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class PDFManager:
    """Quản lý metadata PDF, cấu trúc chương, và preview"""
    
    def __init__(self, pdf_dir: str = "pdfs", cache_file: str = "pdf_cache.json"):
        self.pdf_dir = Path(pdf_dir)
        self.cache_file = cache_file
        self.pdf_cache = self._load_cache()
        
        # Đảm bảo thư mục pdfs tồn tại
        self.pdf_dir.mkdir(exist_ok=True)
    
    def list_all_pdfs(self) -> List[Dict]:
        """Liệt kê tất cả PDF với metadata"""
        pdfs = []
        
        if not self.pdf_dir.exists():
            return []
        
        for i, pdf_file in enumerate(sorted(self.pdf_dir.glob("*.pdf"))):
            try:
                metadata = self._extract_metadata(pdf_file)
                file_stem = pdf_file.stem.strip()
                file_name = pdf_file.name
                display_name = self._resolve_display_name(metadata, file_stem)
                pdfs.append({
                    "id": f"pdf_{i:03d}",
                    "title": display_name,
                    "display_name": display_name,
                    "original_title": metadata.get('title', file_stem),
                    "author": metadata.get('author', 'Unknown'),
                    "year": metadata.get('year', 'N/A'),
                    "pages": metadata.get('pages', 0),
                    "subject": metadata.get('subject', 'PDF'),
                    "file_path": str(pdf_file),
                    "file_name": file_name,
                    "file_stem": file_stem,
                    "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
                    "added_date": datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat(),
                    "thumbnail_url": f"/api/pdfs/pdf_{i:03d}/thumbnail"
                })
            except Exception as e:
                print(f"Lỗi xử lý PDF {pdf_file.name}: {e}")
                continue
        
        return pdfs
    
    def extract_chapters(self, pdf_path: str) -> List[Dict]:
        """
        Trích xuất cấu trúc chương từ PDF
        Chiến lược: 
        1. Cố gắng lấy outline (mục lục)
        2. Nếu không có, chia PDF thành các phần bằng heuristic
        """
        if not PdfReader:
            print("pypdf not available, returning simple chapters")
            return self._generate_simple_chapters(pdf_path)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                chapters = []
                
                # Phương pháp 1: Sử dụng outline nếu có, nhưng chỉ lấy entry hợp lệ
                outline_items = []
                if reader.outline and len(reader.outline) > 0:
                    outline_items = self._flatten_outline(reader.outline)

                valid_outline_titles = [
                    self._sanitize_outline_title(item.get("title", ""))
                    for item in outline_items
                    if self._sanitize_outline_title(item.get("title", ""))
                ]

                if len(valid_outline_titles) >= 2:
                    total_pages = len(reader.pages)
                    pages_per_chapter = max(8, total_pages // max(2, len(valid_outline_titles)))
                    for i, title in enumerate(valid_outline_titles):
                        start_page = min(i * pages_per_chapter, max(total_pages - 1, 0))
                        end_page = min(start_page + pages_per_chapter, total_pages)
                        if end_page <= start_page:
                            end_page = min(start_page + 1, total_pages)

                        chapters.append({
                            "chapter": i + 1,
                            "title": title[:80],
                            "pages": f"{start_page + 1}-{end_page}",
                            "start_page": start_page,
                            "end_page": end_page
                        })

                # Nếu outline chỉ có 1 mục hoặc quá nông, fallback sang heuristic để chia rõ hơn
                if len(chapters) < 2:
                    chapters = self._generate_simple_chapters(pdf_path)
                
                # Phương pháp 2: Tạo chapters theo heuristic nếu không có outline
                if not chapters:
                    chapters = self._generate_simple_chapters(pdf_path)
                
                return chapters
        
        except Exception as e:
            print(f"Lỗi trích xuất chapters: {e}")
            return self._generate_simple_chapters(pdf_path)

    def _flatten_outline(self, outline) -> List[Dict]:
        """Chuyển outline lồng nhau thành danh sách phẳng."""
        flattened = []

        def walk(items):
            for item in items:
                if isinstance(item, list):
                    walk(item)
                    continue

                title = ""
                page_number = None

                try:
                    if hasattr(item, "title"):
                        title = str(item.title)
                    elif isinstance(item, dict):
                        title = str(item.get("/Title") or item.get("title") or "")
                    else:
                        title = str(item)

                    if not title or self._looks_like_bad_outline_title(title):
                        continue

                    if hasattr(item, "page"):
                        page_number = self._safe_page_number(item.page)
                    elif isinstance(item, dict) and item.get("/Page") is not None:
                        page_number = self._safe_page_number(item.get("/Page"))
                except Exception:
                    continue

                flattened.append({"title": title, "page_number": page_number})

        walk(outline)
        return flattened

    def _safe_page_number(self, page_ref) -> Optional[int]:
        """Resolve page reference to zero-based page number if possible."""
        try:
            if not PdfReader:
                return None
            return None
        except Exception:
            return None

    def _looks_like_bad_outline_title(self, title: str) -> bool:
        normalized = str(title).strip().lower()
        if not normalized:
            return True
        bad_markers = [
            "indirectobject",
            "{'title':",
            '[{\'title\':',
            "page': indirectobject",
            "page\": indirectobject",
            "<class '",
        ]
        return any(marker in normalized for marker in bad_markers)

    def _sanitize_outline_title(self, title: str) -> str:
        """Chuẩn hóa title outline; loại bỏ entry rác."""
        if title is None:
            return ""

        cleaned = str(title).strip()
        if self._looks_like_bad_outline_title(cleaned):
            return ""

        cleaned = cleaned.replace("\n", " ").replace("\r", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned
    
    def _generate_simple_chapters(self, pdf_path: str) -> List[Dict]:
        """Tạo chapters đơn giản dựa trên số trang"""
        if not PdfReader:
            # Nếu không có pypdf, trả về 5 chapters chuẩn
            return [
                {
                    "chapter": i + 1,
                    "title": f"Phần {i + 1}",
                    "pages": f"1-20",
                    "start_page": 0,
                    "end_page": 20
                }
                for i in range(5)
            ]
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)
                if total_pages <= 12:
                    pages_per_chapter = max(4, total_pages // 3)
                else:
                    pages_per_chapter = max(10, total_pages // 8)  # chia thành nhiều phần hơn
                
                chapters = []
                for i in range(0, total_pages, pages_per_chapter):
                    chapter_num = (i // pages_per_chapter) + 1
                    chapter = {
                        "chapter": chapter_num,
                        "title": f"Phần {chapter_num}",
                        "pages": f"{i+1}-{min(i+pages_per_chapter, total_pages)}",
                        "start_page": i,
                        "end_page": min(i + pages_per_chapter, total_pages)
                    }
                    chapters.append(chapter)
                
                return chapters
        except:
            return []
    
    def get_chapter_text(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """Trích xuất văn bản từ một chương cụ thể"""
        if not PdfReader:
            return "[pypdf not available - cannot extract text]"
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                text = ""
                
                for page_num in range(max(0, start_page), min(end_page, len(reader.pages))):
                    try:
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    except:
                        text += f"[Trang {page_num + 1}: Không thể trích xuất]\n"
                
                return text if text.strip() else "[Không có nội dung]"
        
        except Exception as e:
            print(f"Lỗi trích xuất text từ PDF: {e}")
            return f"[Lỗi: {str(e)}]"
    
    def _extract_metadata(self, pdf_path: Path) -> Dict:
        """Trích xuất metadata từ PDF"""
        if not PdfReader:
            return {
                'title': pdf_path.stem,
                'author': 'Unknown',
                'pages': 0,
                'subject': 'PDF',
                'year': 'N/A'
            }
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                info = reader.metadata
                
                return {
                    'title': info.get('/Title', pdf_path.stem) if info else pdf_path.stem,
                    'author': info.get('/Author', 'Unknown') if info else 'Unknown',
                    'pages': len(reader.pages),
                    'subject': info.get('/Subject', 'PDF') if info else 'PDF',
                    'year': self._extract_year_from_metadata(info) if info else 'N/A'
                }
        except Exception as e:
            print(f"Lỗi trích xuất metadata: {e}")
            return {
                'title': pdf_path.stem,
                'author': 'Unknown',
                'pages': 0,
                'subject': 'PDF',
                'year': 'N/A'
            }

    def _resolve_display_name(self, metadata: Dict, file_stem: str) -> str:
        """Ưu tiên tên file thay vì title metadata khi metadata là tên chung/generic."""
        title = str(metadata.get('title', '') or '').strip()
        normalized = title.lower()
        generic_titles = {
            'aspose',
            'document',
            'untitled',
            'unknown',
            'pdf'
        }

        if not title or normalized in generic_titles:
            return file_stem

        if len(title) <= 3:
            return file_stem

        return title
    
    def _extract_year_from_metadata(self, info) -> str:
        """Cố gắng trích xuất năm từ creation date"""
        if not info:
            return 'N/A'
        
        try:
            creation_date = info.get('/CreationDate', '')
            if creation_date:
                # Format: D:YYYYMMDDHHmmSS
                year_str = str(creation_date)
                # Tìm 4 chữ số liên tiếp
                import re
                match = re.search(r'(\d{4})', year_str)
                if match:
                    year = match.group(1)
                    if 1900 <= int(year) <= 2100:
                        return year
        except:
            pass
        
        return 'N/A'
    
    def _load_cache(self) -> Dict:
        """Tải cached PDF metadata"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Lưu PDF metadata cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi lưu cache: {e}")
