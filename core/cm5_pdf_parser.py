"""
PDF Parser Module
Extracts text and metadata from PDF documents using PyMuPDF (if available) or PyPDF2 (fallback)
"""

from typing import Dict, List
import re

# Try to import PyMuPDF, fallback to PyPDF2
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("Either PyMuPDF or PyPDF2 must be installed for PDF processing")


class CM5PDFParser:
    """Handles PDF text extraction and parsing"""

    def extract_text(self, filepath: str) -> str:
        """
        Extract all text from PDF document

        Args:
            filepath: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            if HAS_PYMUPDF:
                return self._extract_text_pymupdf(filepath)
            else:
                return self._extract_text_pypdf2(filepath)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def _extract_text_pymupdf(self, filepath: str) -> str:
        """Extract text using PyMuPDF"""
        doc = fitz.open(filepath)
        text_content = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            text_content.append(f"\n--- PAGE {page_num} ---\n")
            text_content.append(text)
        doc.close()
        return ''.join(text_content)

    def _extract_text_pypdf2(self, filepath: str) -> str:
        """Extract text using PyPDF2"""
        reader = PdfReader(filepath)
        text_content = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            text_content.append(f"\n--- PAGE {page_num} ---\n")
            text_content.append(text)
        return ''.join(text_content)

    def extract_text_by_page(self, filepath: str) -> List[Dict[str, any]]:
        """
        Extract text page by page with metadata

        Args:
            filepath: Path to PDF file

        Returns:
            List of dictionaries with page number and text
        """
        try:
            if HAS_PYMUPDF:
                return self._extract_text_by_page_pymupdf(filepath)
            else:
                return self._extract_text_by_page_pypdf2(filepath)
        except Exception as e:
            raise Exception(f"Error extracting text by page: {str(e)}")

    def _extract_text_by_page_pymupdf(self, filepath: str) -> List[Dict[str, any]]:
        """Extract text by page using PyMuPDF"""
        doc = fitz.open(filepath)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append({
                'page_number': page_num,
                'text': text,
                'char_count': len(text)
            })
        doc.close()
        return pages

    def _extract_text_by_page_pypdf2(self, filepath: str) -> List[Dict[str, any]]:
        """Extract text by page using PyPDF2"""
        reader = PdfReader(filepath)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            pages.append({
                'page_number': page_num,
                'text': text,
                'char_count': len(text)
            })
        return pages

    def get_page_count(self, filepath: str) -> int:
        """
        Get total number of pages in PDF

        Args:
            filepath: Path to PDF file

        Returns:
            Number of pages
        """
        try:
            if HAS_PYMUPDF:
                doc = fitz.open(filepath)
                page_count = len(doc)
                doc.close()
                return page_count
            else:
                reader = PdfReader(filepath)
                return len(reader.pages)
        except Exception as e:
            raise Exception(f"Error getting page count: {str(e)}")

    def get_document_metadata(self, filepath: str) -> Dict:
        """
        Extract document metadata

        Args:
            filepath: Path to PDF file

        Returns:
            Dictionary with metadata
        """
        try:
            if HAS_PYMUPDF:
                doc = fitz.open(filepath)
                metadata = doc.metadata
                metadata['page_count'] = len(doc)
                doc.close()
                return metadata
            else:
                reader = PdfReader(filepath)
                metadata = reader.metadata or {}
                metadata['page_count'] = len(reader.pages)
                # Convert metadata keys to strings (PyPDF2 returns them as objects)
                return {str(k): str(v) if v else '' for k, v in metadata.items()}
        except Exception as e:
            raise Exception(f"Error extracting metadata: {str(e)}")

    def search_text(self, filepath: str, search_term: str) -> List[Dict]:
        """
        Search for specific text in PDF and return page references

        Args:
            filepath: Path to PDF file
            search_term: Text to search for

        Returns:
            List of matches with page numbers
        """
        try:
            if HAS_PYMUPDF:
                return self._search_text_pymupdf(filepath, search_term)
            else:
                return self._search_text_pypdf2(filepath, search_term)
        except Exception as e:
            raise Exception(f"Error searching text: {str(e)}")

    def _search_text_pymupdf(self, filepath: str, search_term: str) -> List[Dict]:
        """Search text using PyMuPDF"""
        doc = fitz.open(filepath)
        results = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if search_term.lower() in text.lower():
                for match in re.finditer(re.escape(search_term), text, re.IGNORECASE):
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    results.append({
                        'page': page_num,
                        'context': context.strip()
                    })
        doc.close()
        return results

    def _search_text_pypdf2(self, filepath: str, search_term: str) -> List[Dict]:
        """Search text using PyPDF2"""
        reader = PdfReader(filepath)
        results = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if search_term.lower() in text.lower():
                for match in re.finditer(re.escape(search_term), text, re.IGNORECASE):
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    results.append({
                        'page': page_num,
                        'context': context.strip()
                    })
        return results

    def extract_tables(self, filepath: str) -> List[Dict]:
        """
        Extract tables from PDF (basic implementation)

        Args:
            filepath: Path to PDF file

        Returns:
            List of detected table regions
        """
        try:
            if HAS_PYMUPDF:
                return self._extract_tables_pymupdf(filepath)
            else:
                # PyPDF2 doesn't support table extraction, return empty list
                return []
        except Exception as e:
            raise Exception(f"Error extracting tables: {str(e)}")

    def _extract_tables_pymupdf(self, filepath: str) -> List[Dict]:
        """Extract tables using PyMuPDF"""
        doc = fitz.open(filepath)
        tables = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("blocks")
            for block in text:
                block_text = block[4]
                if self._is_table_like(block_text):
                    tables.append({
                        'page': page_num,
                        'content': block_text,
                        'bbox': block[:4]
                    })
        doc.close()
        return tables

    def _is_table_like(self, text: str) -> bool:
        """
        Heuristic to detect if text block is table-like

        Args:
            text: Text block to analyze

        Returns:
            True if appears to be a table
        """
        lines = text.split('\n')
        if len(lines) < 3:
            return False

        # Check for multiple columns (tabs or multiple spaces)
        tab_count = sum(1 for line in lines if '\t' in line or '  ' in line)
        return tab_count >= len(lines) * 0.5
