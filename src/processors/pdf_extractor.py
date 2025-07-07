"""
PDF extraction module for processing literature files.

This module handles:
- Text extraction from PDFs
- Table of contents detection and parsing
- Metadata extraction
- Chapter boundary detection
- Page-aware text chunking
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from datetime import datetime
from loguru import logger

from ..models.schemas import BookMetadata, TableOfContents, TextChunk
from ..config import settings


class PDFExtractor:
    """
    Comprehensive PDF extraction utility.
    
    This class provides methods to extract:
    - Full text with page boundaries preserved
    - Table of contents with chapter mappings
    - Book metadata (title, author, ISBN, etc.)
    - Structured chunks for processing
    """
    
    def __init__(self):
        self.toc_patterns = [
            # Standard chapter patterns
            r'^Chapter\s+(\d+)[:\s]+(.+)$',
            r'^CHAPTER\s+(\d+)[:\s]+(.+)$',
            r'^(\d+)\.\s+(.+)$',
            r'^Part\s+([IVXLCDM]+)[:\s]+(.+)$',
            r'^PART\s+([IVXLCDM]+)[:\s]+(.+)$',
            # Special sections
            r'^(Prologue|Epilogue|Introduction|Preface|Acknowledgments)[:\s]*(.*)$',
            r'^(PROLOGUE|EPILOGUE|INTRODUCTION|PREFACE|ACKNOWLEDGMENTS)[:\s]*(.*)$',
        ]
        
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract all information from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
            - metadata: BookMetadata object
            - table_of_contents: TableOfContents object
            - full_text: Complete text of the book
            - page_texts: List of text by page
            - chunks: List of TextChunk objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
            
        logger.info(f"Extracting PDF: {path}")
        
        # Extract using both libraries for comprehensive data
        pymupdf_data = self._extract_with_pymupdf(path)
        pdfplumber_data = self._extract_with_pdfplumber(path)
        
        # Combine and reconcile data
        metadata = self._extract_metadata(pymupdf_data, pdfplumber_data, path)
        toc = self._extract_table_of_contents(pymupdf_data, pdfplumber_data)
        full_text, page_texts = self._combine_text_extraction(pymupdf_data, pdfplumber_data)
        
        # Create chunks with chapter awareness
        chunks = self._create_chunks(full_text, page_texts, toc, metadata.book_id)
        
        return {
            "metadata": metadata,
            "table_of_contents": toc,
            "full_text": full_text,
            "page_texts": page_texts,
            "chunks": chunks
        }
        
    def _extract_with_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract data using PyMuPDF."""
        doc = fitz.open(file_path)
        
        # Get document metadata
        metadata = doc.metadata
        
        # Get table of contents
        toc = doc.get_toc()
        
        # Extract text page by page
        page_texts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            page_texts.append({
                "page_num": page_num + 1,
                "text": text,
                "char_count": len(text)
            })
            
        doc.close()
        
        return {
            "metadata": metadata,
            "toc": toc,
            "page_texts": page_texts,
            "page_count": len(page_texts)
        }
        
    def _extract_with_pdfplumber(self, file_path: Path) -> Dict[str, Any]:
        """Extract data using pdfplumber for better table handling."""
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}
            
            page_texts = []
            potential_toc_entries = []
            
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Look for TOC entries in first 20 pages
                if page_num < 20:
                    toc_entries = self._detect_toc_entries(text, page_num + 1)
                    potential_toc_entries.extend(toc_entries)
                
                page_texts.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
                
        return {
            "metadata": metadata,
            "page_texts": page_texts,
            "potential_toc": potential_toc_entries,
            "page_count": len(page_texts)
        }
        
    def _detect_toc_entries(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Detect potential table of contents entries in text."""
        entries = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check against TOC patterns
            for pattern in self.toc_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2:
                        chapter_id = match.group(1)
                        title = match.group(2).strip()
                    else:
                        chapter_id = None
                        title = match.group(1).strip()
                        
                    # Look for page number at end of line
                    page_match = re.search(r'\.{3,}\s*(\d+)$', line)
                    if page_match:
                        target_page = int(page_match.group(1))
                        title = title.replace(page_match.group(0), '').strip()
                    else:
                        target_page = None
                        
                    entries.append({
                        "chapter_id": chapter_id,
                        "title": title,
                        "source_page": page_num,
                        "target_page": target_page,
                        "full_line": line
                    })
                    break
                    
        return entries
        
    def _extract_metadata(self, pymupdf_data: Dict, pdfplumber_data: Dict, 
                         file_path: Path) -> BookMetadata:
        """Extract and combine metadata from both sources."""
        # Combine metadata from both sources
        metadata = {**pymupdf_data.get("metadata", {}), **pdfplumber_data.get("metadata", {})}
        
        # Clean and extract standard fields
        title = metadata.get("title", "") or metadata.get("Title", "") or file_path.stem
        author = metadata.get("author", "") or metadata.get("Author", "") or "Unknown"
        
        # Try to extract ISBN from metadata or early pages
        isbn = self._extract_isbn(metadata, pymupdf_data.get("page_texts", [])[:5])
        
        # Extract publication year
        pub_year = self._extract_publication_year(metadata)
        
        return BookMetadata(
            title=self._clean_title(title),
            author=self._clean_author(author),
            isbn=isbn,
            publication_year=pub_year,
            page_count=pymupdf_data.get("page_count", 0),
            file_path=str(file_path),
            upload_date=datetime.utcnow()
        )
        
    def _extract_isbn(self, metadata: Dict, early_pages: List[Dict]) -> Optional[str]:
        """Extract ISBN from metadata or early pages."""
        # Check metadata
        for key in ["isbn", "ISBN", "Subject"]:
            if key in metadata:
                isbn_match = re.search(r'978[\d-]+|[\d-]{10,}', str(metadata[key]))
                if isbn_match:
                    return isbn_match.group(0).replace('-', '')
                    
        # Check early pages
        for page_data in early_pages:
            text = page_data.get("text", "")
            isbn_match = re.search(r'ISBN[:\s]*(978[\d-]+|[\d-]{10,})', text, re.IGNORECASE)
            if isbn_match:
                return isbn_match.group(1).replace('-', '')
                
        return None
        
    def _extract_publication_year(self, metadata: Dict) -> Optional[int]:
        """Extract publication year from metadata."""
        # Check creation date
        for key in ["creationDate", "CreationDate", "ModDate"]:
            if key in metadata:
                date_str = str(metadata[key])
                year_match = re.search(r'(19|20)\d{2}', date_str)
                if year_match:
                    return int(year_match.group(0))
                    
        return None
        
    def _clean_title(self, title: str) -> str:
        """Clean and format book title."""
        # Remove file extensions
        title = re.sub(r'\.(pdf|epub|mobi)$', '', title, flags=re.IGNORECASE)
        # Remove underscores and extra spaces
        title = title.replace('_', ' ').strip()
        # Title case
        return title.title()
        
    def _clean_author(self, author: str) -> str:
        """Clean and format author name."""
        # Remove common prefixes
        author = re.sub(r'^by\s+', '', author, flags=re.IGNORECASE)
        author = author.strip()
        return author
        
    def _extract_table_of_contents(self, pymupdf_data: Dict, 
                                  pdfplumber_data: Dict) -> TableOfContents:
        """Extract and structure table of contents."""
        # Start with PyMuPDF TOC if available
        toc_entries = []
        
        if pymupdf_data.get("toc"):
            for level, title, page in pymupdf_data["toc"]:
                if level == 1:  # Main chapters
                    toc_entries.append({
                        "title": title,
                        "start_page": page,
                        "level": level
                    })
                    
        # Supplement with detected entries from pdfplumber
        if pdfplumber_data.get("potential_toc"):
            for entry in pdfplumber_data["potential_toc"]:
                # Check if not duplicate
                if not any(e["title"] == entry["title"] for e in toc_entries):
                    toc_entries.append({
                        "title": entry["title"],
                        "start_page": entry.get("target_page"),
                        "chapter_id": entry.get("chapter_id")
                    })
                    
        # Sort by page number
        toc_entries.sort(key=lambda x: x.get("start_page") or 999999)
        
        # Convert to structured format
        chapters = []
        has_prologue = False
        has_epilogue = False
        
        for i, entry in enumerate(toc_entries):
            title = entry["title"]
            
            # Check for special sections
            if re.match(r'^(prologue|preface|introduction)', title, re.IGNORECASE):
                has_prologue = True
            elif re.match(r'^(epilogue|afterword)', title, re.IGNORECASE):
                has_epilogue = True
                
            # Determine chapter number
            chapter_num = None
            if entry.get("chapter_id"):
                try:
                    chapter_num = int(entry["chapter_id"])
                except (ValueError, TypeError):
                    pass
                    
            # Determine end page
            end_page = None
            if i < len(toc_entries) - 1:
                next_entry = toc_entries[i + 1]
                if next_entry.get("start_page"):
                    end_page = next_entry["start_page"] - 1
                    
            chapters.append(TableOfContents.Chapter(
                chapter_number=chapter_num,
                title=title,
                start_page=entry.get("start_page"),
                end_page=end_page
            ))
            
        return TableOfContents(
            chapters=chapters,
            total_chapters=len([c for c in chapters if c.chapter_number is not None]),
            has_prologue=has_prologue,
            has_epilogue=has_epilogue
        )
        
    def _combine_text_extraction(self, pymupdf_data: Dict, 
                               pdfplumber_data: Dict) -> Tuple[str, List[Dict]]:
        """Combine text extraction from both sources."""
        # Use pdfplumber as primary (better formatting) with pymupdf as fallback
        page_texts = []
        full_text_parts = []
        
        for page_num in range(max(pymupdf_data["page_count"], 
                                  pdfplumber_data["page_count"])):
            # Get text from both sources
            plumber_page = next((p for p in pdfplumber_data["page_texts"] 
                               if p["page_num"] == page_num + 1), None)
            mupdf_page = next((p for p in pymupdf_data["page_texts"] 
                             if p["page_num"] == page_num + 1), None)
            
            # Choose best text
            if plumber_page and plumber_page["text"].strip():
                text = plumber_page["text"]
            elif mupdf_page and mupdf_page["text"].strip():
                text = mupdf_page["text"]
            else:
                text = ""
                
            page_texts.append({
                "page_num": page_num + 1,
                "text": text,
                "char_count": len(text)
            })
            
            if text:
                full_text_parts.append(f"[Page {page_num + 1}]\n{text}")
                
        full_text = "\n\n".join(full_text_parts)
        return full_text, page_texts
        
    def _create_chunks(self, full_text: str, page_texts: List[Dict], 
                      toc: TableOfContents, book_id: str) -> List[TextChunk]:
        """Create text chunks with chapter and page awareness."""
        chunks = []
        
        # Create chapter mapping
        chapter_pages = {}
        for chapter in toc.chapters:
            if chapter.start_page:
                chapter_pages[chapter.start_page] = chapter
                
        # Process text into chunks
        current_pos = 0
        chunk_id = 0
        
        for page_data in page_texts:
            page_num = page_data["page_num"]
            page_text = page_data["text"]
            
            if not page_text.strip():
                continue
                
            # Determine current chapter
            current_chapter = None
            for start_page, chapter in sorted(chapter_pages.items()):
                if page_num >= start_page:
                    if chapter.end_page is None or page_num <= chapter.end_page:
                        current_chapter = chapter
                        
            # Create chunks from page
            page_chunks = self._chunk_text(
                page_text, 
                settings.chunk_size,
                settings.chunk_overlap
            )
            
            for chunk_text in page_chunks:
                chunk = TextChunk(
                    book_id=book_id,
                    chapter_number=current_chapter.chapter_number if current_chapter else None,
                    page_number=page_num,
                    content=chunk_text,
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_text)
                )
                chunks.append(chunk)
                current_pos += len(chunk_text) - settings.chunk_overlap
                chunk_id += 1
                
        logger.info(f"Created {len(chunks)} chunks from {len(page_texts)} pages")
        return chunks
        
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
            
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_sentences = []
                
                for sent in reversed(current_chunk):
                    overlap_size += len(sent)
                    overlap_sentences.insert(0, sent)
                    if overlap_size >= overlap:
                        break
                        
                current_chunk = overlap_sentences
                current_size = overlap_size
                
            current_chunk.append(sentence)
            current_size += sentence_size
            
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with nltk
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Convenience function
def extract_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract all information from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extraction results dictionary
    """
    extractor = PDFExtractor()
    return extractor.extract_from_file(file_path)