"""
PDF to text conversion processor for the Virtual Literature Companion system.

This module handles the conversion of PDF files to text, supporting:
- Text-based PDFs (direct text extraction)
- Scanned image PDFs (OCR processing)
- Mixed PDFs (text + images)
- Multi-threaded page processing
- Chapter detection and splitting

The main function `process_book_pdf` returns a list of chapter texts that can be
further processed by other components of the system.
"""

import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import uuid

import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for image extraction

from ..constants import MAX_WORKERS, TESSERACT_CONFIG, DEBUG_MODE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_page(pdf_path: str, page_num: int) -> Tuple[int, str]:
    """
    Extract text from a single PDF page, handling both text-based and image-based content.
    
    This function attempts text extraction first, then falls back to OCR if minimal
    text is found. It handles mixed content by combining both extraction methods.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number to process (0-indexed)
        
    Returns:
        Tuple[int, str]: Page number and extracted text
    """
    try:
        # First attempt: Direct text extraction using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return page_num, ""
            
            page = pdf.pages[page_num]
            text = page.extract_text() or ""
            
            # If we got substantial text, use it
            if len(text.strip()) > 50:
                if DEBUG_MODE:
                    logger.debug(f"Page {page_num + 1}: Extracted {len(text)} chars via text extraction")
                return page_num, text.strip()
        
        # Second attempt: OCR processing for scanned pages
        if DEBUG_MODE:
            logger.debug(f"Page {page_num + 1}: Falling back to OCR (text extraction yielded {len(text)} chars)")
        
        # Convert page to image for OCR
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num]
        
        # Render page as image with high resolution
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        
        # Process with OCR
        image = Image.open(io.BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
        
        pdf_document.close()
        
        # Combine text and OCR results if both have content
        combined_text = text + "\n" + ocr_text if text.strip() else ocr_text
        
        if DEBUG_MODE:
            logger.debug(f"Page {page_num + 1}: OCR extracted {len(ocr_text)} chars")
        
        return page_num, combined_text.strip()
        
    except Exception as e:
        logger.error(f"Error processing page {page_num + 1}: {str(e)}")
        return page_num, ""


def detect_chapter_boundaries(text: str) -> List[Tuple[str, int]]:
    """
    Detect chapter boundaries in the text using common chapter patterns.
    
    This function looks for various chapter heading patterns commonly found in
    literature, including numbered chapters, titled chapters, and part divisions.
    
    Args:
        text (str): Full text content to analyze
        
    Returns:
        List[Tuple[str, int]]: List of (chapter_title, start_position) tuples
    """
    chapter_patterns = [
        # "Chapter 1", "Chapter One", etc.
        r'(?i)^chapter\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?:\s*[:.\-]\s*(.*))?$',
        
        # "1. Chapter Title", "I. Chapter Title", etc.
        r'^(?:\d+|[IVX]+)\.?\s+(.{1,100})$',
        
        # "PART ONE", "BOOK I", etc.
        r'(?i)^(?:part|book|section)\s+(?:\d+|one|two|three|four|five|[ivx]+)(?:\s*[:.\-]\s*(.*))?$',
        
        # Standalone chapter titles (all caps or title case)
        r'^[A-Z][A-Z\s]{10,100}$',
        
        # Numbered sections like "1" or "I"
        r'^(?:\d+|[IVX]+)$'
    ]
    
    chapters = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        for pattern in chapter_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # Calculate character position
                char_pos = sum(len(lines[j]) + 1 for j in range(i))
                
                # Extract chapter title
                if match.groups() and match.group(1):
                    chapter_title = match.group(1).strip()
                else:
                    chapter_title = line.strip()
                
                chapters.append((chapter_title, char_pos))
                
                if DEBUG_MODE:
                    logger.debug(f"Found chapter: '{chapter_title}' at position {char_pos}")
                break
    
    return chapters


def split_into_chapters(text: str, chapters: List[Tuple[str, int]]) -> List[str]:
    """
    Split the full text into individual chapter texts.
    
    Args:
        text (str): Full text content
        chapters (List[Tuple[str, int]]): List of (chapter_title, start_position) tuples
        
    Returns:
        List[str]: List of chapter texts
    """
    if not chapters:
        # If no chapters detected, return entire text as single chapter
        return [text]
    
    chapter_texts = []
    
    for i, (title, start_pos) in enumerate(chapters):
        # Determine end position (start of next chapter or end of text)
        end_pos = chapters[i + 1][1] if i + 1 < len(chapters) else len(text)
        
        # Extract chapter text
        chapter_text = text[start_pos:end_pos].strip()
        
        if chapter_text:
            chapter_texts.append(chapter_text)
            
            if DEBUG_MODE:
                logger.debug(f"Chapter {i + 1}: '{title}' - {len(chapter_text)} characters")
    
    return chapter_texts


def process_book_pdf(pdf_path: str, novel_name: str) -> List[str]:
    """
    Process a PDF book and convert it to a list of chapter texts.
    
    This is the main function that orchestrates the entire PDF processing pipeline:
    1. Multi-threaded page-by-page text extraction
    2. Chapter boundary detection
    3. Chapter text splitting
    4. File output creation
    
    Args:
        pdf_path (str): Path to the PDF file to process
        novel_name (str): Name of the novel for organizing output files
        
    Returns:
        List[str]: List of chapter texts in order
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the PDF cannot be processed
    """
    logger.info(f"Starting PDF processing for '{novel_name}'")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory structure
    from ..constants import BOOKS_DIR
    book_dir = BOOKS_DIR / novel_name
    raw_chapters_dir = book_dir / "raw_chapters"
    raw_chapters_dir.mkdir(parents=True, exist_ok=True)
    
    # Get total number of pages
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
    
    logger.info(f"Processing {total_pages} pages with {MAX_WORKERS} workers")
    
    # Process pages in parallel
    page_texts = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all pages for processing
        future_to_page = {
            executor.submit(extract_text_from_page, str(pdf_path), page_num): page_num
            for page_num in range(total_pages)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num, text = future.result()
            page_texts[page_num] = text
            
            if DEBUG_MODE:
                logger.debug(f"Completed page {page_num + 1}/{total_pages}")
    
    # Combine all page texts in order
    full_text = ""
    for page_num in sorted(page_texts.keys()):
        full_text += page_texts[page_num] + "\n\n"
    
    logger.info(f"Extracted {len(full_text)} characters from {total_pages} pages")
    
    # Detect chapter boundaries
    chapters = detect_chapter_boundaries(full_text)
    logger.info(f"Detected {len(chapters)} chapters")
    
    # Split into chapters
    chapter_texts = split_into_chapters(full_text, chapters)
    
    # Write raw chapter files
    for i, chapter_text in enumerate(chapter_texts, 1):
        chapter_file = raw_chapters_dir / f"{i}.txt"
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        
        logger.info(f"Created raw chapter file: {chapter_file}")
    
    logger.info(f"Successfully processed '{novel_name}' into {len(chapter_texts)} chapters")
    
    return chapter_texts


def validate_pdf_file(pdf_path: str) -> bool:
    """
    Validate that a PDF file can be processed.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        bool: True if the PDF can be processed, False otherwise
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check if we can access pages
            if len(pdf.pages) == 0:
                return False
            
            # Try to extract text from first page
            first_page = pdf.pages[0]
            test_text = first_page.extract_text()
            
            return True
    except Exception as e:
        logger.error(f"PDF validation failed: {str(e)}")
        return False 