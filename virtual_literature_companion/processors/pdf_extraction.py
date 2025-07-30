"""
PDF to text conversion processor for the Virtual Literature Companion system.

This module handles the conversion of PDF files to text, supporting:
- Text-based PDFs (direct text extraction)
- Scanned image PDFs (OCR processing)
- Mixed PDFs (text + images)
- Multi-threaded page processing
- Text cleaning using local language model

The main function returns extracted and cleaned page texts.
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for image extraction

from ..constants import MAX_WORKERS, TESSERACT_CONFIG, DEBUG_MODE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_single_page_text(pdf_path: str, page_num: int) -> Tuple[int, str]:
    """
    Extract text from a single PDF page without categorization.
    
    Attempts direct text extraction first, falls back to OCR if needed.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_num (int): Page number to process (0-indexed)
        
    Returns:
        Tuple[int, str]: Page number and extracted text
    """
    try:
        # First attempt: Direct text extraction using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
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
        logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
        return page_num, ""


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


def extract_page_text(pdf_path: str) -> Tuple[Dict[int, str], int]:
    """
    Extract raw text from all pages in the PDF without cleaning.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Tuple[Dict[int, str], int]: (page_texts, total_pages)
    """
    logger.info(f"Starting PDF extraction for {pdf_path}")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Get total number of pages
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
    
    logger.info(f"Processing {total_pages} pages with {MAX_WORKERS} workers")
    
    page_texts = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(extract_single_page_text, str(pdf_path), page_num): page_num
            for page_num in range(total_pages)
        }
        
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            _, text = future.result()  # extract_single_page_text returns (page_num, text)
            page_texts[page_num] = text
            
            if DEBUG_MODE:
                logger.debug(f"Extracted text for page {page_num + 1}/{total_pages}")
    
    total_chars = sum(len(text) for text in page_texts.values())
    logger.info(f"Successfully extracted {total_pages} pages, {total_chars} total characters")
    
    return page_texts, total_pages 