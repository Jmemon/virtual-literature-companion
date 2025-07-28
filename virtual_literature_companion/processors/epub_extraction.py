"""
EPUB to text extraction processor for the Virtual Literature Companion system.

This module handles the conversion of EPUB files to text, supporting:
- Standard EPUB format (EPUB 2 and 3)
- Text extraction from XHTML chapters
- HTML content parsing and cleaning
- Multi-threaded chapter processing
- Text cleaning using local language model

The main function returns extracted and cleaned page texts similar to PDF extraction.
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, List
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

from ..constants import MAX_WORKERS, DEBUG_MODE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_single_chapter_text(chapter_content: str, chapter_id: str) -> Tuple[str, str]:
    """
    Extract text from a single EPUB chapter (XHTML content).
    
    Args:
        chapter_content (str): Raw XHTML content of the chapter
        chapter_id (str): Identifier for the chapter
        
    Returns:
        Tuple[str, str]: Chapter ID and extracted text
    """
    try:
        # Parse HTML content with BeautifulSoup
        soup = BeautifulSoup(chapter_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if DEBUG_MODE:
            logger.debug(f"Chapter {chapter_id}: Extracted {len(text)} characters")
        
        return chapter_id, text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from chapter {chapter_id}: {str(e)}")
        return chapter_id, ""


def validate_epub_file(epub_path: str) -> bool:
    """
    Validate that an EPUB file can be processed.
    
    Args:
        epub_path (str): Path to the EPUB file
        
    Returns:
        bool: True if the EPUB can be processed, False otherwise
    """
    try:
        # Check if file is a valid ZIP archive (EPUB is a ZIP file)
        with zipfile.ZipFile(epub_path, 'r') as zip_file:
            # Check for required EPUB files
            file_list = zip_file.namelist()
            
            # EPUB must contain META-INF/container.xml
            if 'META-INF/container.xml' not in file_list:
                logger.error("Missing META-INF/container.xml")
                return False
            
            # Try to parse the container.xml to find the OPF file
            container_content = zip_file.read('META-INF/container.xml')
            container_root = ET.fromstring(container_content)
            
            # Find the OPF file path
            rootfile = container_root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
            if rootfile is None:
                logger.error("No rootfile found in container.xml")
                return False
            
            opf_path = rootfile.get('full-path')
            if not opf_path or opf_path not in file_list:
                logger.error(f"OPF file not found: {opf_path}")
                return False
        
        # Try to open with ebooklib as additional validation
        book = epub.read_epub(epub_path)
        items = list(book.get_items())
        
        if not items:
            logger.error("No content items found in EPUB")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"EPUB validation failed: {str(e)}")
        return False


def extract_page_text(epub_path: str) -> Tuple[Dict[int, str], int]:
    """
    Extract raw text from all chapters in the EPUB without cleaning.
    
    Args:
        epub_path (str): Path to the EPUB file
        
    Returns:
        Tuple[Dict[int, str], int]: (page_texts, total_pages)
    """
    logger.info(f"Starting EPUB extraction for {epub_path}")
    
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")
    
    # Read the EPUB file
    book = epub.read_epub(str(epub_path))
    
    # Get all document items (chapters)
    chapters = []
    chapter_contents = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item)
            chapter_contents.append((item.content.decode('utf-8'), item.get_id()))
    
    total_pages = len(chapters)
    logger.info(f"Processing {total_pages} chapters with {MAX_WORKERS} workers")
    
    page_texts = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(extract_single_chapter_text, content, chapter_id): idx
            for idx, (content, chapter_id) in enumerate(chapter_contents)
        }
        
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            chapter_id, text = future.result()
            page_texts[page_num] = text
            
            if DEBUG_MODE:
                logger.debug(f"Extracted text for chapter {page_num + 1}/{total_pages} ({chapter_id})")
    
    total_chars = sum(len(text) for text in page_texts.values())
    logger.info(f"Successfully extracted {total_pages} chapters, {total_chars} total characters")
    
    return page_texts, total_pages