"""
EPUB to text extraction and ingestion for the Virtual Literature Companion system.

This module handles the complete EPUB ingestion pipeline:
1. Validate epub file.
2. Extract sections from epub file. Epubs are already broken into sections, so this should be straightforward.
3. Classify sections into section types from types.SectionType.
4. Text cleanup. In this case basically amounts to removing headings and some formatting.
5. Save to json as list of dicts containing keys: section_type, raw_text, clean_text.

The main function returns extracted and cleaned section texts for ingestion into the system.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

from ..constants import DEBUG_MODE
from ..types import SectionType
from ..llm.clean_text import clean_text

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


def epub_extract(epub_path: str) -> List[Dict]:
    """
    Extract sections from an EPUB file with proper metadata and structure.
    
    Args:
        epub_path (str): Path to the EPUB file
        
    Returns:
        List[Dict]: List of section dictionaries containing:
            - id: Section identifier
            - title: Section title (if available)
            - type: Section type (chapter, toc, acknowledgements, etc.)
            - content: Extracted text content
            - spine_index: Position in reading order (if applicable)
    """
    logger.info(f"Starting structured EPUB extraction for {epub_path}")
    
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")
    
    # Read the EPUB file
    book = epub.read_epub(str(epub_path))
    sections = []
    
    # Get spine order for proper sequencing
    spine_items = [item[0] for item in book.spine]
    
    # Extract navigation document (TOC)
    nav_doc = book.get_item_with_id('nav') or book.get_item_with_id('ncx')
    if nav_doc:
        nav_content = nav_doc.content.decode('utf-8')
        nav_text = extract_single_chapter_text(nav_content, nav_doc.get_id())[1]
        sections.append({
            'id': nav_doc.get_id(),
            'title': 'Table of Contents',
            'type': 'toc',
            'content': nav_text,
            'spine_index': None
        })
    
    # Process all document items
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.content.decode('utf-8')
            text = extract_single_chapter_text(content, item.get_id())[1]
            
            # Determine section type and title
            title = _extract_section_title(content, item)
            section_type = _classify_section(content, title)
            
            # Get spine position
            spine_index = None
            if item.get_id() in spine_items:
                spine_index = spine_items.index(item.get_id())
            
            sections.append({
                'id': item.get_id(),
                'title': title,
                'type': section_type,
                'content': text,
                'spine_index': spine_index
            })
    
    # Sort by spine order for main content, keep special sections at appropriate positions
    sections.sort(key=lambda x: (x['spine_index'] is None, x['spine_index'] or 0))
    
    logger.info(f"Successfully extracted {len(sections)} sections from EPUB")
    return sections


def _extract_section_title(content: str, item) -> str:
    """
    Extract title from EPUB content.
    Args:
        content (str): XHTML content of the section
        item: EPUB item object
    Returns:
        str: Extracted title
    """
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract title from various sources
    title = ""
    
    # Try to get title from heading tags
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        heading = soup.find(tag)
        if heading:
            title = heading.get_text().strip()
            break
    
    # Fallback to title tag
    if not title:
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
    
    # Fallback to item ID
    if not title:
        title = item.get_id().replace('_', ' ').replace('-', ' ').title()
        
    return title


def _classify_section(content: str, title: str) -> SectionType:
    """
    Determine the section type from EPUB content.
    Args:
        content (str): XHTML content of the section
        title (str): The title of the section
    Returns:
        SectionType: The classified section type
    """
    soup = BeautifulSoup(content, 'html.parser')
    
    # Determine section type based on epub:type attributes
    epub_type_elem = soup.find(attrs={"epub:type": True})
    if epub_type_elem:
        epub_type = epub_type_elem.get("epub:type", "").lower()
        if epub_type == 'toc':
            return SectionType.TABLE_OF_CONTENTS
        elif epub_type == 'acknowledgments':
            return SectionType.ACKNOWLEDGEMENTS
        elif epub_type == 'bibliography':
            return SectionType.BIBLIOGRAPHY
        elif epub_type == 'glossary':
            return SectionType.GLOSSARY
        elif epub_type == 'index':
            return SectionType.INDEX
        elif epub_type == 'titlepage':
            return SectionType.TITLE_PAGE
            
    # Determine section type based on title patterns
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['table of contents', 'contents']):
        return SectionType.TABLE_OF_CONTENTS
    elif any(word in title_lower for word in ['acknowledgment', 'acknowledgement', 'thanks']):
        return SectionType.ACKNOWLEDGEMENTS
    elif any(word in title_lower for word in ['bibliography', 'references', 'works cited']):
        return SectionType.BIBLIOGRAPHY
    elif any(word in title_lower for word in ['glossary', 'terms', 'definitions']):
        return SectionType.GLOSSARY
    elif any(word in title_lower for word in ['index']):
        return SectionType.INDEX
    elif any(word in title_lower for word in ['title page', 'cover']):
        return SectionType.TITLE_PAGE
    elif any(word in title_lower for word in ['preface', 'foreword', 'introduction']):
        return SectionType.PREFACE
    elif any(word in title_lower for word in ['epilogue', 'afterword', 'conclusion']):
        return SectionType.EPILOGUE
    elif any(word in title_lower for word in ['appendix']):
        return SectionType.APPENDIX
    elif any(word in title_lower for word in ['copyright', 'copyright page', 'copyright notice']):
        return SectionType.COPYRIGHT_PAGE
    elif any(word in title_lower for word in ['dedication', 'dedication page']):
        return SectionType.DEDICATION
    elif any(word in title_lower for word in ['chapter', 'chapters', 'part', 'parts']):
        return SectionType.CHAPTER
        
    return SectionType.CHAPTER # Default


def validate_epub(epub_path: str) -> bool:
    """
    Validate that an EPUB file is readable (EPUB 1.0+ compatible).
    
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


def load_ingest_epub(epub_path: Path) -> List[Dict]:
    """
    Ingest an EPUB file by validating, extracting sections, and cleaning text.
    
    Args:
        epub_path (Path): Path to the EPUB file
        
    Returns:
        List[Dict]: List of dictionaries with keys:
            - section_type: The type of section (SectionType enum value)
            - raw_text: The original extracted text
            - clean_text: The cleaned text after LLM processing
            
    Raises:
        FileNotFoundError: If the EPUB file doesn't exist
        ValueError: If the EPUB file is invalid
    """
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")
    
    # Step 1: Validate EPUB file
    logger.info(f"Validating EPUB file: {epub_path}")
    if not validate_epub(str(epub_path)):
        raise ValueError(f"Invalid EPUB file: {epub_path}")
    
    # Step 2 & 3: Extract sections and classify them
    logger.info(f"Extracting sections from EPUB: {epub_path}")
    sections = epub_extract(str(epub_path))
    
    # Step 4 & 5: Clean text and format output
    logger.info(f"Processing {len(sections)} sections for text cleanup")
    result = []
    
    for section in sections:
        raw_text = section['content']
        section_type = section['type']
        
        # Clean the text using LLM
        try:
            clean_text_result = clean_text(raw_text)
            if clean_text_result is None:
                logger.warning(f"Failed to clean text for section {section['id']}, using raw text")
                clean_text_result = raw_text
        except Exception as e:
            logger.error(f"Error cleaning text for section {section['id']}: {e}")
            clean_text_result = raw_text
        
        result.append({
            'section_type': section_type,
            'raw_text': raw_text,
            'clean_text': clean_text_result
        })
    
    logger.info(f"Successfully processed {len(result)} sections from EPUB")
    return result