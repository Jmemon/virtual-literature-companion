"""
Processing modules for the Virtual Literature Companion system.

This package contains individual processors that handle different aspects
of book processing:

- pdf2txt: PDF to text conversion with OCR support
- page_cleaning: Text cleaning using local language models
- parse_novel_text: Text parsing and literary analysis
- create_vector_indexes: Vector embedding creation for search

Each processor is designed to be modular and can be used independently
or as part of the complete ingestion pipeline.
"""

from .pdf_extraction import extract_page_text, validate_pdf_file
from .page_cleaning import clean_pages_async
from .process_novel_text import process_extracted_pages
from .novel_text_to_json import process_chapters_to_structured

__all__ = [
    'extract_page_text',
    'clean_pages_async', 
    'validate_pdf_file',
    'process_extracted_pages',
    'process_chapters_to_structured'
] 