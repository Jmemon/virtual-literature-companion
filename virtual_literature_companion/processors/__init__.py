"""
Processing modules for the Virtual Literature Companion system.

This package contains individual processors that handle different aspects
of book processing:

- pdf2txt: PDF to text conversion with OCR support
- parse_novel_text: Text parsing and literary analysis
- create_vector_indexes: Vector embedding creation for search

Each processor is designed to be modular and can be used independently
or as part of the complete ingestion pipeline.
"""

from .pdf2txt import extract_and_clean_pages, validate_pdf_file
from .process_novel_text import process_extracted_pages
from .parse_novel_text import process_chapters_to_structured

__all__ = [
    'extract_and_clean_pages',
    'validate_pdf_file',
    'process_extracted_pages',
    'process_chapters_to_structured'
] 