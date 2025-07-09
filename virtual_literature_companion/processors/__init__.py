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

from .pdf2txt import process_book_pdf, validate_pdf_file
from .parse_novel_text import process_novel_to_structured_json
from .create_vector_indexes import create_vector_indexes

__all__ = [
    'process_book_pdf',
    'validate_pdf_file',
    'process_novel_to_structured_json',
    'create_vector_indexes'
] 