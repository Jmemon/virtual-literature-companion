"""
Virtual Literature Companion - Advanced PDF book processing and analysis system.

This package provides comprehensive tools for converting PDF books into structured,
searchable, and analyzable formats including:

- Multi-threaded PDF text extraction with OCR fallback
- Intelligent chapter detection and text structuring
- AI-powered literary analysis (characters, settings, dialogue)
- Vector embeddings for semantic search
- Comprehensive metadata generation
- Command-line interface for easy use

Main components:
- ingest: Complete book processing pipeline
- processors: Individual processing modules
- cli: Command-line interface
- constants: Configuration and paths

Example usage:
    from virtual_literature_companion import ingest_book_pdf
    
    result = ingest_book_pdf(
        pdf_path="book.pdf",
        novel_name="Pride and Prejudice",
        author_name="Jane Austen"
    )
"""

__version__ = "1.0.0"
__author__ = "Virtual Literature Companion Team"
__email__ = "contact@vlc.com"

# Import main functions for easy access
from .ingest import ingest_book_pdf, list_ingested_books
from .constants import BOOKS_DIR, SRC_DIR, REPO_DIR

# Export key functions and constants
__all__ = [
    'ingest_book_pdf',
    'list_ingested_books',
    'BOOKS_DIR',
    'SRC_DIR',
    'REPO_DIR',
    '__version__'
] 