"""
Virtual Literature Companion - Advanced book processing and analysis system.

This package provides comprehensive tools for converting books into structured,
searchable, and analyzable formats including:

- EPUB text extraction and section classification
- AI-powered text cleaning and formatting
- Section type detection (chapters, TOC, etc.)
- Extensible architecture for multiple file formats
- Command-line interface for easy use

Main components:
- ingest: Complete book processing pipeline
- processors: Individual processing modules
- cli: Command-line interface
- constants: Configuration and paths

Example usage:
    from pathlib import Path
    from virtual_literature_companion import load_ingest_file
    
    def main():
        result = load_ingest_file(Path("book.epub"))
        print(f"Processed {len(result)} sections")
        for section in result[:3]:  # Show first 3 sections
            print(f"- {section['section_type']}: {len(section['clean_text'])} chars")

    if __name__ == "__main__":
        main()
"""

__version__ = "1.0.0"
__author__ = "Virtual Literature Companion Team"
__email__ = "contact@vlc.com"

# Import main functions for easy access
from .ingest import load_ingest_file
from .constants import BOOKS_DIR, SRC_DIR, REPO_DIR
from .llm.request import get_ai_status, make_llm_request

# Export key functions and constants
__all__ = [
    'load_ingest_file',
    'BOOKS_DIR',
    'SRC_DIR',
    'REPO_DIR',
    'get_ai_status',
    'make_llm_request',
    '__version__'
] 