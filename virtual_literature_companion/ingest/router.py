"""
File ingestion routing for the Virtual Literature Companion system.

This module provides a unified interface for ingesting different file types
and routing them to appropriate processing functions.
"""

from pathlib import Path
from typing import List, Dict

from .epub import load_ingest_epub


def load_ingest_file(file_path: Path) -> List[Dict]:
    """
    Route file ingestion to the appropriate function based on file type.
    
    Args:
        file_path (Path): Path to the file to ingest
        
    Returns:
        List[Dict]: List of dictionaries with keys:
            - section_type: The type of section (SectionType enum value)
            - raw_text: The original extracted text
            - clean_text: The cleaned text after processing
            
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is not supported or file is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    suffix = file_path.suffix.lower()
    
    if suffix == '.epub':
        return load_ingest_epub(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Currently only EPUB files are supported.")