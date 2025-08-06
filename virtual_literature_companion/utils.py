"""
Utility functions for the Virtual Literature Companion system.

This module contains helper functions for managing ingested books,
file operations, and data persistence.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .constants import BOOKS_DIR
from .types import Book

# Configure logging
logger = logging.getLogger(__name__)


def save_ingested_book(book: Book) -> None:
    """Save ingested book data as JSON in the books directory."""
    # Create a filename based on novel name (sanitized)
    filename = book.title.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
    # Remove any other problematic characters
    filename = ''.join(c for c in filename if c.isalnum() or c in '_-')
    filename = f"{filename}.json"
    
    # Create the full path
    output_path = BOOKS_DIR / filename
    
    # Get JSON-compatible data
    book_data = book.save_json_compatible()
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(book_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved ingested book data to: {output_path}")


def list_ingested_books() -> List[Dict[str, Any]]:
    """
    List all books that have been ingested into the system.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing book information
        (converted from Book dataclass for backward compatibility)
    """
    books = []
    
    if not BOOKS_DIR.exists():
        return books
    
    # Find all JSON files in the books directory
    json_files = list(BOOKS_DIR.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            
            # Create Book instance and populate additional fields
            book = Book.from_dict(book_data)
            book.file_path = str(json_file)
            book.directory_name = json_file.stem
            book.directory_path = str(json_file.parent)
            book.has_indexes = False  # TODO: Update when indexes are implemented
            
            # Convert back to dict for backward compatibility
            books.append(book.to_dict())
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading book file {json_file}: {e}")
            continue
    
    return books