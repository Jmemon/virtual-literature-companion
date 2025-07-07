"""
Book processing tools for the Literary Companion.

This module provides functionality to:
- Parse and ingest book text from various formats
- Extract table of contents with chapter boundaries
- Navigate and retrieve text based on reading progress
- Ensure contextual awareness by limiting access to read portions
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import nltk
from bs4 import BeautifulSoup
import logging

from ..models.state import BookLocation, TableOfContents

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """Represents a single chapter in the book."""
    number: int
    title: str
    start_page: int
    end_page: int
    start_position: int  # Character position in the full text
    end_position: int
    text: str


class BookProcessor:
    """
    Handles book text processing and navigation.
    
    This class is responsible for:
    - Parsing raw book text into structured format
    - Extracting table of contents
    - Providing safe access to text based on reading progress
    """
    
    def __init__(self):
        self.chapters: List[Chapter] = []
        self.full_text: str = ""
        self.page_boundaries: List[int] = []  # Character positions for page breaks
        self.toc: Optional[TableOfContents] = None
        
    def ingest_book(self, text: str, book_title: str, author: str) -> TableOfContents:
        """
        Process raw book text and extract structure.
        
        This method attempts to identify:
        - Chapter boundaries and titles
        - Page numbers and breaks
        - Overall book structure
        """
        self.full_text = text
        self.book_title = book_title
        self.author = author
        
        # Extract chapters and structure
        self._extract_chapters()
        self._identify_page_boundaries()
        
        # Build table of contents
        self.toc = self._build_table_of_contents()
        
        return self.toc
    
    def _extract_chapters(self) -> None:
        """
        Extract chapter information from the text.
        
        This uses multiple strategies:
        - Pattern matching for common chapter formats
        - Heuristic analysis of text structure
        - Fallback to uniform division if no clear structure
        """
        # Common chapter patterns
        chapter_patterns = [
            r'(?i)^chapter\s+(\d+|[IVXLCDM]+)(?:\s*[:\.\-]\s*(.+?))?$',
            r'(?i)^(\d+|[IVXLCDM]+)\.?\s+(.+?)$',
            r'(?i)^part\s+(\d+|[IVXLCDM]+)(?:\s*[:\.\-]\s*(.+?))?$',
        ]
        
        # Find all potential chapter markers
        lines = self.full_text.split('\n')
        chapter_markers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in chapter_patterns:
                match = re.match(pattern, line)
                if match:
                    chapter_num = self._parse_chapter_number(match.group(1))
                    chapter_title = match.group(2) if match.lastindex and match.lastindex >= 2 else f"Chapter {chapter_num}"
                    
                    # Calculate character position
                    char_pos = sum(len(l) + 1 for l in lines[:i])  # +1 for newline
                    
                    chapter_markers.append({
                        'number': chapter_num,
                        'title': chapter_title.strip() if chapter_title else f"Chapter {chapter_num}",
                        'line_index': i,
                        'char_position': char_pos
                    })
                    break
        
        # Build chapter objects
        for i, marker in enumerate(chapter_markers):
            start_pos = marker['char_position']
            end_pos = chapter_markers[i + 1]['char_position'] if i + 1 < len(chapter_markers) else len(self.full_text)
            
            # Estimate pages (assuming ~250 words per page, ~5 chars per word)
            chars_per_page = 1250
            start_page = (start_pos // chars_per_page) + 1
            end_page = (end_pos // chars_per_page) + 1
            
            chapter = Chapter(
                number=marker['number'],
                title=marker['title'],
                start_page=start_page,
                end_page=end_page,
                start_position=start_pos,
                end_position=end_pos,
                text=self.full_text[start_pos:end_pos]
            )
            self.chapters.append(chapter)
        
        # If no chapters found, create a single chapter
        if not self.chapters:
            self.chapters.append(Chapter(
                number=1,
                title="Full Text",
                start_page=1,
                end_page=len(self.full_text) // 1250 + 1,
                start_position=0,
                end_position=len(self.full_text),
                text=self.full_text
            ))
    
    def _parse_chapter_number(self, num_str: str) -> int:
        """Convert chapter number string (arabic or roman) to integer."""
        num_str = num_str.upper().strip()
        
        # Try arabic first
        try:
            return int(num_str)
        except ValueError:
            pass
        
        # Try roman numerals
        roman_map = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        try:
            result = 0
            for i in range(len(num_str)):
                if i > 0 and roman_map[num_str[i]] > roman_map[num_str[i-1]]:
                    result += roman_map[num_str[i]] - 2 * roman_map[num_str[i-1]]
                else:
                    result += roman_map[num_str[i]]
            return result
        except:
            return 1  # Default to 1 if parsing fails
    
    def _identify_page_boundaries(self) -> None:
        """
        Identify approximate page boundaries in the text.
        
        Uses heuristics based on:
        - Character count (approximately 250 words per page)
        - Natural break points (paragraph boundaries)
        - Explicit page markers if present
        """
        chars_per_page = 1250  # Approximate
        current_pos = 0
        
        while current_pos < len(self.full_text):
            # Look for natural break point near the target position
            target_pos = current_pos + chars_per_page
            
            # Find nearest paragraph break
            search_start = max(0, target_pos - 200)
            search_end = min(len(self.full_text), target_pos + 200)
            
            paragraph_break = self.full_text.find('\n\n', search_start, search_end)
            
            if paragraph_break != -1:
                self.page_boundaries.append(paragraph_break)
                current_pos = paragraph_break
            else:
                self.page_boundaries.append(target_pos)
                current_pos = target_pos
    
    def _build_table_of_contents(self) -> TableOfContents:
        """Build the table of contents from extracted chapters."""
        chapters_data = [
            {
                'number': ch.number,
                'title': ch.title,
                'start_page': ch.start_page,
                'end_page': ch.end_page
            }
            for ch in self.chapters
        ]
        
        total_pages = max(ch.end_page for ch in self.chapters) if self.chapters else 1
        
        return TableOfContents(
            chapters=chapters_data,
            total_pages=total_pages,
            book_title=self.book_title,
            author=self.author
        )
    
    def get_text_up_to_location(self, location: BookLocation) -> str:
        """
        Retrieve all text up to the specified location.
        
        This is the primary method for ensuring contextual awareness - it only
        returns text that the user has indicated they've read.
        """
        # Find the chapter
        target_chapter = None
        for chapter in self.chapters:
            if chapter.number == location.chapter_number:
                target_chapter = chapter
                break
        
        if not target_chapter:
            logger.warning(f"Chapter {location.chapter_number} not found")
            return ""
        
        # Calculate the character position based on page number
        chars_per_page = 1250
        relative_page = location.page_number - target_chapter.start_page
        char_offset = relative_page * chars_per_page
        
        # Get all text up to this point
        end_position = min(
            target_chapter.start_position + char_offset,
            target_chapter.end_position
        )
        
        # Include all previous chapters plus current chapter up to the page
        accessible_text = self.full_text[:end_position]
        
        return accessible_text
    
    def get_chapter_text(self, chapter_number: int) -> Optional[str]:
        """Get the full text of a specific chapter."""
        for chapter in self.chapters:
            if chapter.number == chapter_number:
                return chapter.text
        return None
    
    def search_accessible_text(self, query: str, location: BookLocation, 
                             context_size: int = 100) -> List[Dict[str, Any]]:
        """
        Search for text within the user's read portion.
        
        Returns matches with surrounding context and citations.
        """
        accessible_text = self.get_text_up_to_location(location)
        results = []
        
        # Case-insensitive search
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        for match in pattern.finditer(accessible_text):
            start = max(0, match.start() - context_size)
            end = min(len(accessible_text), match.end() + context_size)
            
            # Determine which chapter and page this match is in
            match_location = self._position_to_location(match.start())
            
            results.append({
                'text': match.group(),
                'context': accessible_text[start:end],
                'location': match_location,
                'position': match.start()
            })
        
        return results
    
    def _position_to_location(self, char_position: int) -> BookLocation:
        """Convert a character position to a BookLocation."""
        # Find which chapter contains this position
        for chapter in self.chapters:
            if chapter.start_position <= char_position < chapter.end_position:
                # Calculate page within chapter
                relative_pos = char_position - chapter.start_position
                page_offset = relative_pos // 1250
                page_number = chapter.start_page + page_offset
                
                return BookLocation(
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    page_number=page_number
                )
        
        # Default to first chapter if not found
        return BookLocation(
            chapter_number=1,
            chapter_title=self.chapters[0].title if self.chapters else "Unknown",
            page_number=1
        )
    
    def extract_quote_with_citation(self, start_pos: int, end_pos: int) -> Tuple[str, BookLocation]:
        """Extract a quote from the text with its proper citation."""
        quote = self.full_text[start_pos:end_pos]
        location = self._position_to_location(start_pos)
        return quote, location