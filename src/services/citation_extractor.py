"""
Citation extraction service for the Literary Companion.

This module handles:
- Extraction of citations from AI responses
- Validation of citations against the book text
- Formatting citations for display
- Linking citations to specific book locations
"""

import re
from typing import List, Dict, Optional, Tuple, Any
import logging

from ..models.state import Citation, BookLocation
from ..tools.book_processor import BookProcessor

logger = logging.getLogger(__name__)


class CitationExtractor:
    """
    Extracts and validates citations from text.
    
    Citations can appear in various formats:
    - (Chapter X, p. Y)
    - (Chapter Title, p. Y)
    - (p. Y)
    - (Ch. X)
    """
    
    # Patterns for different citation formats
    CITATION_PATTERNS = [
        # (Chapter X, p. Y) or (Chapter X, page Y)
        re.compile(r'\(Chapter\s+(\d+|[IVXLCDM]+),?\s*p(?:age)?\s*\.?\s*(\d+)\)', re.IGNORECASE),
        
        # (Ch. X, p. Y)
        re.compile(r'\(Ch\.\s*(\d+|[IVXLCDM]+),?\s*p(?:age)?\s*\.?\s*(\d+)\)', re.IGNORECASE),
        
        # (Chapter Title, p. Y)
        re.compile(r'\(([^,\)]+),\s*p(?:age)?\s*\.?\s*(\d+)\)', re.IGNORECASE),
        
        # (p. Y) - page only
        re.compile(r'\(p(?:age)?\s*\.?\s*(\d+)\)', re.IGNORECASE),
        
        # (Chapter X) - chapter only
        re.compile(r'\(Chapter\s+(\d+|[IVXLCDM]+)\)', re.IGNORECASE),
    ]
    
    def extract_citations(self, text: str, current_location: Optional[BookLocation],
                         book_processor: BookProcessor) -> List[Citation]:
        """
        Extract all citations from the given text.
        
        Args:
            text: Text containing citations
            current_location: User's current reading location
            book_processor: Book processor instance for validation
            
        Returns:
            List of extracted Citation objects
        """
        citations = []
        
        # Track processed positions to avoid duplicates
        processed_positions = set()
        
        for pattern_idx, pattern in enumerate(self.CITATION_PATTERNS):
            for match in pattern.finditer(text):
                # Skip if already processed this position
                if match.start() in processed_positions:
                    continue
                
                processed_positions.add(match.start())
                
                # Extract citation based on pattern type
                citation = self._extract_citation_from_match(
                    match, pattern_idx, text, current_location, book_processor
                )
                
                if citation:
                    citations.append(citation)
        
        return citations
    
    def _extract_citation_from_match(self, match: re.Match, pattern_idx: int,
                                   full_text: str, current_location: Optional[BookLocation],
                                   book_processor: BookProcessor) -> Optional[Citation]:
        """Extract a single citation from a regex match."""
        try:
            # Extract components based on pattern type
            if pattern_idx in [0, 1]:  # Chapter + page patterns
                chapter_str = match.group(1)
                page_str = match.group(2)
                
                chapter_num = self._parse_chapter_number(chapter_str)
                page_num = int(page_str)
                
                # Get chapter title from book processor
                chapter_title = self._get_chapter_title_by_number(
                    book_processor, chapter_num
                )
                
                location = BookLocation(
                    chapter_number=chapter_num,
                    chapter_title=chapter_title,
                    page_number=page_num
                )
                
            elif pattern_idx == 2:  # Chapter title + page pattern
                chapter_title = match.group(1).strip()
                page_str = match.group(2)
                page_num = int(page_str)
                
                # Try to find chapter number from title
                chapter_num = self._get_chapter_number_by_title(
                    book_processor, chapter_title
                )
                
                location = BookLocation(
                    chapter_number=chapter_num or 1,
                    chapter_title=chapter_title,
                    page_number=page_num
                )
                
            elif pattern_idx == 3:  # Page only pattern
                page_str = match.group(1)
                page_num = int(page_str)
                
                # Use current chapter if available
                if current_location:
                    location = BookLocation(
                        chapter_number=current_location.chapter_number,
                        chapter_title=current_location.chapter_title,
                        page_number=page_num
                    )
                else:
                    # Default to chapter 1
                    location = BookLocation(
                        chapter_number=1,
                        chapter_title="Chapter 1",
                        page_number=page_num
                    )
                    
            elif pattern_idx == 4:  # Chapter only pattern
                chapter_str = match.group(1)
                chapter_num = self._parse_chapter_number(chapter_str)
                
                chapter_title = self._get_chapter_title_by_number(
                    book_processor, chapter_num
                )
                
                location = BookLocation(
                    chapter_number=chapter_num,
                    chapter_title=chapter_title,
                    page_number=1  # Default to first page
                )
            
            else:
                return None
            
            # Extract surrounding context
            context_start = max(0, match.start() - 100)
            context_end = min(len(full_text), match.end() + 100)
            context = full_text[context_start:context_end]
            
            # Extract the quoted text if it appears before the citation
            quoted_text = self._extract_quoted_text(full_text, match.start())
            
            return Citation(
                location=location,
                text=quoted_text or match.group(0),
                context=context
            )
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse citation: {match.group(0)}, error: {e}")
            return None
    
    def _parse_chapter_number(self, num_str: str) -> int:
        """Convert chapter number string to integer."""
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
    
    def _get_chapter_title_by_number(self, book_processor: BookProcessor,
                                   chapter_num: int) -> str:
        """Get chapter title from chapter number."""
        if book_processor.toc:
            for chapter in book_processor.toc.chapters:
                if chapter.get('number') == chapter_num:
                    return chapter.get('title', f'Chapter {chapter_num}')
        
        return f'Chapter {chapter_num}'
    
    def _get_chapter_number_by_title(self, book_processor: BookProcessor,
                                    title: str) -> Optional[int]:
        """Try to find chapter number from title."""
        if not book_processor.toc:
            return None
        
        title_lower = title.lower().strip()
        
        for chapter in book_processor.toc.chapters:
            chapter_title = chapter.get('title', '').lower().strip()
            if title_lower in chapter_title or chapter_title in title_lower:
                return chapter.get('number')
        
        return None
    
    def _extract_quoted_text(self, full_text: str, citation_pos: int) -> Optional[str]:
        """
        Try to extract quoted text that appears before the citation.
        
        Looks for text in quotes that immediately precedes the citation.
        """
        # Look backwards for closing quote
        search_start = max(0, citation_pos - 500)
        text_before = full_text[search_start:citation_pos]
        
        # Find the last quoted text
        quote_pattern = re.compile(r'"([^"]+)"(?:\s*\([^)]*\))?$')
        match = quote_pattern.search(text_before)
        
        if match:
            return match.group(1)
        
        # Try single quotes
        quote_pattern = re.compile(r"'([^']+)'(?:\s*\([^)]*\))?$")
        match = quote_pattern.search(text_before)
        
        if match:
            return match.group(1)
        
        return None
    
    def format_citation(self, citation: Citation) -> str:
        """
        Format a citation for display.
        
        Args:
            citation: Citation object to format
            
        Returns:
            Formatted citation string
        """
        location = citation.location
        
        if location.page_number and location.page_number > 1:
            return f"({location.chapter_title}, p. {location.page_number})"
        else:
            return f"({location.chapter_title})"
    
    def validate_citation(self, citation: Citation, current_location: Optional[BookLocation]) -> bool:
        """
        Validate that a citation doesn't reference content beyond user's reading progress.
        
        Args:
            citation: Citation to validate
            current_location: User's current reading location
            
        Returns:
            True if citation is valid (within read content), False otherwise
        """
        if not current_location:
            return True  # Can't validate without current location
        
        cite_loc = citation.location
        curr_loc = current_location
        
        # Check chapter
        if cite_loc.chapter_number > curr_loc.chapter_number:
            return False
        
        # If same chapter, check page
        if (cite_loc.chapter_number == curr_loc.chapter_number and 
            cite_loc.page_number > curr_loc.page_number):
            return False
        
        return True