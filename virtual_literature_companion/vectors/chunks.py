"""
Text chunking utilities for creating different types of text segments.

This module provides data structures and utilities for segmenting book text
into sentences, paragraphs, and overlapping word segments for vector indexing.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ChunkData:
    """Represents a text chunk with metadata for indexing."""
    text: str
    section_index: int
    section_type: str
    start_char_position: int
    end_char_position: int
    book_title: str
    author_name: str
    chunk_type: str  # 'sentence', 'paragraph', or 'generic'
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk data to ChromaDB metadata format."""
        return {
            "section_index": self.section_index,
            "section_type": self.section_type,
            "start_char_position": self.start_char_position,
            "end_char_position": self.end_char_position,
            "book_title": self.book_title,
            "author_name": self.author_name,
            "chunk_type": self.chunk_type,
            "text_length": len(self.text)
        }


class TextChunker:
    """Modular text chunking utility for different segmentation strategies."""
    
    @staticmethod
    def chunk_into_sentences(text: str, section_index: int, section_type: str, 
                           book_title: str, author_name: str) -> List[ChunkData]:
        """Chunk text into sentences."""
        # Simple sentence boundary detection using regex
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        char_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            start_pos = char_position
            end_pos = char_position + len(sentence)
            
            chunk = ChunkData(
                text=sentence,
                section_index=section_index,
                section_type=section_type,
                start_char_position=start_pos,
                end_char_position=end_pos,
                book_title=book_title,
                author_name=author_name,
                chunk_type="sentence"
            )
            chunks.append(chunk)
            
            # Update position accounting for the separator
            char_position = end_pos + 1
            
        return chunks
    
    @staticmethod
    def chunk_into_paragraphs(text: str, section_index: int, section_type: str,
                            book_title: str, author_name: str) -> List[ChunkData]:
        """Chunk text into paragraphs."""
        paragraphs = text.split('\n\n')
        
        chunks = []
        char_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            start_pos = char_position
            end_pos = char_position + len(paragraph)
            
            chunk = ChunkData(
                text=paragraph,
                section_index=section_index,
                section_type=section_type,
                start_char_position=start_pos,
                end_char_position=end_pos,
                book_title=book_title,
                author_name=author_name,
                chunk_type="paragraph"
            )
            chunks.append(chunk)
            
            # Update position accounting for paragraph breaks
            char_position = end_pos + 2  # \n\n
            
        return chunks
    
    @staticmethod
    def chunk_into_overlapping_segments(text: str, section_index: int, section_type: str,
                                      book_title: str, author_name: str,
                                      chunk_size: int = 200, overlap: int = 50) -> List[ChunkData]:
        """Chunk text into overlapping word segments."""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            # If text is smaller than chunk size, create single chunk
            chunk = ChunkData(
                text=text,
                section_index=section_index,
                section_type=section_type,
                start_char_position=0,
                end_char_position=len(text),
                book_title=book_title,
                author_name=author_name,
                chunk_type="generic"
            )
            chunks.append(chunk)
            return chunks
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions approximately
            words_before = words[:i]
            start_pos = len(' '.join(words_before)) + (1 if words_before else 0)
            end_pos = start_pos + len(chunk_text)
            
            chunk = ChunkData(
                text=chunk_text,
                section_index=section_index,
                section_type=section_type,
                start_char_position=start_pos,
                end_char_position=end_pos,
                book_title=book_title,
                author_name=author_name,
                chunk_type="generic"
            )
            chunks.append(chunk)
            
            # Stop if we've covered all words
            if i + chunk_size >= len(words):
                break
                
        return chunks