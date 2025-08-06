from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class PageType(Enum):
    """Enumeration of different page types found in books."""
    BLANK = "blank"
    TITLE_PAGE = "title_page"
    TABLE_OF_CONTENTS = "table_of_contents"
    COPYWRIGHT_PAGE = "copyright_page"
    STORY_BREAK = "story_break"
    FRONT_MATTER_BREAK = "front_matter_break"
    BACK_MATTER_BREAK = "back_matter_break"
    CONTENT = "content"


class SectionType(str, Enum):
    TABLE_OF_CONTENTS = 'table_of_contents'
    ACKNOWLEDGEMENTS = 'acknowledgements'
    BIBLIOGRAPHY = 'bibliography'
    GLOSSARY = 'glossary'
    INDEX = 'index'
    TITLE_PAGE = 'title_page'
    PREFACE = 'preface'
    EPILOGUE = 'epilogue'
    APPENDIX = 'appendix'
    DEDICATION = 'dedication'
    CHAPTER = 'chapter'
    COPYRIGHT_PAGE = 'copyright_page'
    OTHER = 'other'


@dataclass
class BookStatistics:
    """Statistics about a book's content."""
    total_sections: int = 0
    total_characters: int = 0
    total_chapters: int = 0
    total_word_count: int = 0
    total_paragraphs: int = 0
    unique_characters: int = 0
    unique_settings: int = 0

    @classmethod
    def from_sections(cls, sections: List[Dict]) -> 'BookStatistics':
        """Create statistics from sections data."""
        total_sections = len(sections)
        total_characters = sum(len(s.get('clean_text', '')) for s in sections)
        
        # Count chapters
        total_chapters = len([
            s for s in sections 
            if s.get('section_type') == 'chapter' or 'chapter' in str(s.get('section_type', '')).lower()
        ])
        
        # Count words
        total_word_count = sum(
            len(s.get('clean_text', '').split()) for s in sections
        )
        
        # Count paragraphs
        total_paragraphs = sum(
            s.get('clean_text', '').count('\n\n') + 1 for s in sections
            if s.get('clean_text')
        )
        
        return cls(
            total_sections=total_sections,
            total_characters=total_characters,
            total_chapters=total_chapters,
            total_word_count=total_word_count,
            total_paragraphs=total_paragraphs
        )


@dataclass
class Book:
    """Unified representation of a book in the system."""
    title: str
    author_name: str
    sections: List[Dict] = field(default_factory=list)
    ingestion_date: Optional[str] = None
    statistics: BookStatistics = field(default_factory=BookStatistics)
    file_path: Optional[str] = None
    directory_name: Optional[str] = None
    directory_path: Optional[str] = None
    has_indexes: bool = False
    publication_date: Optional[str] = None
    author_bio: Optional[str] = None
    table_of_contents: List[Dict] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.ingestion_date is None:
            self.ingestion_date = datetime.now().isoformat()
        
        if self.sections and isinstance(self.statistics, dict):
            # Convert dict to BookStatistics if needed
            self.statistics = BookStatistics(**self.statistics)
        elif self.sections and not hasattr(self.statistics, 'total_sections'):
            # Generate statistics from sections
            self.statistics = BookStatistics.from_sections(self.sections)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Book':
        """Create a Book instance from a dictionary."""
        # Handle statistics
        stats_data = data.get('statistics', {})
        if isinstance(stats_data, dict):
            statistics = BookStatistics(**stats_data)
        else:
            statistics = stats_data

        return cls(
            title=data.get('title', 'Unknown'),
            author_name=data.get('author_name', 'Unknown'),
            sections=data.get('sections', []),
            ingestion_date=data.get('ingestion_date'),
            statistics=statistics,
            file_path=data.get('file_path'),
            directory_name=data.get('directory_name'),
            directory_path=data.get('directory_path'),
            has_indexes=data.get('has_indexes', False),
            publication_date=data.get('publication_date'),
            author_bio=data.get('author_bio'),
            table_of_contents=data.get('table_of_contents', []),
            processing_metadata=data.get('processing_metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Book instance to a dictionary for serialization."""
        return {
            'title': self.title,
            'author_name': self.author_name,
            'ingestion_date': self.ingestion_date,
            'sections': self.sections,
            'statistics': {
                'total_sections': self.statistics.total_sections,
                'total_characters': self.statistics.total_characters,
                'total_chapters': self.statistics.total_chapters,
                'total_word_count': self.statistics.total_word_count,
                'total_paragraphs': self.statistics.total_paragraphs,
                'unique_characters': self.statistics.unique_characters,
                'unique_settings': self.statistics.unique_settings
            },
            'file_path': self.file_path,
            'directory_name': self.directory_name,
            'directory_path': self.directory_path,
            'has_indexes': self.has_indexes,
            'publication_date': self.publication_date,
            'author_bio': self.author_bio,
            'table_of_contents': self.table_of_contents,
            'processing_metadata': self.processing_metadata
        }

    def save_json_compatible(self) -> Dict[str, Any]:
        """Get a JSON-compatible representation for saving to file."""
        return {
            'title': self.title,
            'author_name': self.author_name,
            'ingestion_date': self.ingestion_date,
            'sections': self.sections,
            'statistics': {
                'total_sections': self.statistics.total_sections,
                'total_characters': self.statistics.total_characters
            }
        }