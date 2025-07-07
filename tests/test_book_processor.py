"""
Tests for the Book Processor module.

These tests verify:
- Chapter extraction functionality
- Table of contents generation
- Text navigation based on reading progress
- Citation extraction and validation
"""

import pytest
from src.tools.book_processor import BookProcessor, Chapter
from src.models.state import BookLocation


class TestBookProcessor:
    """Test suite for BookProcessor functionality."""
    
    @pytest.fixture
    def sample_book_text(self):
        """Provide sample book text for testing."""
        return """
The Brothers Karamazov
by Fyodor Dostoevsky

PART I

Chapter 1: Fyodor Pavlovitch Karamazov

Alexey Fyodorovitch Karamazov was the third son of Fyodor Pavlovitch Karamazov, 
a land owner well known in our district in his own day, and still remembered among 
us owing to his gloomy and tragic death, which happened thirteen years ago, and 
which I shall describe in its proper place.

This opening introduces us to the Karamazov family and sets the stage for the 
dramatic events that will unfold. The father, Fyodor Pavlovitch, is a character 
of particular interest - a sensualist and buffoon who nonetheless possesses a 
certain cunning intelligence.

Chapter 2: He Gets Rid of His Eldest Son

You must know that Fyodor Pavlovitch was twice married, and had three sons, the 
eldest Dmitri, by his first wife, and two, Ivan and Alexey, by his second. 
Fyodor Pavlovitch's first wife, Adelaïda Ivanovna, belonged to a fairly rich 
and distinguished noble family.

The abandonment of young Dmitri reveals much about Fyodor Pavlovitch's character.
He is a man who shirks responsibility and pursues his own pleasures without 
regard for others, even his own children.

Chapter 3: The Second Marriage and the Second Family

Very shortly after getting his four-year-old Mitya off his hands Fyodor 
Pavlovitch married a second time. His second marriage lasted eight years. He 
took this second wife, Sofya Ivanovna, also a very young girl, from another 
province, where he had gone upon some small piece of business.
"""
    
    @pytest.fixture
    def book_processor(self):
        """Create a BookProcessor instance."""
        return BookProcessor()
    
    def test_chapter_extraction(self, book_processor, sample_book_text):
        """Test that chapters are correctly extracted from the text."""
        toc = book_processor.ingest_book(
            text=sample_book_text,
            book_title="The Brothers Karamazov",
            author="Fyodor Dostoevsky"
        )
        
        assert toc is not None
        assert len(book_processor.chapters) == 3
        assert book_processor.chapters[0].title == "Fyodor Pavlovitch Karamazov"
        assert book_processor.chapters[1].title == "He Gets Rid of His Eldest Son"
        assert book_processor.chapters[2].title == "The Second Marriage and the Second Family"
    
    def test_table_of_contents_generation(self, book_processor, sample_book_text):
        """Test that table of contents is properly generated."""
        toc = book_processor.ingest_book(
            text=sample_book_text,
            book_title="The Brothers Karamazov",
            author="Fyodor Dostoevsky"
        )
        
        assert toc.book_title == "The Brothers Karamazov"
        assert toc.author == "Fyodor Dostoevsky"
        assert len(toc.chapters) == 3
        assert toc.total_pages > 0
    
    def test_text_navigation_by_location(self, book_processor, sample_book_text):
        """Test retrieving text up to a specific reading location."""
        book_processor.ingest_book(
            text=sample_book_text,
            book_title="The Brothers Karamazov",
            author="Fyodor Dostoevsky"
        )
        
        # Test reading up to Chapter 2
        location = BookLocation(
            chapter_number=2,
            chapter_title="He Gets Rid of His Eldest Son",
            page_number=1
        )
        
        accessible_text = book_processor.get_text_up_to_location(location)
        
        # Should include Chapter 1 and beginning of Chapter 2
        assert "Alexey Fyodorovitch Karamazov was the third son" in accessible_text
        assert "You must know that Fyodor Pavlovitch was twice married" in accessible_text
        # Should NOT include Chapter 3
        assert "The Second Marriage and the Second Family" not in accessible_text
    
    def test_chapter_number_parsing(self, book_processor):
        """Test parsing of different chapter number formats."""
        assert book_processor._parse_chapter_number("1") == 1
        assert book_processor._parse_chapter_number("10") == 10
        assert book_processor._parse_chapter_number("I") == 1
        assert book_processor._parse_chapter_number("IV") == 4
        assert book_processor._parse_chapter_number("IX") == 9
        assert book_processor._parse_chapter_number("XIV") == 14
    
    def test_search_accessible_text(self, book_processor, sample_book_text):
        """Test searching within accessible text based on reading progress."""
        book_processor.ingest_book(
            text=sample_book_text,
            book_title="The Brothers Karamazov",
            author="Fyodor Dostoevsky"
        )
        
        location = BookLocation(
            chapter_number=2,
            chapter_title="He Gets Rid of His Eldest Son",
            page_number=1
        )
        
        # Search for text that appears in Chapter 1
        results = book_processor.search_accessible_text("sensualist", location)
        assert len(results) > 0
        assert "sensualist and buffoon" in results[0]['context']
        
        # Search for text that appears in Chapter 3 (should not be found)
        results = book_processor.search_accessible_text("eight years", location)
        assert len(results) == 0


class TestChapterBoundaries:
    """Test edge cases and boundaries in chapter processing."""
    
    def test_empty_text(self):
        """Test handling of empty book text."""
        processor = BookProcessor()
        toc = processor.ingest_book("", "Empty Book", "No Author")
        
        assert toc is not None
        assert len(processor.chapters) == 1  # Should create default chapter
        assert processor.chapters[0].title == "Full Text"
    
    def test_no_chapters_found(self):
        """Test handling of text without clear chapter markers."""
        processor = BookProcessor()
        text = """This is a book without any clear chapter divisions.
        It just continues as one long narrative without breaks.
        The story flows from beginning to end."""
        
        toc = processor.ingest_book(text, "No Chapters", "Author")
        
        assert len(processor.chapters) == 1
        assert processor.chapters[0].title == "Full Text"
        assert processor.chapters[0].text == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])