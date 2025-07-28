"""
Comprehensive unit tests for the EPUB extraction module.

This test suite covers:
- Single chapter text extraction from HTML content
- EPUB file validation
- Full EPUB text extraction with multithreading
- Error handling for invalid files and corrupted data
- Edge cases and boundary conditions
"""

import pytest
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

import ebooklib
from ebooklib import epub

# Import directly from the module to avoid package initialization issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import constants first to avoid circular imports
from virtual_literature_companion.constants import MAX_WORKERS, DEBUG_MODE

# Now import the functions
from virtual_literature_companion.processors.epub_extraction import (
    extract_single_chapter_text,
    validate_epub_file,
    extract_page_text
)


class TestExtractSingleChapterText:
    """Test cases for the extract_single_chapter_text function."""
    
    def test_basic_html_extraction(self):
        """Test basic HTML text extraction."""
        html_content = """
        <html>
            <head><title>Chapter 1</title></head>
            <body>
                <h1>Chapter One</h1>
                <p>This is the first paragraph of the story.</p>
                <p>This is the second paragraph with <em>emphasis</em>.</p>
            </body>
        </html>
        """
        chapter_id = "chapter01"
        
        result_id, result_text = extract_single_chapter_text(html_content, chapter_id)
        
        assert result_id == chapter_id
        assert "Chapter One" in result_text
        assert "This is the first paragraph of the story." in result_text
        assert "This is the second paragraph with emphasis." in result_text
        assert "<html>" not in result_text
        assert "<p>" not in result_text
    
    def test_html_with_script_and_style_removal(self):
        """Test that script and style elements are removed."""
        html_content = """
        <html>
            <head>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <p>Visible content</p>
                <script>document.write('hidden');</script>
                <style>.hidden { display: none; }</style>
            </body>
        </html>
        """
        chapter_id = "chapter01"
        
        result_id, result_text = extract_single_chapter_text(html_content, chapter_id)
        
        assert result_id == chapter_id
        assert "Visible content" in result_text
        assert "alert" not in result_text
        assert "color: red" not in result_text
        assert "document.write" not in result_text
        assert ".hidden" not in result_text
    
    def test_whitespace_cleanup(self):
        """Test proper whitespace cleanup and normalization."""
        html_content = """
        <html>
            <body>
                <p>   Multiple    spaces   between    words   </p>
                <p>
                    Line breaks
                    and multiple
                    lines
                </p>
                <p>    Leading and trailing spaces    </p>
            </body>
        </html>
        """
        chapter_id = "chapter01"
        
        result_id, result_text = extract_single_chapter_text(html_content, chapter_id)
        
        assert result_id == chapter_id
        assert "Multiple spaces between words" in result_text
        assert "Line breaks and multiple lines" in result_text
        assert "Leading and trailing spaces" in result_text
        # Should not have excessive whitespace
        assert "    " not in result_text
        assert result_text.strip() == result_text
    
    def test_empty_html_content(self):
        """Test handling of empty or minimal HTML content."""
        html_content = "<html><body></body></html>"
        chapter_id = "empty_chapter"
        
        result_id, result_text = extract_single_chapter_text(html_content, chapter_id)
        
        assert result_id == chapter_id
        assert result_text == ""
    
    def test_plain_text_content(self):
        """Test handling of plain text without HTML tags."""
        text_content = "Just plain text without any HTML tags."
        chapter_id = "plain_text"
        
        result_id, result_text = extract_single_chapter_text(text_content, chapter_id)
        
        assert result_id == chapter_id
        assert result_text == "Just plain text without any HTML tags."
    
    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        malformed_html = """
        <html>
            <body>
                <p>Paragraph without closing tag
                <div>Unclosed div
                <span>Some text</p>
            </body>
        """
        chapter_id = "malformed"
        
        result_id, result_text = extract_single_chapter_text(malformed_html, chapter_id)
        
        assert result_id == chapter_id
        assert "Paragraph without closing tag" in result_text
        assert "Some text" in result_text
    
    def test_special_characters_and_unicode(self):
        """Test handling of special characters and Unicode."""
        html_content = """
        <html>
            <body>
                <p>Special characters: &amp; &lt; &gt; &quot; &apos;</p>
                <p>Unicode: café, naïve, résumé, 中文, العربية</p>
                <p>Symbols: © ® ™ € £ ¥</p>
            </body>
        </html>
        """
        chapter_id = "unicode_test"
        
        result_id, result_text = extract_single_chapter_text(html_content, chapter_id)
        
        assert result_id == chapter_id
        assert "Special characters: & < > \" '" in result_text
        assert "café" in result_text
        assert "中文" in result_text
        assert "©" in result_text
    
    @patch('virtual_literature_companion.processors.epub_extraction.logger')
    def test_exception_handling(self, mock_logger):
        """Test exception handling during text extraction."""
        # Simulate an exception during parsing
        with patch('virtual_literature_companion.processors.epub_extraction.BeautifulSoup', 
                   side_effect=Exception("Parsing error")):
            chapter_id = "error_chapter"
            result_id, result_text = extract_single_chapter_text("content", chapter_id)
            
            assert result_id == chapter_id
            assert result_text == ""
            mock_logger.error.assert_called_once()
            assert "error_chapter" in mock_logger.error.call_args[0][0]


class TestValidateEpubFile:
    """Test cases for the validate_epub_file function."""
    
    def create_valid_epub_structure(self, temp_dir):
        """Helper method to create a valid EPUB structure for testing."""
        epub_path = temp_dir / "valid.epub"
        
        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
            <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
            </rootfiles>
        </container>"""
        
        content_opf = """<?xml version="1.0" encoding="UTF-8"?>
        <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
            <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:identifier id="uid">test-book</dc:identifier>
                <dc:title>Test Book</dc:title>
            </metadata>
            <manifest>
                <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
            </manifest>
            <spine>
                <itemref idref="chapter1"/>
            </spine>
        </package>"""
        
        chapter_content = """<?xml version="1.0" encoding="UTF-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml">
            <body><p>Chapter content</p></body>
        </html>"""
        
        with zipfile.ZipFile(epub_path, 'w') as zf:
            zf.writestr("META-INF/container.xml", container_xml)
            zf.writestr("OEBPS/content.opf", content_opf)
            zf.writestr("OEBPS/chapter1.xhtml", chapter_content)
        
        return epub_path
    
    def test_valid_epub_file(self):
        """Test validation of a properly structured EPUB file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = self.create_valid_epub_structure(temp_path)
            
            result = validate_epub_file(str(epub_path))
            assert result is True
    
    def test_missing_container_xml(self):
        """Test validation fails when META-INF/container.xml is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = temp_path / "invalid.epub"
            
            with zipfile.ZipFile(epub_path, 'w') as zf:
                zf.writestr("OEBPS/content.opf", "dummy content")
            
            result = validate_epub_file(str(epub_path))
            assert result is False
    
    def test_invalid_container_xml(self):
        """Test validation fails with malformed container.xml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = temp_path / "invalid.epub"
            
            invalid_container = "not xml content"
            
            with zipfile.ZipFile(epub_path, 'w') as zf:
                zf.writestr("META-INF/container.xml", invalid_container)
            
            result = validate_epub_file(str(epub_path))
            assert result is False
    
    def test_missing_rootfile_in_container(self):
        """Test validation fails when container.xml has no rootfile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = temp_path / "invalid.epub"
            
            container_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                <rootfiles>
                </rootfiles>
            </container>"""
            
            with zipfile.ZipFile(epub_path, 'w') as zf:
                zf.writestr("META-INF/container.xml", container_xml)
            
            result = validate_epub_file(str(epub_path))
            assert result is False
    
    def test_missing_opf_file(self):
        """Test validation fails when referenced OPF file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = temp_path / "invalid.epub"
            
            container_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                <rootfiles>
                    <rootfile full-path="missing.opf" media-type="application/oebps-package+xml"/>
                </rootfiles>
            </container>"""
            
            with zipfile.ZipFile(epub_path, 'w') as zf:
                zf.writestr("META-INF/container.xml", container_xml)
            
            result = validate_epub_file(str(epub_path))
            assert result is False
    
    def test_not_a_zip_file(self):
        """Test validation fails for files that aren't ZIP archives."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = temp_path / "not_zip.epub"
            
            # Create a regular text file
            epub_path.write_text("This is not a ZIP file")
            
            result = validate_epub_file(str(epub_path))
            assert result is False
    
    def test_nonexistent_file(self):
        """Test validation fails for nonexistent files."""
        result = validate_epub_file("/nonexistent/path/book.epub")
        assert result is False
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    def test_ebooklib_validation_failure(self, mock_read_epub):
        """Test validation fails when ebooklib can't read the EPUB."""
        mock_read_epub.side_effect = Exception("Cannot read EPUB")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = self.create_valid_epub_structure(temp_path)
            
            result = validate_epub_file(str(epub_path))
            assert result is False
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    def test_empty_epub_content(self, mock_read_epub):
        """Test validation fails for EPUB with no content items."""
        mock_book = Mock()
        mock_book.get_items.return_value = []
        mock_read_epub.return_value = mock_book
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = self.create_valid_epub_structure(temp_path)
            
            result = validate_epub_file(str(epub_path))
            assert result is False


class TestExtractPageText:
    """Test cases for the extract_page_text function."""
    
    def create_mock_epub_item(self, item_id, content):
        """Helper method to create a mock EPUB item."""
        mock_item = Mock()
        mock_item.get_type.return_value = ebooklib.ITEM_DOCUMENT
        mock_item.get_id.return_value = item_id
        mock_item.content = content.encode('utf-8')
        return mock_item
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_successful_extraction(self, mock_exists, mock_read_epub):
        """Test successful extraction of text from multiple chapters."""
        mock_exists.return_value = True
        
        # Create mock book with multiple chapters
        mock_book = Mock()
        chapter1_content = "<html><body><p>First chapter content</p></body></html>"
        chapter2_content = "<html><body><p>Second chapter content</p></body></html>"
        
        mock_items = [
            self.create_mock_epub_item("ch1", chapter1_content),
            self.create_mock_epub_item("ch2", chapter2_content)
        ]
        mock_book.get_items.return_value = mock_items
        mock_read_epub.return_value = mock_book
        
        page_texts, total_pages = extract_page_text("/path/to/book.epub")
        
        assert total_pages == 2
        assert len(page_texts) == 2
        assert "First chapter content" in page_texts[0]
        assert "Second chapter content" in page_texts[1]
    
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_file_not_found(self, mock_exists):
        """Test FileNotFoundError for nonexistent files."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_page_text("/nonexistent/book.epub")
        
        assert "EPUB file not found" in str(exc_info.value)
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_empty_epub(self, mock_exists, mock_read_epub):
        """Test extraction from EPUB with no document items."""
        mock_exists.return_value = True
        
        mock_book = Mock()
        mock_book.get_items.return_value = []
        mock_read_epub.return_value = mock_book
        
        page_texts, total_pages = extract_page_text("/path/to/empty.epub")
        
        assert total_pages == 0
        assert len(page_texts) == 0
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_mixed_item_types(self, mock_exists, mock_read_epub):
        """Test extraction filters only document items."""
        mock_exists.return_value = True
        
        mock_book = Mock()
        
        # Create mix of document and non-document items
        doc_item = self.create_mock_epub_item("ch1", "<html><body><p>Chapter text</p></body></html>")
        
        css_item = Mock()
        css_item.get_type.return_value = ebooklib.ITEM_STYLE
        
        image_item = Mock()
        image_item.get_type.return_value = ebooklib.ITEM_IMAGE
        
        mock_book.get_items.return_value = [doc_item, css_item, image_item]
        mock_read_epub.return_value = mock_book
        
        page_texts, total_pages = extract_page_text("/path/to/book.epub")
        
        assert total_pages == 1
        assert len(page_texts) == 1
        assert "Chapter text" in page_texts[0]
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_unicode_content(self, mock_exists, mock_read_epub):
        """Test extraction of Unicode content."""
        mock_exists.return_value = True
        
        mock_book = Mock()
        unicode_content = "<html><body><p>测试内容 with émojis 🚀</p></body></html>"
        
        mock_item = self.create_mock_epub_item("unicode_ch", unicode_content)
        mock_book.get_items.return_value = [mock_item]
        mock_read_epub.return_value = mock_book
        
        page_texts, total_pages = extract_page_text("/path/to/unicode.epub")
        
        assert total_pages == 1
        assert "测试内容 with émojis 🚀" in page_texts[0]
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    @patch('virtual_literature_companion.processors.epub_extraction.ThreadPoolExecutor')
    def test_multithreading_execution(self, mock_executor_class, mock_exists, mock_read_epub):
        """Test that multithreading is properly used for processing."""
        mock_exists.return_value = True
        
        # Setup mock executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Create futures that return expected results
        future1 = Future()
        future1.set_result(("ch1", "First chapter text"))
        future2 = Future()
        future2.set_result(("ch2", "Second chapter text"))
        
        mock_executor.submit.side_effect = [future1, future2]
        
        # Setup mock book
        mock_book = Mock()
        mock_items = [
            self.create_mock_epub_item("ch1", "<html><body><p>First</p></body></html>"),
            self.create_mock_epub_item("ch2", "<html><body><p>Second</p></body></html>")
        ]
        mock_book.get_items.return_value = mock_items
        mock_read_epub.return_value = mock_book
        
        page_texts, total_pages = extract_page_text("/path/to/book.epub")
        
        # Verify executor was used correctly
        mock_executor_class.assert_called_once()
        assert mock_executor.submit.call_count == 2
        assert total_pages == 2
    
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_chapter_extraction_error_handling(self, mock_exists, mock_read_epub):
        """Test handling of errors during individual chapter extraction."""
        mock_exists.return_value = True
        
        mock_book = Mock()
        # Create a mock item with problematic content that will cause extraction to return empty string
        problematic_item = Mock()
        problematic_item.get_type.return_value = ebooklib.ITEM_DOCUMENT
        problematic_item.get_id.return_value = "problematic"
        problematic_item.content = b"invalid content"  # This might cause issues in parsing
        
        mock_book.get_items.return_value = [problematic_item]
        mock_read_epub.return_value = mock_book
        
        # Even with problematic content, the function should complete
        page_texts, total_pages = extract_page_text("/path/to/problematic.epub")
        
        assert total_pages == 1
        assert len(page_texts) == 1
        # The content might be empty or processed, but function should not crash
    
    @patch('virtual_literature_companion.processors.epub_extraction.logger')
    @patch('virtual_literature_companion.processors.epub_extraction.epub.read_epub')
    @patch('virtual_literature_companion.processors.epub_extraction.Path.exists')
    def test_logging_behavior(self, mock_exists, mock_read_epub, mock_logger):
        """Test that appropriate logging occurs during extraction."""
        mock_exists.return_value = True
        
        mock_book = Mock()
        mock_item = self.create_mock_epub_item("ch1", "<html><body><p>Test</p></body></html>")
        mock_book.get_items.return_value = [mock_item]
        mock_read_epub.return_value = mock_book
        
        extract_page_text("/path/to/book.epub")
        
        # Verify logging calls
        assert mock_logger.info.call_count >= 2  # Start and completion messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Starting EPUB extraction" in call for call in log_calls)
        assert any("Successfully extracted" in call for call in log_calls)


# Additional integration-style tests
class TestEpubExtractionIntegration:
    """Integration tests that test the module components together."""
    
    def test_end_to_end_with_real_epub_structure(self):
        """Test the complete flow with a realistic EPUB structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            epub_path = temp_path / "test_book.epub"
            
            # Create a more realistic EPUB structure
            container_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                <rootfiles>
                    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
                </rootfiles>
            </container>"""
            
            content_opf = """<?xml version="1.0" encoding="UTF-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
                <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                    <dc:identifier id="uid">test-book-123</dc:identifier>
                    <dc:title>Test Novel</dc:title>
                    <dc:creator>Test Author</dc:creator>
                </metadata>
                <manifest>
                    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
                    <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
                </manifest>
                <spine>
                    <itemref idref="chapter1"/>
                    <itemref idref="chapter2"/>
                </spine>
            </package>"""
            
            chapter1_content = """<?xml version="1.0" encoding="UTF-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml">
                <head><title>Chapter 1</title></head>
                <body>
                    <h1>The Beginning</h1>
                    <p>It was a dark and stormy night. The rain fell in torrents, and the wind howled through the trees.</p>
                    <p>Our hero, <em>John Smith</em>, was walking down the lonely street when he heard a strange noise.</p>
                </body>
            </html>"""
            
            chapter2_content = """<?xml version="1.0" encoding="UTF-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml">
                <head><title>Chapter 2</title></head>
                <body>
                    <h1>The Mystery Deepens</h1>
                    <p>The noise grew louder as John approached the old mansion at the end of the street.</p>
                    <p>"Who's there?" he called out into the darkness.</p>
                </body>
            </html>"""
            
            with zipfile.ZipFile(epub_path, 'w') as zf:
                zf.writestr("META-INF/container.xml", container_xml)
                zf.writestr("OEBPS/content.opf", content_opf)
                zf.writestr("OEBPS/chapter1.xhtml", chapter1_content)
                zf.writestr("OEBPS/chapter2.xhtml", chapter2_content)
            
            # Test validation
            assert validate_epub_file(str(epub_path)) is True
            
            # Test extraction
            page_texts, total_pages = extract_page_text(str(epub_path))
            
            assert total_pages == 2
            assert len(page_texts) == 2
            
            # Verify content extraction
            chapter1_text = page_texts[0]
            chapter2_text = page_texts[1]
            
            assert "The Beginning" in chapter1_text
            assert "dark and stormy night" in chapter1_text
            assert "John Smith" in chapter1_text
            assert "The Mystery Deepens" in chapter2_text
            assert "Who's there?" in chapter2_text
            
            # Verify HTML tags are removed
            assert "<h1>" not in chapter1_text
            assert "<p>" not in chapter1_text
            assert "<em>" not in chapter1_text


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])