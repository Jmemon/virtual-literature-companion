"""
Novel text processing from extracted pages for the Virtual Literature Companion system.

This module handles page categorization, statistics computation, front/back matter processing,
chapter extraction, and saving to raw directory.
"""

import logging
import re
import statistics
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

from ..constants import BOOKS_DIR, DEBUG_MODE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PageType(Enum):
    """Enumeration of different page types found in books."""
    TITLE_PAGE = "title_page"
    COPYRIGHT_PAGE = "copyright_page"
    DEDICATION_PAGE = "dedication_page"
    TABLE_OF_CONTENTS_PAGE = "table_of_contents_page"
    FOREWORD_PREFACE_START = "foreword_preface_start"
    ACKNOWLEDGEMENTS_START = "acknowledgements_start"
    INTRODUCTION_START = "introduction_start"
    CHAPTER_START = "chapter_start"
    PART_START = "part_start"
    CONTENT = "content"
    APPENDIX_START = "appendix_start"
    GLOSSARY_START = "glossary_start"
    BIBLIOGRAPHY_PAGE = "bibliography_page"
    INDEX_PAGE = "index_page"
    THROWAWAY = "throwaway"


def calculate_word_stats(text: str) -> Tuple[int, float]:
    """
    Calculate word count and word diversity for text analysis.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        Tuple[int, float]: (word_count, word_diversity_ratio)
            word_diversity_ratio is unique_words / total_words
    """
    # Clean and split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    if word_count == 0:
        return 0, 0.0
    
    # Calculate diversity as ratio of unique words to total words
    unique_words = len(set(words))
    diversity_ratio = unique_words / word_count
    
    return word_count, diversity_ratio


def categorize_page(text: str, page_num: int, total_pages: int, mean_words: float = 0, std_words: float = 0) -> PageType:
    """
    Categorize a page based on its text content, position in the book, and global word statistics.
    
    Enhanced with global mean and std dev of words per page to make density checks relative to the book.
    For example, title pages are identified if they have very low word density compared to the book's average
    (word_count < mean - 2*std) in addition to pattern matching.
    """
    text_lower = text.lower().strip()
    text_lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Calculate word statistics for sanity checks
    word_count, word_diversity = calculate_word_stats(text)
    
    # Density classifications (with safeguards for low std)
    effective_std = std_words if std_words > 0 else max(100, mean_words * 0.2)  # Fallback if no variation
    is_very_low_density = word_count < max(20, mean_words - 2 * effective_std)
    is_low_density = word_count < max(50, mean_words - effective_std)
    is_high_density = word_count > mean_words + effective_std
    is_medium_density = not (is_very_low_density or is_low_density or is_high_density)
    
    # Early pages (first 10% of book) - front matter
    if page_num < total_pages * 0.1:
        # Title page patterns
        # Enhanced: Require very low density
        if (is_very_low_density and len(text_lines) <= 10 and word_count < 100 and word_diversity < 0.8 and
            any(word in text_lower for word in ['title', 'author', 'novel', 'book']) and
            not any(word in text_lower for word in ['copyright', '©', 'published', 'isbn'])):
            if DEBUG_MODE:
                logger.debug(f"Page {page_num + 1}: Categorized as TITLE_PAGE - "
                             f"word_count={word_count} (mean={mean_words:.1f}, std={std_words:.1f}), "
                             f"word_diversity={word_diversity:.3f}")
            return PageType.TITLE_PAGE
        
        # Copyright page patterns
        # Enhanced: Require low density
        if (is_low_density and word_count < 400 and word_diversity < 0.7 and
            any(word in text_lower for word in ['copyright', '©', 'published', 'isbn', 'publisher', 'printing', 'edition'])):
            if DEBUG_MODE:
                logger.debug(f"Page {page_num + 1}: Categorized as COPYRIGHT_PAGE - "
                             f"word_count={word_count} (mean={mean_words:.1f}, std={std_words:.1f}), "
                             f"word_diversity={word_diversity:.3f}")
            return PageType.COPYRIGHT_PAGE
        
        # Dedication page patterns
        # Enhanced: Require very low density
        if (is_very_low_density and len(text_lines) <= 8 and word_count < 80 and word_diversity < 0.9 and
            any(phrase in text_lower for phrase in ['dedicated to', 'for ', 'in memory of', 'to my'])):
            return PageType.DEDICATION_PAGE
        
        # Table of contents patterns
        # Enhanced: Require low density (TOC is sparse)
        if (is_low_density and word_count < 500 and word_diversity < 0.6 and
            (any(word in text_lower for word in ['contents', 'table of contents']) or
             (len(text_lines) > 5 and 
              any(re.search(r'\d+$', line) for line in text_lines[-5:])))):  # Page numbers at end of lines
            return PageType.TABLE_OF_CONTENTS_PAGE
        
        # Foreword/Preface patterns
        # Enhanced: Require medium density, not too low or high
        if (is_medium_density and word_count < 800 and
            any(word in text_lower for word in ['foreword', 'preface', 'prologue'])):
            return PageType.FOREWORD_PREFACE_START
        
        # Acknowledgements patterns
        # Enhanced: Require low to medium density
        if ((is_low_density or is_medium_density) and word_count < 600 and
            any(word in text_lower for word in ['acknowledgment', 'acknowledgement', 'thanks', 'grateful'])):
            return PageType.ACKNOWLEDGEMENTS_START
        
        # Introduction patterns
        # Enhanced: Allow medium to high density but not excessive
        if (not is_very_low_density and word_count < 1000 and
            any(word in text_lower for word in ['introduction', 'overview'])):
            return PageType.INTRODUCTION_START
    
    # Late pages (last 20% of book) - back matter
    elif page_num > total_pages * 0.8:
        # Appendix patterns
        # Enhanced: Allow medium to high density
        if (not is_low_density and word_count < 1200 and
            any(word in text_lower for word in ['appendix', 'supplementary'])):
            return PageType.APPENDIX_START
        
        # Glossary patterns
        # Enhanced: Require low density due to list format
        if (is_low_density and word_diversity < 0.6 and
            any(word in text_lower for word in ['glossary', 'definitions', 'terms'])):
            return PageType.GLOSSARY_START
        
        # Bibliography patterns
        # Enhanced: Require low density
        if (is_low_density and word_diversity < 0.5 and
            any(word in text_lower for word in ['bibliography', 'references', 'works cited', 'sources'])):
            return PageType.BIBLIOGRAPHY_PAGE
        
        # Index patterns
        # Enhanced: Require very low density (highly structured)
        if (is_very_low_density and word_diversity < 0.4 and
            (any(word in text_lower for word in ['index']) or
             (len(text_lines) > 10 and 
              sum(1 for line in text_lines if re.search(r'\d+(-\d+)?$', line)) > len(text_lines) * 0.3))):
            return PageType.INDEX_PAGE
    
    # Chapter start patterns (can occur anywhere in main content)
    chapter_patterns = [
        r'(?i)^chapter\s+(?:d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'(?i)^(?:d+|[ivx]+)\.\s+\w+',  # "1. Title" or "I. Title"
        r'^[A-Z][A-Z\s]{5,50}'  # All caps titles
    ]
    
    for line in text_lines[:5]:  # Check first few lines
        for pattern in chapter_patterns:
            if re.match(pattern, line):
                if DEBUG_MODE:
                    logger.debug(f"Page {page_num + 1}: Categorized as CHAPTER_START - "
                                 f"word_count={word_count} (mean={mean_words:.1f}, std={std_words:.1f}), "
                                 f"word_diversity={word_diversity:.3f}, matched_line='{line[:50]}...'")
                return PageType.CHAPTER_START
    
    # Part start patterns
    if any(re.match(r'(?i)^part\s+(?:d+|[ivx]+)', line) for line in text_lines[:3]):
        return PageType.PART_START
    
    # Default to content if none of the above patterns match
    # Enhanced: Only if not low density (content should be substantial)
    if not is_low_density:
        if DEBUG_MODE:
            logger.debug(f"Page {page_num + 1}: Categorized as CONTENT - "
                         f"word_count={word_count} (mean={mean_words:.1f}, std={std_words:.1f}), "
                         f"word_diversity={word_diversity:.3f}, position={page_num/total_pages:.2%}")
        return PageType.CONTENT
    
    # Fallback for low density pages that don't match other categories
    return PageType.CONTENT


def process_front_matter_pages(pages_by_type: Dict[PageType, List[Tuple[int, str]]], raw_dir: Path) -> None:
    """
    Process and save front matter pages (title, copyright, dedication, etc.).
    
    This function handles pages that appear at the beginning of the book and
    concatenates multi-page sections like table of contents.
    
    Args:
        pages_by_type (Dict[PageType, List[Tuple[int, str]]]): Pages grouped by type
        raw_dir (Path): Directory to save the processed files
    """
    front_matter_types = [
        (PageType.TITLE_PAGE, "title.txt"),
        (PageType.COPYRIGHT_PAGE, "copyright.txt"),
        (PageType.DEDICATION_PAGE, "dedication.txt"),
        (PageType.TABLE_OF_CONTENTS_PAGE, "table_of_contents.txt"),
        (PageType.FOREWORD_PREFACE_START, "foreword_preface.txt"),
        (PageType.ACKNOWLEDGEMENTS_START, "acknowledgements.txt"),
        (PageType.INTRODUCTION_START, "introduction.txt"),
    ]
    
    for page_type, filename in front_matter_types:
        if page_type in pages_by_type:
            # Sort pages by page number and concatenate
            sorted_pages = sorted(pages_by_type[page_type], key=lambda x: x[0])
            combined_text = "\n\n".join(text for _, text in sorted_pages)
            
            if combined_text.strip():
                file_path = raw_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                
                logger.info(f"Created {filename} from {len(sorted_pages)} page(s)")


def process_back_matter_pages(pages_by_type: Dict[PageType, List[Tuple[int, str]]], raw_dir: Path) -> None:
    """
    Process and save back matter pages (appendix, glossary, bibliography, index).
    
    This function handles pages that appear at the end of the book and
    concatenates multi-page sections.
    
    Args:
        pages_by_type (Dict[PageType, List[Tuple[int, str]]]): Pages grouped by type
        raw_dir (Path): Directory to save the processed files
    """
    back_matter_types = [
        (PageType.APPENDIX_START, "appendix.txt"),
        (PageType.GLOSSARY_START, "glossary.txt"),
        (PageType.BIBLIOGRAPHY_PAGE, "bibliography.txt"),
        (PageType.INDEX_PAGE, "index.txt"),
    ]
    
    for page_type, filename in back_matter_types:
        if page_type in pages_by_type:
            # Sort pages by page number and concatenate
            sorted_pages = sorted(pages_by_type[page_type], key=lambda x: x[0])
            combined_text = "\n\n".join(text for _, text in sorted_pages)
            
            if combined_text.strip():
                file_path = raw_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                
                logger.info(f"Created {filename} from {len(sorted_pages)} page(s)")


def extract_chapters_from_pages(pages_by_type: Dict[PageType, List[Tuple[int, str]]], 
                               all_pages: Dict[int, Tuple[str, PageType]]) -> List[str]:
    """
    Extract chapter texts based on chapter start pages and content pages.
    
    This function identifies chapter boundaries using CHAPTER_START pages and
    concatenates all content until the next chapter or end of content.
    
    Args:
        pages_by_type (Dict[PageType, List[Tuple[int, str]]]): Pages grouped by type
        all_pages (Dict[int, Tuple[str, PageType]]): All pages with their content and types
        
    Returns:
        List[str]: List of chapter texts in order
    """
    chapter_texts = []
    
    # Get all chapter start pages and sort by page number
    chapter_starts = []
    if PageType.CHAPTER_START in pages_by_type:
        chapter_starts = sorted(pages_by_type[PageType.CHAPTER_START], key=lambda x: x[0])
    
    if not chapter_starts:
        logger.warning("No chapter start pages found, treating all content as single chapter")
        # If no chapters found, concatenate all content pages
        content_pages = []
        for page_num in sorted(all_pages.keys()):
            text, page_type = all_pages[page_num]
            if page_type in [PageType.CONTENT, PageType.CHAPTER_START]:
                content_pages.append(text)
        
        if content_pages:
            chapter_texts.append("\n\n".join(content_pages))
        
        return chapter_texts
    
    # Process each chapter
    for i, (chapter_start_page, chapter_start_text) in enumerate(chapter_starts):
        chapter_content = []
        
        # Determine the range of pages for this chapter
        next_chapter_page = chapter_starts[i + 1][0] if i + 1 < len(chapter_starts) else float('inf')
        
        # Add the chapter start text (may contain some content after the heading)
        chapter_content.append(chapter_start_text)
        
        # Add all content pages until next chapter
        for page_num in sorted(all_pages.keys()):
            if chapter_start_page < page_num < next_chapter_page:
                text, page_type = all_pages[page_num]
                if page_type == PageType.CONTENT:
                    chapter_content.append(text)
                elif page_type == PageType.PART_START:
                    # Include part starts in the chapter content
                    chapter_content.append(text)
        
        # Combine all content for this chapter
        if chapter_content:
            combined_chapter = "\n\n".join(chapter_content)
            chapter_texts.append(combined_chapter)
            
            if DEBUG_MODE:
                logger.debug(f"Chapter {i + 1}: {len(combined_chapter)} characters from "
                           f"page {chapter_start_page + 1} to {next_chapter_page}")
    
    return chapter_texts


def process_extracted_pages(page_texts: Dict[int, str], novel_name: str, total_pages: int) -> List[str]:
    """
    Process extracted and cleaned page texts into categorized content with chapter splitting.
    
    This function computes statistics, categorizes pages, processes front/back matter,
    extracts chapters, and saves to raw directory.
    
    Args:
        page_texts (Dict[int, str]): Dictionary of page number to cleaned text
        novel_name (str): Name of the novel
        total_pages (int): Total number of pages
        
    Returns:
        List[str]: List of chapter texts
    """
    logger.info(f"Processing extracted pages for '{novel_name}'")
    
    # Compute global statistics
    word_counts = []
    for text in page_texts.values():
        if text.strip():
            word_count, _ = calculate_word_stats(text)
            word_counts.append(word_count)
    
    if word_counts:
        mean_words = statistics.mean(word_counts)
        std_words = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
    else:
        mean_words = 0
        std_words = 0
    
    logger.info(f"Global stats: mean_words={mean_words:.1f}, std_words={std_words:.1f}, "
                f"from {len(word_counts)} non-empty pages")
    
    # Categorize pages using statistics
    all_pages = {}
    for page_num, text in page_texts.items():
        page_type = categorize_page(text, page_num, total_pages, mean_words, std_words)
        all_pages[page_num] = (text, page_type)
        
        if DEBUG_MODE:
            logger.debug(f"Categorized page {page_num + 1} as {page_type.value}")
    
    # Group pages by type
    pages_by_type = {}
    for page_num, (text, page_type) in all_pages.items():
        if text.strip():  # Only include pages with content
            pages_by_type.setdefault(page_type, []).append((page_num, text))
    
    # Log page type distribution
    for page_type, pages in pages_by_type.items():
        logger.info(f"Found {len(pages)} page(s) of type: {page_type.value}")
    
    # Create output directory
    book_dir = BOOKS_DIR / novel_name
    raw_dir = book_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Process different page types
    process_front_matter_pages(pages_by_type, raw_dir)
    process_back_matter_pages(pages_by_type, raw_dir)
    
    # Extract chapters based on page categorization
    chapter_texts = extract_chapters_from_pages(pages_by_type, all_pages)
    
    # Write chapter files
    for i, chapter_text in enumerate(chapter_texts, 1):
        chapter_file = raw_dir / f"{i}.txt"
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        
        logger.info(f"Created chapter file: {chapter_file} ({len(chapter_text)} characters)")
    
    total_chars = sum(len(text) for text in page_texts.values())
    logger.info(f"Successfully processed '{novel_name}': {total_chars} total characters, "
                f"{len(chapter_texts)} chapters, {len(pages_by_type)} different page types")
    
    return chapter_texts 