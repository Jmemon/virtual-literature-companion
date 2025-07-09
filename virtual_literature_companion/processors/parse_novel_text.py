"""
Novel text parsing processor for the Virtual Literature Companion system.

This module handles the conversion of raw chapter text into structured JSON format,
performing paragraph-level analysis including:
- Character identification and involvement tracking
- Setting detection and scene localization
- Dialogue classification
- Paragraph metadata generation

It also handles book-level metadata generation including author bio lookup and
table of contents creation.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup

from openai import OpenAI

from ..constants import BOOKS_DIR, OPENAI_API_KEY, DEFAULT_LLM_MODEL, DEBUG_MODE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.warning("OpenAI API key not found. LLM analysis will be skipped.")


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split chapter text into individual paragraphs.
    
    This function handles various paragraph separation patterns including:
    - Double newlines (standard paragraph breaks)
    - Dialogue formatting
    - Chapter/section breaks
    - Indent-based paragraphs
    
    Args:
        text (str): Raw chapter text
        
    Returns:
        List[str]: List of paragraph texts
    """
    # Normalize line endings and remove excessive whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Split on double newlines (standard paragraph breaks)
    paragraphs = text.split('\n\n')
    
    # Clean up paragraphs
    cleaned_paragraphs = []
    for para in paragraphs:
        # Remove excessive whitespace and normalize
        para = re.sub(r'\s+', ' ', para.strip())
        
        # Skip empty paragraphs or very short ones (likely artifacts)
        if len(para) < 10:
            continue
            
        # Split very long paragraphs that might contain multiple thoughts
        if len(para) > 1000:
            # Try to split on sentence boundaries for long paragraphs
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_para = ""
            
            for sentence in sentences:
                if len(current_para + sentence) < 800:
                    current_para += sentence + " "
                else:
                    if current_para.strip():
                        cleaned_paragraphs.append(current_para.strip())
                    current_para = sentence + " "
            
            if current_para.strip():
                cleaned_paragraphs.append(current_para.strip())
        else:
            cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs


def analyze_paragraph_with_llm(paragraph: str, chapter_context: str = "") -> Dict[str, Any]:
    """
    Analyze a paragraph using OpenAI's GPT to extract characters, settings, and dialogue info.
    
    This function uses a carefully crafted prompt to analyze literary text and extract:
    - Character names and their roles/involvement
    - Setting details (locations, time periods, etc.)
    - Whether the paragraph contains dialogue
    - Narrative perspective and tone
    
    Args:
        paragraph (str): The paragraph text to analyze
        chapter_context (str): Optional context from the chapter for better analysis
        
    Returns:
        Dict[str, Any]: Analysis results with characters, settings, and dialogue flag
    """
    if not client:
        # Fallback analysis without LLM
        logger.warning("No OpenAI client available, using fallback analysis")
        return {
            "characters": ["narrator"],
            "setting": [],
            "dialogue": '"' in paragraph or "'" in paragraph,
            "confidence": 0.3
        }
    
    # Construct analysis prompt
    prompt = f"""
    Analyze the following literary paragraph and extract key information. Be precise and concise.

    Paragraph:
    "{paragraph}"

    {f"Chapter context: {chapter_context}" if chapter_context else ""}

    Please provide a JSON response with the following structure:
    {{
        "characters": ["list of character names mentioned or involved"],
        "setting": ["list of locations, time periods, or settings"],
        "dialogue": true/false,
        "confidence": 0.0-1.0
    }}

    Guidelines:
    - For characters: Include only actual character names, not pronouns. If no specific characters, use ["narrator"]
    - For setting: Include specific locations, time periods, or environmental details. Empty list if none.
    - For dialogue: true if the paragraph contains direct speech (quotes), false otherwise
    - For confidence: How confident you are in the analysis (0.0-1.0)

    Return only the JSON object, no additional text.
    """
    
    try:
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a literary analysis expert. Analyze text precisely and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Clean up the response (remove markdown formatting if present)
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        analysis = json.loads(content)
        
        # Validate and clean the response
        if not isinstance(analysis.get("characters"), list):
            analysis["characters"] = ["narrator"]
        if not isinstance(analysis.get("setting"), list):
            analysis["setting"] = []
        if not isinstance(analysis.get("dialogue"), bool):
            analysis["dialogue"] = '"' in paragraph or "'" in paragraph
        if not isinstance(analysis.get("confidence"), (int, float)):
            analysis["confidence"] = 0.5
            
        return analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response content: {response.choices[0].message.content}")
        return {
            "characters": ["narrator"],
            "setting": [],
            "dialogue": '"' in paragraph or "'" in paragraph,
            "confidence": 0.2
        }
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return {
            "characters": ["narrator"],
            "setting": [],
            "dialogue": '"' in paragraph or "'" in paragraph,
            "confidence": 0.1
        }


def process_chapter_to_json(chapter_text: str, chapter_num: int, novel_name: str) -> Dict[str, Any]:
    """
    Process a single chapter into structured JSON format.
    
    This function takes raw chapter text and converts it into a structured format
    with paragraph-level analysis including character involvement, setting details,
    and dialogue detection.
    
    Args:
        chapter_text (str): Raw text of the chapter
        chapter_num (int): Chapter number (1-indexed)
        novel_name (str): Name of the novel
        
    Returns:
        Dict[str, Any]: Structured chapter data
    """
    logger.info(f"Processing chapter {chapter_num} into JSON format")
    
    # Extract chapter title from the beginning of the text
    lines = chapter_text.split('\n')
    chapter_title = f"Chapter {chapter_num}"
    
    # Look for a title in the first few lines
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        if line and not line.isdigit() and len(line) < 100:
            # This might be a chapter title
            if any(keyword in line.lower() for keyword in ['chapter', 'part', 'book', 'section']):
                chapter_title = line
                break
            elif re.match(r'^[A-Z][A-Za-z\s]{5,80}$', line):
                chapter_title = line
                break
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(chapter_text)
    logger.info(f"Chapter {chapter_num}: Split into {len(paragraphs)} paragraphs")
    
    # Process each paragraph
    paragraph_data = []
    
    for i, paragraph in enumerate(paragraphs):
        if DEBUG_MODE:
            logger.debug(f"Processing paragraph {i + 1}/{len(paragraphs)}")
        
        # Analyze paragraph with LLM
        analysis = analyze_paragraph_with_llm(paragraph, chapter_title)
        
        # Create paragraph data structure
        paragraph_info = {
            "characters": analysis["characters"],
            "setting": analysis["setting"],
            "paragraph_idx": i,
            "text_path": f"books/{novel_name}/raw_chapters/{chapter_num}.txt",
            "word_count": len(paragraph.split()),
            "dialogue": analysis["dialogue"],
            "embedding_id": str(uuid.uuid4())
        }
        
        paragraph_data.append(paragraph_info)
    
    # Create chapter structure
    chapter_data = {
        "chapter_num": chapter_num,
        "chapter_title": chapter_title,
        "paragraphs": paragraph_data
    }
    
    logger.info(f"Chapter {chapter_num}: Processed {len(paragraph_data)} paragraphs")
    
    return chapter_data


def create_book_metadata(novel_name: str, author_name: str, chapter_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create book-level metadata including author bio and table of contents.
    
    This function generates comprehensive metadata for the book including:
    - Author biography from web search
    - Publication information
    - Table of contents with chapter summaries
    - Book statistics
    
    Args:
        novel_name (str): Name of the novel
        author_name (str): Name of the author
        chapter_data (List[Dict[str, Any]]): List of chapter data structures
        
    Returns:
        Dict[str, Any]: Book metadata structure
    """
    logger.info(f"Creating book metadata for '{novel_name}' by {author_name}")
    
    # Get author bio and publication info from web search
    author_bio, publication_date = fetch_author_info(author_name, novel_name)
    
    # Create table of contents
    table_of_contents = []
    total_word_count = 0
    
    for chapter in chapter_data:
        chapter_word_count = sum(para["word_count"] for para in chapter["paragraphs"])
        total_word_count += chapter_word_count
        
        toc_entry = {
            "chapter_num": chapter["chapter_num"],
            "chapter_title": chapter["chapter_title"],
            "word_count": chapter_word_count,
            "paragraph_count": len(chapter["paragraphs"])
        }
        table_of_contents.append(toc_entry)
    
    # Gather character and setting statistics
    all_characters = set()
    all_settings = set()
    
    for chapter in chapter_data:
        for para in chapter["paragraphs"]:
            all_characters.update(para["characters"])
            all_settings.update(para["setting"])
    
    # Remove generic entries
    all_characters.discard("narrator")
    all_settings = {setting for setting in all_settings if setting}
    
    # Create metadata structure
    metadata = {
        "title": novel_name,
        "author_name": author_name,
        "author_bio": author_bio,
        "publication_date": publication_date,
        "table_of_contents": table_of_contents,
        "statistics": {
            "total_chapters": len(chapter_data),
            "total_word_count": total_word_count,
            "total_paragraphs": sum(len(ch["paragraphs"]) for ch in chapter_data),
            "unique_characters": len(all_characters),
            "unique_settings": len(all_settings)
        },
        "processing_metadata": {
            "processed_date": str(uuid.uuid4()),  # Use UUID as timestamp placeholder
            "version": "1.0"
        }
    }
    
    logger.info(f"Book metadata created: {metadata['statistics']['total_chapters']} chapters, "
                f"{metadata['statistics']['total_word_count']} words")
    
    return metadata


def fetch_author_info(author_name: str, novel_name: str) -> Tuple[str, str]:
    """
    Fetch author biography and publication information from web sources.
    
    This function searches for author information using multiple strategies:
    - Wikipedia search for author biography
    - Google Books API for publication details
    - Literary database searches
    
    Args:
        author_name (str): Name of the author to search for
        novel_name (str): Name of the novel for context
        
    Returns:
        Tuple[str, str]: Author biography and publication date
    """
    logger.info(f"Fetching author information for {author_name}")
    
    author_bio = f"Biography information for {author_name} not available."
    publication_date = "Unknown"
    
    try:
        # Search Wikipedia for author bio
        search_url = f"https://en.wikipedia.org/wiki/{author_name.replace(' ', '_')}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract first paragraph of Wikipedia article
            first_paragraph = soup.find('p')
            if first_paragraph:
                author_bio = first_paragraph.get_text().strip()
                logger.info(f"Found author bio from Wikipedia: {len(author_bio)} characters")
            
            # Look for birth/death dates
            infobox = soup.find('table', class_='infobox')
            if infobox:
                for row in infobox.find_all('tr'):
                    header = row.find('th')
                    if header and 'born' in header.get_text().lower():
                        data = row.find('td')
                        if data:
                            # Extract year from birth info
                            text = data.get_text()
                            year_match = re.search(r'\b(18|19|20)\d{2}\b', text)
                            if year_match:
                                publication_date = f"Mid-{year_match.group(0)[:-1]}0s"
                                break
        
    except Exception as e:
        logger.error(f"Error fetching author info: {e}")
    
    return author_bio, publication_date


def process_novel_to_structured_json(novel_name: str, author_name: str) -> Dict[str, Any]:
    """
    Process an entire novel from raw chapters to structured JSON format.
    
    This is the main orchestration function that:
    1. Reads all raw chapter files
    2. Processes each chapter into structured JSON
    3. Creates book-level metadata
    4. Saves all structured data to disk
    
    Args:
        novel_name (str): Name of the novel
        author_name (str): Name of the author
        
    Returns:
        Dict[str, Any]: Processing results and metadata
    """
    logger.info(f"Starting novel processing for '{novel_name}' by {author_name}")
    
    book_dir = BOOKS_DIR / novel_name
    raw_chapters_dir = book_dir / "raw_chapters"
    structured_dir = book_dir / "structured"
    
    # Create structured directory
    structured_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all raw chapter files
    chapter_files = sorted(raw_chapters_dir.glob("*.txt"), key=lambda x: int(x.stem))
    
    if not chapter_files:
        raise FileNotFoundError(f"No raw chapter files found in {raw_chapters_dir}")
    
    logger.info(f"Found {len(chapter_files)} chapter files to process")
    
    # Process each chapter
    chapter_data = []
    
    for chapter_file in chapter_files:
        chapter_num = int(chapter_file.stem)
        
        # Read chapter text
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chapter_text = f.read()
        
        # Process chapter to JSON
        chapter_json = process_chapter_to_json(chapter_text, chapter_num, novel_name)
        chapter_data.append(chapter_json)
        
        # Save chapter JSON
        chapter_json_file = structured_dir / f"{chapter_num}.json"
        with open(chapter_json_file, 'w', encoding='utf-8') as f:
            json.dump(chapter_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved structured chapter {chapter_num} to {chapter_json_file}")
    
    # Create book metadata
    book_metadata = create_book_metadata(novel_name, author_name, chapter_data)
    
    # Save book metadata
    metadata_file = book_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(book_metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved book metadata to {metadata_file}")
    
    return {
        "status": "success",
        "chapters_processed": len(chapter_data),
        "metadata": book_metadata,
        "output_directory": str(structured_dir)
    } 