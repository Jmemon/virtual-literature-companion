"""
Main ingestion module for the Virtual Literature Companion system.

This module provides the primary entry point for processing PDF books through
the complete pipeline:

1. PDF to text conversion (multi-threaded, OCR-capable)
2. Text parsing and structuring into JSON format
3. Paragraph-level literary analysis (characters, settings, dialogue)
4. Book metadata generation (author bio, table of contents)
5. Vector index creation (siloed and contextual embeddings)

The `ingest_book_pdf` function is the main orchestrator that handles error
recovery, progress tracking, and cleanup, providing a robust end-to-end
book processing experience.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from .constants import DEBUG_MODE, BOOKS_DIR
from .processors.pdf2txt import extract_and_clean_pages, validate_pdf_file
from .processors.process_novel_text import process_extracted_pages
from .processors.parse_novel_text import process_chapters_to_structured
from .indexes.create_vector_indexes import create_vector_indexes
from .novel_artifacts_manager import NovelArtifactsManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_book_pdf(
    pdf_path: str, 
    novel_name: str, 
    author_name: str,
    skip_existing: bool = False,
    clean_on_error: bool = True
) -> Dict[str, Any]:
    """
    Complete end-to-end ingestion of a PDF book into the Virtual Literature Companion system.
    
    This function orchestrates the entire book processing pipeline, handling:
    - PDF validation and text extraction
    - Chapter detection and text structuring
    - Literary analysis with character and setting identification
    - Metadata generation including author biography
    - Vector index creation for semantic search
    - Error handling and recovery
    - Progress tracking and logging
    
    Processing Steps:
    1. Validate PDF file and input parameters
    2. Extract text from PDF using multi-threaded processing
    3. Detect chapter boundaries and split content
    4. Create structured JSON with paragraph-level analysis
    5. Generate book metadata and author information
    6. Create vector embeddings for semantic search
    7. Validate and test the created indexes
    
    Args:
        pdf_path (str): Path to the PDF file to process
        novel_name (str): Name of the novel (used for organizing output)
        author_name (str): Name of the author (used for metadata lookup)
        skip_existing (bool): If True, skip processing if output already exists
        clean_on_error (bool): If True, clean up partial results on error
        
    Returns:
        Dict[str, Any]: Processing results containing:
            - status: "success" or "error"
            - message: Descriptive message about the result
            - novel_name: Name of the processed novel
            - author_name: Name of the author
            - statistics: Processing statistics (chapters, paragraphs, etc.)
            - output_files: Paths to created files and directories
            - error_details: Error information if status is "error"
            
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If input parameters are invalid
        Exception: For unexpected processing errors
    """
    logger.info(f"Starting book ingestion for '{novel_name}' by {author_name}")
    logger.info(f"PDF source: {pdf_path}")

    novel_name = novel_name.lower().replace(" ", "_")
    author_name = author_name.lower().replace(" ", "_")
    
    # Validate inputs
    validation_result = _validate_inputs(pdf_path, novel_name, author_name)
    if validation_result["status"] == "error":
        return validation_result
    
    pdf_path = Path(pdf_path)
    book_dir = BOOKS_DIR / novel_name
    
    # Check if processing should be skipped
    if skip_existing and _check_existing_processing(book_dir):
        logger.info(f"Skipping '{novel_name}' - already processed")
        return {
            "status": "success",
            "message": f"Book '{novel_name}' already processed (skipped)",
            "novel_name": novel_name,
            "author_name": author_name,
            "skipped": True
        }
    
    # Create progress tracker
    progress = ProcessingProgress(novel_name)
    
    artifacts_manager = NovelArtifactsManager(novel_name)
    
    try:
        # Step 1: PDF extraction and cleaning
        logger.info("=" * 60)
        logger.info("STEP 1: PDF extraction and cleaning")
        logger.info("=" * 60)
        
        progress.start_step("pdf_extraction")
        
        if not validate_pdf_file(str(pdf_path)):
            raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")
        
        page_texts, total_pages = extract_and_clean_pages(str(pdf_path))
        
        if not page_texts:
            raise ValueError("No text could be extracted from the PDF")
        
        progress.complete_step("pdf_extraction", {
            "pages_extracted": len(page_texts),
            "total_characters": sum(len(text) for text in page_texts.values())
        })
        
        logger.info(f"✓ PDF extraction and cleaning complete: {len(page_texts)} pages")
        
        # Step 2: Page processing and raw text creation
        front_matter, back_matter, chapter_texts = process_extracted_pages(page_texts, novel_name, total_pages)
        
        # Save raw files using manager
        for filename, text in front_matter.items():
            artifacts_manager.save_raw_file(filename, text)
        for filename, text in back_matter.items():
            artifacts_manager.save_raw_file(filename, text)
        for i, text in enumerate(chapter_texts, 1):
            artifacts_manager.save_raw_file(f"{i}.txt", text)
        
        progress.complete_step("page_processing", {
            "chapters_created": len(chapter_texts),
            "front_matter_files": len(front_matter),
            "back_matter_files": len(back_matter)
        })
        
        # Step 3: Text parsing and structuring
        chapter_data, book_metadata = process_chapters_to_structured(chapter_texts, author_name, novel_name)
        
        # Save structured files using manager
        for chapter in chapter_data:
            artifacts_manager.save_structured_chapter(chapter['chapter_num'], chapter)
        artifacts_manager.save_metadata(book_metadata)
        
        progress.complete_step("text_parsing", {
            "chapters_processed": len(chapter_data),
            "metadata_created": True
        })
        
        # Step 4: Vector index creation
        indexing_result = create_vector_indexes(novel_name)
        
        if indexing_result["status"] != "success":
            raise ValueError(f"Vector indexing failed: {indexing_result}")
        
        progress.complete_step("vector_indexing", {
            "indexes_created": len(indexing_result["indexes_created"]),
            "total_paragraphs": indexing_result["statistics"]["total_paragraphs"]
        })
        
        logger.info(f"✓ Vector indexing complete: {indexing_result['statistics']['total_paragraphs']} paragraphs indexed")
        
        # Step 5: Final validation
        logger.info("=" * 60)
        logger.info("STEP 5: Final validation")
        logger.info("=" * 60)
        
        progress.start_step("validation")
        
        validation_result = _validate_complete_processing(book_dir)
        
        if validation_result["status"] != "success":
            raise ValueError(f"Processing validation failed: {validation_result['message']}")
        
        progress.complete_step("validation", validation_result["files_created"])
        
        # Create final success result
        result = {
            "status": "success",
            "message": f"Book '{novel_name}' by {author_name} successfully ingested",
            "novel_name": novel_name,
            "author_name": author_name,
            "statistics": {
                "chapters": len(chapter_data),
                "paragraphs": indexing_result["statistics"]["total_paragraphs"],
                "total_characters": progress.get_step_data("pdf_extraction").get("total_characters", 0),
                "indexes_created": len(indexing_result["indexes_created"])
            },
            "output_files": {
                "book_directory": str(book_dir),
                "raw": str(book_dir / "raw"),
                "structured_data": str(book_dir / "structured"),
                "vector_indexes": str(book_dir / "indexes"),
                "metadata": str(book_dir / "metadata.json")
            },
            "processing_time": progress.get_total_time(),
            "index_collections": indexing_result["indexes_created"]
        }
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Novel: {novel_name}")
        logger.info(f"✓ Author: {author_name}")
        logger.info(f"✓ Chapters: {result['statistics']['chapters']}")
        logger.info(f"✓ Paragraphs: {result['statistics']['paragraphs']}")
        logger.info(f"✓ Processing time: {result['processing_time']:.2f} seconds")
        logger.info(f"✓ Output directory: {book_dir}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during book ingestion: {str(e)}")
        
        if DEBUG_MODE:
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Clean up partial results if requested
        if clean_on_error:
            _cleanup_partial_processing(book_dir)
        
        # Create error result
        error_result = {
            "status": "error",
            "message": f"Error ingesting book '{novel_name}': {str(e)}",
            "novel_name": novel_name,
            "author_name": author_name,
            "error_details": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_step": progress.get_current_step(),
                "partial_cleanup": clean_on_error
            }
        }
        
        if DEBUG_MODE:
            error_result["error_details"]["traceback"] = traceback.format_exc()
        
        return error_result


def _validate_inputs(pdf_path: str, novel_name: str, author_name: str) -> Dict[str, Any]:
    """
    Validate the input parameters for book ingestion.
    
    Args:
        pdf_path (str): Path to the PDF file
        novel_name (str): Name of the novel
        author_name (str): Name of the author
        
    Returns:
        Dict[str, Any]: Validation result
    """
    # Check PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return {
            "status": "error",
            "message": f"PDF file not found: {pdf_path}"
        }
    
    if not pdf_file.is_file():
        return {
            "status": "error",
            "message": f"Path is not a file: {pdf_path}"
        }
    
    if pdf_file.suffix.lower() != '.pdf':
        return {
            "status": "error",
            "message": f"File is not a PDF: {pdf_path}"
        }
    
    # Validate novel name
    if not novel_name or not novel_name.strip():
        return {
            "status": "error",
            "message": "Novel name cannot be empty"
        }
    
    # Validate author name
    if not author_name or not author_name.strip():
        return {
            "status": "error",
            "message": "Author name cannot be empty"
        }
    
    # Check for invalid characters in names
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        if char in novel_name:
            return {
                "status": "error",
                "message": f"Novel name contains invalid character: {char}"
            }
        if char in author_name:
            return {
                "status": "error",
                "message": f"Author name contains invalid character: {char}"
            }
    
    return {"status": "success"}


def _check_existing_processing(book_dir: Path) -> bool:
    """
    Check if a book has already been processed.
    
    Args:
        book_dir (Path): Book directory to check
        
    Returns:
        bool: True if already processed, False otherwise
    """
    if not book_dir.exists():
        return False
    
    # Check for required directories and files
    required_paths = [
        book_dir / "raw",
        book_dir / "structured",
        book_dir / "indexes",
        book_dir / "metadata.json"
    ]
    
    return all(path.exists() for path in required_paths)


def _validate_complete_processing(book_dir: Path) -> Dict[str, Any]:
    """
    Validate that all processing steps completed successfully.
    
    Args:
        book_dir (Path): Book directory to validate
        
    Returns:
        Dict[str, Any]: Validation result
    """
    files_created = {}
    
    # Check raw chapters
    raw_dir = book_dir / "raw"
    if raw_dir.exists():
        chapter_files = list(raw_dir.glob("*.txt"))
        files_created["raw"] = len(chapter_files)
    else:
        return {
            "status": "error",
            "message": "Raw chapters directory not found"
        }
    
    # Check structured data
    structured_dir = book_dir / "structured"
    if structured_dir.exists():
        json_files = list(structured_dir.glob("*.json"))
        files_created["structured_chapters"] = len(json_files)
    else:
        return {
            "status": "error",
            "message": "Structured data directory not found"
        }
    
    # Check metadata
    metadata_file = book_dir / "metadata.json"
    if metadata_file.exists():
        files_created["metadata"] = True
    else:
        return {
            "status": "error",
            "message": "Metadata file not found"
        }
    
    # Check indexes
    indexes_dir = book_dir / "indexes"
    if indexes_dir.exists():
        files_created["indexes_directory"] = True
    else:
        return {
            "status": "error",
            "message": "Indexes directory not found"
        }
    
    return {
        "status": "success",
        "files_created": files_created
    }


def _cleanup_partial_processing(book_dir: Path) -> None:
    """
    Clean up partial processing results.
    
    Args:
        book_dir (Path): Book directory to clean up
    """
    try:
        if book_dir.exists():
            import shutil
            shutil.rmtree(book_dir)
            logger.info(f"Cleaned up partial processing directory: {book_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up directory {book_dir}: {e}")


class ProcessingProgress:
    """
    Track processing progress and timing for book ingestion.
    
    This class provides detailed progress tracking including step timing,
    statistics collection, and error state management.
    """
    
    def __init__(self, novel_name: str):
        """
        Initialize progress tracking for a novel.
        
        Args:
            novel_name (str): Name of the novel being processed
        """
        self.novel_name = novel_name
        self.steps = {}
        self.current_step = None
        self.start_time = None
        
        import time
        self.start_time = time.time()
    
    def start_step(self, step_name: str) -> None:
        """
        Start tracking a processing step.
        
        Args:
            step_name (str): Name of the step to track
        """
        import time
        
        self.current_step = step_name
        self.steps[step_name] = {
            "start_time": time.time(),
            "status": "running",
            "data": {}
        }
        
        logger.info(f"Started step: {step_name}")
    
    def complete_step(self, step_name: str, data: Dict[str, Any] = None) -> None:
        """
        Complete tracking of a processing step.
        
        Args:
            step_name (str): Name of the step to complete
            data (Dict[str, Any]): Optional data to store with the step
        """
        import time
        
        if step_name in self.steps:
            self.steps[step_name]["end_time"] = time.time()
            self.steps[step_name]["status"] = "completed"
            self.steps[step_name]["duration"] = (
                self.steps[step_name]["end_time"] - 
                self.steps[step_name]["start_time"]
            )
            
            if data:
                self.steps[step_name]["data"] = data
            
            logger.info(f"Completed step: {step_name} ({self.steps[step_name]['duration']:.2f}s)")
    
    def get_current_step(self) -> Optional[str]:
        """
        Get the currently running step.
        
        Returns:
            Optional[str]: Name of the current step, or None if none running
        """
        return self.current_step
    
    def get_step_data(self, step_name: str) -> Dict[str, Any]:
        """
        Get data for a specific step.
        
        Args:
            step_name (str): Name of the step
            
        Returns:
            Dict[str, Any]: Step data
        """
        return self.steps.get(step_name, {}).get("data", {})
    
    def get_total_time(self) -> float:
        """
        Get total processing time.
        
        Returns:
            float: Total time in seconds
        """
        import time
        return time.time() - self.start_time if self.start_time else 0.0


def list_ingested_books() -> List[Dict[str, Any]]:
    """
    List all books that have been ingested into the system.
    
    This function scans the books directory and returns information about
    all successfully processed books, including their metadata and processing
    statistics.
    
    Returns:
        List[Dict[str, Any]]: List of book information dictionaries
    """
    import json
    
    books = []
    
    if not BOOKS_DIR.exists():
        return books
    
    for book_dir in BOOKS_DIR.iterdir():
        if book_dir.is_dir():
            metadata_file = book_dir / "metadata.json"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Add directory information
                    metadata["directory_path"] = str(book_dir)
                    metadata["directory_name"] = book_dir.name
                    
                    # Check for indexes
                    indexes_dir = book_dir / "indexes"
                    metadata["has_indexes"] = indexes_dir.exists()
                    
                    books.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Error reading metadata for {book_dir}: {e}")
    
    return books 