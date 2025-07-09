"""
Constants and configuration for the Virtual Literature Companion system.

This module defines all the essential paths and configuration values used throughout
the application for processing literature PDFs, creating structured data, and
generating embeddings.
"""

from pathlib import Path
import os

# Core directory paths
REPO_DIR = Path(__file__).parent.parent.absolute()
BOOKS_DIR = REPO_DIR / "books"
SRC_DIR = REPO_DIR / "virtual_literature_companion"

# Ensure books directory exists
BOOKS_DIR.mkdir(exist_ok=True)

# Default configuration for processing
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Processing configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # For multithreading PDF processing
TESSERACT_CONFIG = "--psm 6"  # Page segmentation mode for OCR 