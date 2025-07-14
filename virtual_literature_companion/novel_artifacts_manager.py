"""
Artifact management for the Virtual Literature Companion system.

This module provides a manager class for handling the saving of various
novel artifacts in a consistent directory structure.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from .constants import BOOKS_DIR

class NovelArtifactsManager:
    """
    Manager for saving novel artifacts in the appropriate directory structure.
    """
    
    def __init__(self, novel_name: str):
        """
        Initialize the artifacts manager.
        
        Args:
            novel_name (str): Name of the novel
        """
        self.novel_dir = BOOKS_DIR / novel_name
        self.raw_dir = self.novel_dir / "raw"
        self.structured_dir = self.novel_dir / "structured"
        self.indexes_dir = self.novel_dir / "indexes"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.structured_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
    
    def save_raw_file(self, filename: str, text: str) -> Path:
        """
        Save a raw text file (e.g., front/back matter or chapter).
        
        Args:
            filename (str): Name of the file (e.g., 'title.txt' or '1.txt')
            text (str): Text content to save
        
        Returns:
            Path: Path to the saved file
        """
        file_path = self.raw_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return file_path
    
    def save_structured_chapter(self, chapter_num: int, chapter_data: Dict[str, Any]) -> Path:
        """
        Save a structured chapter JSON file.
        
        Args:
            chapter_num (int): Chapter number
            chapter_data (Dict[str, Any]): Chapter data to save
        
        Returns:
            Path: Path to the saved file
        """
        file_path = self.structured_dir / f"{chapter_num}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chapter_data, f, indent=2, ensure_ascii=False)
        return file_path
    
    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Save the book metadata JSON file.
        
        Args:
            metadata (Dict[str, Any]): Metadata to save
        
        Returns:
            Path: Path to the saved file
        """
        file_path = self.novel_dir / "metadata.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return file_path 