"""
Shared utilities for vector indexing and retrieval.

This module provides base classes and common functionality for ChromaDB operations.
"""

from typing import List
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ..constants import VECTOR_EMBEDDING_MODEL, VECTOR_DB_PATH


class ChromaInterface:
    """Base class for ChromaDB operations with shared functionality."""
    
    def __init__(self, db_path: str = str(VECTOR_DB_PATH), model_name: str = VECTOR_EMBEDDING_MODEL):
        """
        Initialize ChromaDB interface.
        
        Args:
            db_path: Path to ChromaDB storage directory
            model_name: Name of the sentence transformer model to use
        """
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.embedding_model = None
        self.chroma_client = None
        
    def _initialize_model(self):
        """Lazy initialization of the embedding model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.model_name)
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
    
    def _create_collection(self, collection_name: str):
        """Create or get ChromaDB collection."""
        self._initialize_client()
        try:
            collection = self.chroma_client.get_collection(collection_name)
        except Exception:
            collection = self.chroma_client.create_collection(collection_name)
        return collection
    
    def _get_collection(self, collection_name: str):
        """Get an existing ChromaDB collection."""
        self._initialize_client()
        return self.chroma_client.get_collection(collection_name)