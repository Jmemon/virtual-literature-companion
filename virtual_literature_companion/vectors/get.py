"""
IndexRetriever for performing similarity search over vector indexes.

This module provides functionality to search across sentence-level, paragraph-level,
and generic word-segment indexes for books using ChromaDB.
"""

from enum import Enum
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from ..constants import VECTOR_EMBEDDING_MODEL, VECTOR_DB_PATH
from .utils import ChromaInterface


class CollectionType(str, Enum):
    """Enum for different collection types."""
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    GENERIC = "generic"


class IndexRetriever(ChromaInterface):
    """Performs similarity search over multiple vector collections for a book."""
    
    def __init__(self, novel_name: str, db_path: str = str(VECTOR_DB_PATH), 
                 model_name: str = VECTOR_EMBEDDING_MODEL):
        """
        Initialize retriever for a specific novel.
        
        Args:
            novel_name: Name of the novel (will be normalized for collection names)
            db_path: Path to ChromaDB storage directory
            model_name: Name of the sentence transformer model to use
        """
        super().__init__(db_path, model_name)
        self.novel_name = novel_name
        self.normalized_name = novel_name.lower().replace(' ', '_')
        self.collections = {}
        
        # Collection names based on create_indexes.py patterns
        self.collection_names = {
            CollectionType.SENTENCES: f"sentences_{self.normalized_name}",
            CollectionType.PARAGRAPHS: f"paragraphs_{self.normalized_name}",
            CollectionType.GENERIC: f"generic_200w_{self.normalized_name}"  # Default 200-word chunks
        }
    
    def _get_collection_by_type(self, collection_type: CollectionType):
        """Get a specific collection by type."""
        if collection_type not in self.collections:
            collection_name = self.collection_names[collection_type]
            try:
                self.collections[collection_type] = self._get_collection(collection_name)
            except Exception as e:
                raise ValueError(f"Collection '{collection_name}' not found. Has the index been created?") from e
        
        return self.collections[collection_type]
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string."""
        self._initialize_model()
        embedding = self.embedding_model.encode([query])
        return embedding[0].tolist()
    
    def search(self, collection_type: CollectionType, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search a specific collection type.
        """
        collection = self._get_collection_by_type(collection_type)
        query_embedding = self._embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results
    
    def search_sentences(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search the sentence-level index.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results with ids, distances, metadatas, documents
        """
        return self.search(CollectionType.SENTENCES, query, n_results)
    
    def search_paragraphs(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search the paragraph-level index.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results with ids, distances, metadatas, documents
        """
        return self.search(CollectionType.PARAGRAPHS, query, n_results)
    
    def search_generic(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search the generic word-segment index.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results with ids, distances, metadatas, documents
        """
        return self.search(CollectionType.GENERIC, query, n_results)
    
    def search_all(self, query: str, n_results_per_type: int = 3) -> Dict[str, Dict[str, Any]]:
        """
        Search all three collection types and return combined results.
        
        Args:
            query: Search query
            n_results_per_type: Number of results to return from each collection type
            
        Returns:
            Dictionary with keys 'sentences', 'paragraphs', 'generic' containing respective results
        """
        results = {}
        
        try:
            results[CollectionType.SENTENCES] = self.search_sentences(query, n_results_per_type)
        except ValueError:
            results[CollectionType.SENTENCES] = None
            
        try:
            results[CollectionType.PARAGRAPHS] = self.search_paragraphs(query, n_results_per_type)
        except ValueError:
            results[CollectionType.PARAGRAPHS] = None
            
        try:
            results[CollectionType.GENERIC] = self.search_generic(query, n_results_per_type)
        except ValueError:
            results[CollectionType.GENERIC] = None
            
        return results
    
    def get_available_collections(self) -> List[str]:
        """
        Check which collections are available for this novel.
        
        Returns:
            List of available collection types
        """
        available = []
        for collection_type, collection_name in self.collection_names.items():
            try:
                self._get_collection(collection_name)
                available.append(collection_type)
            except Exception:
                continue
                
        return available