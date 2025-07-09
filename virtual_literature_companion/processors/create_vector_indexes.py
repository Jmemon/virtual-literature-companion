"""
Vector index creation processor for the Virtual Literature Companion system.

This module handles the creation of vector embeddings and indexes using ChromaDB:
- Siloed index: Direct paragraph text embeddings
- Contextual index: Text embeddings enhanced with character and setting context
- Persistent storage with proper metadata handling
- Efficient batch processing for large novels

The indexes enable semantic search capabilities across the novel's content,
allowing for character-specific queries, setting-based searches, and contextual
literary analysis.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ..constants import (
    BOOKS_DIR, 
    DEFAULT_EMBEDDING_MODEL, 
    DEFAULT_CHUNK_SIZE, 
    DEBUG_MODE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexCreator:
    """
    Vector index creator for literary content using ChromaDB.
    
    This class handles the creation and management of vector indexes for literary
    analysis, providing both siloed (content-only) and contextual (enhanced with
    character/setting information) embedding capabilities.
    """
    
    def __init__(self, novel_name: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the vector index creator.
        
        Args:
            novel_name (str): Name of the novel to process
            embedding_model (str): Name of the sentence transformer model to use
        """
        self.novel_name = novel_name
        self.embedding_model = embedding_model
        self.book_dir = BOOKS_DIR / novel_name
        self.indexes_dir = self.book_dir / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.indexes_dir))
        
        logger.info(f"Initialized vector index creator for '{novel_name}'")
    
    def create_siloed_index(self, chapter_data: List[Dict[str, Any]]) -> str:
        """
        Create a siloed index with direct paragraph text embeddings.
        
        This index contains embeddings of raw paragraph text without additional
        context, suitable for direct content matching and similarity searches.
        
        Args:
            chapter_data (List[Dict[str, Any]]): List of structured chapter data
            
        Returns:
            str: Collection ID for the siloed index
        """
        logger.info(f"Creating siloed index for '{self.novel_name}'")
        
        # Create or get collection
        collection_name = f"{self.novel_name}_siloed"
        
        try:
            # Delete existing collection if it exists
            self.client.delete_collection(collection_name)
        except Exception:
            pass  # Collection doesn't exist, which is fine
        
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"type": "siloed", "novel": self.novel_name}
        )
        
        # Prepare data for embedding
        texts = []
        metadatas = []
        ids = []
        
        for chapter in chapter_data:
            chapter_num = chapter["chapter_num"]
            chapter_title = chapter["chapter_title"]
            
            for paragraph in chapter["paragraphs"]:
                # Read paragraph text from raw file
                paragraph_text = self._read_paragraph_text(
                    chapter_num, paragraph["paragraph_idx"]
                )
                
                if paragraph_text:
                    texts.append(paragraph_text)
                    ids.append(paragraph["embedding_id"])
                    
                    # Create metadata
                    metadata = {
                        "chapter_num": chapter_num,
                        "chapter_title": chapter_title,
                        "paragraph_idx": paragraph["paragraph_idx"],
                        "word_count": paragraph["word_count"],
                        "dialogue": paragraph["dialogue"],
                        "characters": json.dumps(paragraph["characters"]),
                        "setting": json.dumps(paragraph["setting"]),
                        "text_path": paragraph["text_path"]
                    }
                    metadatas.append(metadata)
        
        # Create embeddings and add to collection in batches
        batch_size = 50
        total_paragraphs = len(texts)
        
        logger.info(f"Processing {total_paragraphs} paragraphs in batches of {batch_size}")
        
        for i in range(0, total_paragraphs, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.encoder.encode(batch_texts).tolist()
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            if DEBUG_MODE:
                logger.debug(f"Added batch {i//batch_size + 1}/{(total_paragraphs + batch_size - 1)//batch_size}")
        
        logger.info(f"Created siloed index with {total_paragraphs} paragraphs")
        
        return collection_name
    
    def create_contextual_index(self, chapter_data: List[Dict[str, Any]]) -> str:
        """
        Create a contextual index with character and setting enhanced embeddings.
        
        This index prepends character and setting information to the paragraph text
        before embedding, enabling context-aware searches that consider who is
        involved and where the action takes place.
        
        Args:
            chapter_data (List[Dict[str, Any]]): List of structured chapter data
            
        Returns:
            str: Collection ID for the contextual index
        """
        logger.info(f"Creating contextual index for '{self.novel_name}'")
        
        # Create or get collection
        collection_name = f"{self.novel_name}_contextual"
        
        try:
            # Delete existing collection if it exists
            self.client.delete_collection(collection_name)
        except Exception:
            pass  # Collection doesn't exist, which is fine
        
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"type": "contextual", "novel": self.novel_name}
        )
        
        # Prepare data for embedding
        texts = []
        metadatas = []
        ids = []
        
        for chapter in chapter_data:
            chapter_num = chapter["chapter_num"]
            chapter_title = chapter["chapter_title"]
            
            for paragraph in chapter["paragraphs"]:
                # Read paragraph text from raw file
                paragraph_text = self._read_paragraph_text(
                    chapter_num, paragraph["paragraph_idx"]
                )
                
                if paragraph_text:
                    # Create contextual text by prepending character and setting info
                    contextual_text = self._create_contextual_text(
                        paragraph_text, 
                        paragraph["characters"], 
                        paragraph["setting"]
                    )
                    
                    texts.append(contextual_text)
                    ids.append(paragraph["embedding_id"])
                    
                    # Create metadata
                    metadata = {
                        "chapter_num": chapter_num,
                        "chapter_title": chapter_title,
                        "paragraph_idx": paragraph["paragraph_idx"],
                        "word_count": paragraph["word_count"],
                        "dialogue": paragraph["dialogue"],
                        "characters": json.dumps(paragraph["characters"]),
                        "setting": json.dumps(paragraph["setting"]),
                        "text_path": paragraph["text_path"],
                        "original_text": paragraph_text[:200] + "..." if len(paragraph_text) > 200 else paragraph_text
                    }
                    metadatas.append(metadata)
        
        # Create embeddings and add to collection in batches
        batch_size = 50
        total_paragraphs = len(texts)
        
        logger.info(f"Processing {total_paragraphs} contextual paragraphs in batches of {batch_size}")
        
        for i in range(0, total_paragraphs, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.encoder.encode(batch_texts).tolist()
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            if DEBUG_MODE:
                logger.debug(f"Added contextual batch {i//batch_size + 1}/{(total_paragraphs + batch_size - 1)//batch_size}")
        
        logger.info(f"Created contextual index with {total_paragraphs} paragraphs")
        
        return collection_name
    
    def _read_paragraph_text(self, chapter_num: int, paragraph_idx: int) -> str:
        """
        Read the text of a specific paragraph from the raw chapter file.
        
        Args:
            chapter_num (int): Chapter number (1-indexed)
            paragraph_idx (int): Paragraph index within the chapter (0-indexed)
            
        Returns:
            str: The paragraph text, or empty string if not found
        """
        try:
            chapter_file = self.book_dir / "raw_chapters" / f"{chapter_num}.txt"
            
            if not chapter_file.exists():
                logger.error(f"Chapter file not found: {chapter_file}")
                return ""
            
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_text = f.read()
            
            # Split into paragraphs using the same logic as the parser
            paragraphs = self._split_into_paragraphs(chapter_text)
            
            if paragraph_idx < len(paragraphs):
                return paragraphs[paragraph_idx]
            else:
                logger.error(f"Paragraph index {paragraph_idx} out of range for chapter {chapter_num}")
                return ""
                
        except Exception as e:
            logger.error(f"Error reading paragraph {paragraph_idx} from chapter {chapter_num}: {e}")
            return ""
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using the same logic as the parser.
        
        This ensures consistency between the parser and the vector index creator.
        
        Args:
            text (str): Raw chapter text
            
        Returns:
            List[str]: List of paragraph texts
        """
        import re
        
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
    
    def _create_contextual_text(self, paragraph_text: str, characters: List[str], setting: List[str]) -> str:
        """
        Create contextual text by prepending character and setting information.
        
        This function creates an enhanced version of the paragraph text that includes
        contextual information about characters and settings, improving the quality
        of semantic search.
        
        Args:
            paragraph_text (str): Original paragraph text
            characters (List[str]): List of characters involved
            setting (List[str]): List of setting elements
            
        Returns:
            str: Contextually enhanced text
        """
        context_parts = []
        
        # Add character context
        if characters and characters != ["narrator"]:
            character_context = f"Characters: {', '.join(characters)}"
            context_parts.append(character_context)
        
        # Add setting context
        if setting:
            setting_context = f"Setting: {', '.join(setting)}"
            context_parts.append(setting_context)
        
        # Combine context with paragraph text
        if context_parts:
            context_prefix = " | ".join(context_parts) + " | "
            return context_prefix + paragraph_text
        else:
            return paragraph_text
    
    def test_index_query(self, collection_name: str, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Test a query against the specified index.
        
        This function provides a way to test and validate the created indexes
        by performing sample queries and returning results.
        
        Args:
            collection_name (str): Name of the collection to query
            query (str): Query text to search for
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results with metadata
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = self.encoder.encode([query]).tolist()
            
            # Perform search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying index {collection_name}: {e}")
            return []


def create_vector_indexes(novel_name: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Dict[str, Any]:
    """
    Create both siloed and contextual vector indexes for a novel.
    
    This is the main function that orchestrates the creation of both types of
    vector indexes for the specified novel. It reads the structured chapter data
    and creates ChromaDB collections for semantic search.
    
    Args:
        novel_name (str): Name of the novel to process
        embedding_model (str): Name of the sentence transformer model to use
        
    Returns:
        Dict[str, Any]: Results of the index creation process
        
    Raises:
        FileNotFoundError: If structured chapter data is not found
        ValueError: If chapter data is malformed
    """
    logger.info(f"Starting vector index creation for '{novel_name}'")
    
    # Initialize the index creator
    creator = VectorIndexCreator(novel_name, embedding_model)
    
    # Load structured chapter data
    book_dir = BOOKS_DIR / novel_name
    structured_dir = book_dir / "structured"
    
    if not structured_dir.exists():
        raise FileNotFoundError(f"Structured data directory not found: {structured_dir}")
    
    # Load all chapter JSON files
    chapter_files = sorted(structured_dir.glob("*.json"), key=lambda x: int(x.stem))
    
    if not chapter_files:
        raise FileNotFoundError(f"No structured chapter files found in {structured_dir}")
    
    chapter_data = []
    for chapter_file in chapter_files:
        try:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_json = json.load(f)
                chapter_data.append(chapter_json)
        except Exception as e:
            logger.error(f"Error loading chapter file {chapter_file}: {e}")
            continue
    
    if not chapter_data:
        raise ValueError("No valid chapter data found")
    
    logger.info(f"Loaded {len(chapter_data)} chapters for index creation")
    
    # Create siloed index
    siloed_collection = creator.create_siloed_index(chapter_data)
    
    # Create contextual index
    contextual_collection = creator.create_contextual_index(chapter_data)
    
    # Test both indexes with a sample query
    sample_query = "main character dialogue"
    
    siloed_results = creator.test_index_query(siloed_collection, sample_query, 3)
    contextual_results = creator.test_index_query(contextual_collection, sample_query, 3)
    
    # Calculate statistics
    total_paragraphs = sum(len(chapter["paragraphs"]) for chapter in chapter_data)
    
    results = {
        "status": "success",
        "novel_name": novel_name,
        "embedding_model": embedding_model,
        "indexes_created": {
            "siloed": siloed_collection,
            "contextual": contextual_collection
        },
        "statistics": {
            "total_chapters": len(chapter_data),
            "total_paragraphs": total_paragraphs,
            "index_directory": str(creator.indexes_dir)
        },
        "sample_queries": {
            "query": sample_query,
            "siloed_results": len(siloed_results),
            "contextual_results": len(contextual_results)
        }
    }
    
    logger.info(f"Successfully created vector indexes for '{novel_name}'")
    logger.info(f"Siloed index: {siloed_collection}")
    logger.info(f"Contextual index: {contextual_collection}")
    logger.info(f"Total paragraphs indexed: {total_paragraphs}")
    
    return results 