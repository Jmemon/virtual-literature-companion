"""
Vector index creation for the Virtual Literature Companion system.

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

from ..novel_artifacts_manager import NovelArtifactsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    artifacts_manager = NovelArtifactsManager(novel_name)
    structured_dir = artifacts_manager.structured_dir
    indexes_dir = artifacts_manager.indexes_dir
    
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
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(indexes_dir))
    
    # Create siloed index
    siloed_texts, siloed_metadatas, siloed_ids, siloed_embeddings = prepare_siloed_data(chapter_data, embedding_model)
    siloed_collection = client.get_or_create_collection(name=f"{novel_name}_siloed")
    siloed_collection.add(embeddings=siloed_embeddings, documents=siloed_texts, metadatas=siloed_metadatas, ids=siloed_ids)
    
    # Create contextual index
    contextual_texts, contextual_metadatas, contextual_ids, contextual_embeddings = prepare_contextual_data(chapter_data, embedding_model)
    contextual_collection = client.get_or_create_collection(name=f"{novel_name}_contextual")
    contextual_collection.add(embeddings=contextual_embeddings, documents=contextual_texts, metadatas=contextual_metadatas, ids=contextual_ids)
    
    # Calculate statistics
    total_paragraphs = sum(len(chapter["paragraphs"]) for chapter in chapter_data)
    
    results = {
        "status": "success",
        "novel_name": novel_name,
        "embedding_model": embedding_model,
        "indexes_created": [
            f"{novel_name}_siloed",
            f"{novel_name}_contextual"
        ],
        "statistics": {
            "total_chapters": len(chapter_data),
            "total_paragraphs": total_paragraphs,
            "index_directory": str(indexes_dir)
        }
    }
    
    logger.info(f"Successfully created vector indexes for '{novel_name}'")
    logger.info(f"Total paragraphs indexed: {total_paragraphs}")
    
    return results


def prepare_siloed_data(chapter_data: List[Dict[str, Any]], embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[List[float]]]:
    """
    Prepare data for siloed index.
    
    Returns texts, metadatas, ids, embeddings
    """
    encoder = SentenceTransformer(embedding_model)
    texts = []
    metadatas = []
    ids = []
    for chapter in chapter_data:
        for para in chapter['paragraphs']:
            text = para['text']
            texts.append(text)
            ids.append(para['embedding_id'])
            metadata = {
                'chapter_num': chapter['chapter_num'],
                'chapter_title': chapter['chapter_title'],
                'paragraph_idx': para['paragraph_idx'],
                'word_count': para['word_count'],
                'dialogue': para['dialogue'],
                'characters': json.dumps(para['characters']),
                'setting': json.dumps(para['setting'])
            }
            metadatas.append(metadata)
    embeddings = encoder.encode(texts).tolist()
    return texts, metadatas, ids, embeddings


def prepare_contextual_data(chapter_data: List[Dict[str, Any]], embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[List[float]]]:
    """
    Prepare data for contextual index.
    
    Returns contextual_texts, metadatas, ids, embeddings
    """
    encoder = SentenceTransformer(embedding_model)
    texts = []
    metadatas = []
    ids = []
    for chapter in chapter_data:
        for para in chapter['paragraphs']:
            text = para['text']
            contextual_text = create_contextual_text(text, para['characters'], para['setting'])
            texts.append(contextual_text)
            ids.append(para['embedding_id'])
            metadata = {
                'chapter_num': chapter['chapter_num'],
                'chapter_title': chapter['chapter_title'],
                'paragraph_idx': para['paragraph_idx'],
                'word_count': para['word_count'],
                'dialogue': para['dialogue'],
                'characters': json.dumps(para['characters']),
                'setting': json.dumps(para['setting']),
                'original_text': text[:200] + '...' if len(text) > 200 else text
            }
            metadatas.append(metadata)
    embeddings = encoder.encode(texts).tolist()
    return texts, metadatas, ids, embeddings


def create_contextual_text(paragraph_text: str, characters: List[str], setting: List[str]) -> str:
    context_parts = []
    if characters and characters != ["narrator"]:
        context_parts.append(f"Characters: {', '.join(characters)}")
    if setting:
        context_parts.append(f"Setting: {', '.join(setting)}")
    if context_parts:
        return " | ".join(context_parts) + " | " + paragraph_text
    return paragraph_text 