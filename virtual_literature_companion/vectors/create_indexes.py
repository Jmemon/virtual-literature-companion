"""
Create vector indexes for books using different chunking strategies.

This module provides functionality to create sentence-level, paragraph-level,
and generic (200-word overlapping) indexes for books using ChromaDB and 
sentence transformers for embeddings.
"""

from typing import List
from pathlib import Path
import uuid

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..types import Book
from ..constants import VECTOR_EMBEDDING_MODEL, VECTOR_DB_PATH
from .chunks import ChunkData, TextChunker


class IndexCreator:
    """Base class for creating vector indexes using ChromaDB."""
    
    def __init__(self, db_path: str = str(VECTOR_DB_PATH), model_name: str = VECTOR_EMBEDDING_MODEL):
        """Initialize the index creator with embedding model and database path."""
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.embedding_model = None
        self.chroma_client = None
        
    def _initialize_model(self):
        """Lazy initialization of the embedding model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.model_name}")
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
        except ValueError:
            collection = self.chroma_client.create_collection(collection_name)
        return collection
    
    def _embed_chunks(self, chunks: List[ChunkData]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        self._initialize_model()
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def _store_chunks(self, collection, chunks: List[ChunkData], embeddings: List[List[float]]):
        """Store chunks and embeddings in ChromaDB collection."""
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )


def create_sentence_index(book: Book, db_path: str = str(VECTOR_DB_PATH)) -> str:
    """
    Create a sentence-level vector index for a book.
    
    Args:
        book: Book object containing sections with clean_text
        db_path: Path to ChromaDB storage directory
        
    Returns:
        Collection name of the created index
    """
    creator = IndexCreator(db_path)
    collection_name = f"sentences_{book.title.lower().replace(' ', '_')}"
    
    print(f"Creating sentence index for: {book.title}")
    
    # Extract all chunks from book sections
    all_chunks = []
    for section_idx, section in enumerate(tqdm(book.sections, desc="Processing sections")):
        clean_text = section.get('clean_text', '')
        if not clean_text:
            continue
            
        section_type = section.get('section_type', 'unknown')
        
        sentence_chunks = TextChunker.chunk_into_sentences(
            clean_text, section_idx, section_type, 
            book.title, book.author_name
        )
        all_chunks.extend(sentence_chunks)
    
    if not all_chunks:
        print("No text chunks found to index")
        return collection_name
    
    print(f"Generated {len(all_chunks)} sentence chunks")
    
    # Create embeddings and store
    collection = creator._create_collection(collection_name)
    embeddings = creator._embed_chunks(all_chunks)
    creator._store_chunks(collection, all_chunks, embeddings)
    
    print(f"Sentence index created successfully: {collection_name}")
    return collection_name


def create_paragraph_index(book: Book, db_path: str = str(VECTOR_DB_PATH)) -> str:
    """
    Create a paragraph-level vector index for a book.
    
    Args:
        book: Book object containing sections with clean_text
        db_path: Path to ChromaDB storage directory
        
    Returns:
        Collection name of the created index
    """
    creator = IndexCreator(db_path)
    collection_name = f"paragraphs_{book.title.lower().replace(' ', '_')}"
    
    print(f"Creating paragraph index for: {book.title}")
    
    # Extract all chunks from book sections
    all_chunks = []
    for section_idx, section in enumerate(tqdm(book.sections, desc="Processing sections")):
        clean_text = section.get('clean_text', '')
        if not clean_text:
            continue
            
        section_type = section.get('section_type', 'unknown')
        
        paragraph_chunks = TextChunker.chunk_into_paragraphs(
            clean_text, section_idx, section_type,
            book.title, book.author_name
        )
        all_chunks.extend(paragraph_chunks)
    
    if not all_chunks:
        print("No text chunks found to index")
        return collection_name
    
    print(f"Generated {len(all_chunks)} paragraph chunks")
    
    # Create embeddings and store
    collection = creator._create_collection(collection_name)
    embeddings = creator._embed_chunks(all_chunks)
    creator._store_chunks(collection, all_chunks, embeddings)
    
    print(f"Paragraph index created successfully: {collection_name}")
    return collection_name


def create_generic_index(book: Book, db_path: str = str(VECTOR_DB_PATH), 
                        chunk_size: int = 200, overlap: int = 50) -> str:
    """
    Create a generic overlapping word-segment vector index for a book.
    
    Args:
        book: Book object containing sections with clean_text
        db_path: Path to ChromaDB storage directory
        chunk_size: Number of words per chunk (default 200)
        overlap: Number of overlapping words between chunks (default 50)
        
    Returns:
        Collection name of the created index
    """
    creator = IndexCreator(db_path)
    collection_name = f"generic_{chunk_size}w_{book.title.lower().replace(' ', '_')}"
    
    print(f"Creating generic {chunk_size}-word index for: {book.title}")
    
    # Extract all chunks from book sections
    all_chunks = []
    for section_idx, section in enumerate(tqdm(book.sections, desc="Processing sections")):
        clean_text = section.get('clean_text', '')
        if not clean_text:
            continue
            
        section_type = section.get('section_type', 'unknown')
        
        generic_chunks = TextChunker.chunk_into_overlapping_segments(
            clean_text, section_idx, section_type,
            book.title, book.author_name, chunk_size, overlap
        )
        all_chunks.extend(generic_chunks)
    
    if not all_chunks:
        print("No text chunks found to index")
        return collection_name
    
    print(f"Generated {len(all_chunks)} generic chunks")
    
    # Create embeddings and store
    collection = creator._create_collection(collection_name)
    embeddings = creator._embed_chunks(all_chunks)
    creator._store_chunks(collection, all_chunks, embeddings)
    
    print(f"Generic index created successfully: {collection_name}")
    return collection_name