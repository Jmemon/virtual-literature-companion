"""
PDF preprocessing agent for book extraction and analysis.

This agent handles:
- Table of contents extraction
- Text chunking with chapter awareness
- Theme and emotion analysis
- Metadata extraction
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
import json

from ..config import settings, get_prompt_template
from ..models.schemas import (
    BookMetadata,
    TableOfContents,
    TextChunk,
    PreprocessorResult
)
from ..models.state import ProcessingState
from ..processors.pdf_extractor import PDFExtractor


class PDFPreprocessorAgent:
    """
    Agent responsible for preprocessing PDF books.
    
    This agent:
    1. Extracts raw text and metadata from PDFs
    2. Identifies and structures table of contents
    3. Chunks text intelligently with chapter boundaries
    4. Analyzes themes and emotional content
    5. Prepares data for the persona agent
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.llm_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.1,  # Low temperature for consistent extraction
            api_key=settings.openai_api_key
        )
        self.pdf_extractor = PDFExtractor()
        
    async def process_book(self, state: ProcessingState) -> ProcessingState:
        """
        Main processing function for the agent.
        
        Args:
            state: Current processing state
            
        Returns:
            Updated processing state
        """
        try:
            logger.info(f"Starting PDF processing for: {state.file_path}")
            state.update_progress(0.1, "Extracting PDF content")
            
            # Extract PDF content
            extraction_results = self.pdf_extractor.extract_from_file(state.file_path)
            
            state.extracted_text = extraction_results["full_text"]
            state.metadata = extraction_results["metadata"]
            state.update_progress(0.3, "PDF extraction complete")
            
            # Enhance table of contents with LLM if needed
            toc = extraction_results["table_of_contents"]
            if toc.total_chapters < 3:  # Likely incomplete TOC
                logger.info("Enhancing TOC with LLM analysis")
                toc = await self._enhance_toc_with_llm(
                    extraction_results["page_texts"][:20],  # First 20 pages
                    toc
                )
            state.table_of_contents = toc
            state.update_progress(0.5, "Table of contents extracted")
            
            # Analyze and enhance chunks
            chunks = extraction_results["chunks"]
            book_id = state.metadata.book_id if state.metadata else ""
            enhanced_chunks = await self._enhance_chunks(chunks, book_id)
            state.chunks = enhanced_chunks
            state.update_progress(0.8, "Text analysis complete")
            
            # Generate chapter summaries
            await self._generate_chapter_summaries(state)
            state.update_progress(0.9, "Chapter summaries generated")
            
            state.processing_status = "completed"
            state.update_progress(1.0, "Processing complete")
            
            book_title = state.metadata.title if state.metadata else "Unknown"
            logger.info(f"Successfully processed book: {book_title}")
            return state
            
        except Exception as e:
            logger.error(f"Error processing book: {e}")
            state.errors.append(str(e))
            state.processing_status = "failed"
            return state
            
    async def _enhance_toc_with_llm(self, early_pages: List[Dict[str, Any]], 
                                   initial_toc: TableOfContents) -> TableOfContents:
        """
        Use LLM to enhance table of contents extraction.
        
        Args:
            early_pages: First pages of the book
            initial_toc: Initial TOC from basic extraction
            
        Returns:
            Enhanced TableOfContents
        """
        # Combine early pages text
        combined_text = "\n\n".join([
            f"[Page {page['page_num']}]\n{page['text'][:1000]}"  # First 1000 chars per page
            for page in early_pages
        ])
        
        prompt = get_prompt_template(
            "preprocessor",
            "toc_extraction",
            text=combined_text
        )
        
        messages = [
            SystemMessage(content="You are an expert at analyzing book structure."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Parse LLM response
            toc_data = json.loads(response.content)
            
            chapters = []
            for chapter_data in toc_data.get("chapters", []):
                chapters.append(TableOfContents.Chapter(
                    chapter_number=chapter_data.get("number"),
                    title=chapter_data.get("title", ""),
                    start_page=chapter_data.get("start_page"),
                    end_page=chapter_data.get("end_page"),
                    summary=chapter_data.get("summary"),
                    themes=chapter_data.get("themes", [])
                ))
                
            return TableOfContents(
                chapters=chapters,
                total_chapters=len([c for c in chapters if c.chapter_number]),
                has_prologue=any("prologue" in c.title.lower() for c in chapters),
                has_epilogue=any("epilogue" in c.title.lower() for c in chapters)
            )
            
        except Exception as e:
            logger.warning(f"Failed to enhance TOC with LLM: {e}")
            return initial_toc
            
    async def _enhance_chunks(self, chunks: List[TextChunk], 
                            book_id: str) -> List[TextChunk]:
        """
        Enhance chunks with theme and emotion analysis.
        
        Args:
            chunks: Raw text chunks
            book_id: Book identifier
            
        Returns:
            Enhanced chunks with metadata
        """
        enhanced_chunks = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Analyze each chunk
            for chunk in batch:
                try:
                    analysis = await self._analyze_chunk(chunk.content)
                    
                    # Update chunk with analysis
                    chunk.themes = analysis.get("themes", [])
                    chunk.emotional_tone = analysis.get("emotional_tone")
                    chunk.key_quotes = analysis.get("key_quotes", [])
                    chunk.character_mentions = analysis.get("characters", [])
                    
                    # Mark potential spoilers
                    plot_points = analysis.get("plot_points", [])
                    for point in plot_points:
                        if point.get("is_major_event"):
                            point["spoiler_warning"] = True
                            
                    chunk.plot_points = plot_points
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze chunk {chunk.chunk_id}: {e}")
                    
                enhanced_chunks.append(chunk)
                
        return enhanced_chunks
        
    async def _analyze_chunk(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text chunk for themes and emotions.
        
        Args:
            text: Chunk text
            
        Returns:
            Analysis results
        """
        prompt = get_prompt_template(
            "preprocessor",
            "chunk_analysis",
            chunk=text[:2000]  # Limit text length
        )
        
        messages = [
            SystemMessage(content="You are a literary analyst. Be concise and specific."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            return json.loads(response.content)
        except:
            # Fallback to basic parsing
            return {
                "themes": [],
                "emotional_tone": "neutral",
                "key_quotes": [],
                "characters": [],
                "plot_points": []
            }
            
    async def _generate_chapter_summaries(self, state: ProcessingState):
        """
        Generate summaries for each chapter.
        
        Args:
            state: Current processing state
        """
        if not state.table_of_contents:
            return
            
        for chapter in state.table_of_contents.chapters:
            if chapter.summary:  # Skip if already has summary
                continue
                
            # Get chunks for this chapter
            chapter_chunks = [
                chunk for chunk in state.chunks
                if chunk.chapter_number == chapter.chapter_number
            ]
            
            if not chapter_chunks:
                continue
                
            # Combine first few chunks
            chapter_text = " ".join([
                chunk.content for chunk in chapter_chunks[:3]
            ])[:3000]  # Limit to 3000 chars
            
            prompt = f"""Provide a brief, spoiler-free summary of this chapter:

Chapter: {chapter.title}

Text excerpt:
{chapter_text}

Summary (2-3 sentences, no spoilers):"""
            
            messages = [
                SystemMessage(content="You are a helpful book summarizer. Never reveal plot twists or major events."),
                HumanMessage(content=prompt)
            ]
            
            try:
                response = await self.llm.ainvoke(messages)
                chapter.summary = response.content.strip()
            except Exception as e:
                logger.warning(f"Failed to generate summary for chapter {chapter.title}: {e}")
                
    def create_preprocessor_result(self, state: ProcessingState) -> PreprocessorResult:
        """
        Create a PreprocessorResult from the processing state.
        
        Args:
            state: Completed processing state
            
        Returns:
            PreprocessorResult object
        """
        upload_date = state.metadata.upload_date if state.metadata else datetime.utcnow()
        processing_time = (datetime.utcnow() - upload_date).total_seconds()
        
        return PreprocessorResult(
            book_metadata=state.metadata,
            table_of_contents=state.table_of_contents,
            chunks=state.chunks,
            processing_time=processing_time,
            errors=state.errors
        )


# Agent node function for LangGraph
async def pdf_preprocessor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node function for the PDF preprocessor agent in LangGraph.
    
    Args:
        state: Graph state dictionary
        
    Returns:
        Updated state dictionary
    """
    agent = PDFPreprocessorAgent()
    
    # Create processing state from graph state
    processing_state = ProcessingState(
        file_path=state.get("file_path"),
        file_content=state.get("file_content")
    )
    
    # Process the book
    updated_state = await agent.process_book(processing_state)
    
    # Update graph state with results
    state["book_metadata"] = updated_state.metadata
    state["table_of_contents"] = updated_state.table_of_contents
    state["processed_chunks"] = updated_state.chunks
    state["preprocessing_complete"] = True
    state["preprocessing_errors"] = updated_state.errors
    
    # Set agent output
    if "agent_outputs" not in state:
        state["agent_outputs"] = {}
        
    state["agent_outputs"]["pdf_preprocessor"] = agent.create_preprocessor_result(updated_state)
    
    return state