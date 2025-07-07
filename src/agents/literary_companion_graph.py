"""
Main LangGraph agent for the Literary Companion application.

This module defines the conversation flow graph that:
- Manages reading progress tracking
- Orchestrates thoughtful literary discussions
- Ensures contextual awareness boundaries
- Generates appropriate gestures and citations
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import logging
import re
from pathlib import Path

from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..models.state import (
    ConversationState, BookLocation, Message, 
    Gesture, Citation, GestureType
)
from ..tools.book_processor import BookProcessor
from ..services.gesture_parser import GestureParser
from ..services.citation_extractor import CitationExtractor

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State definition for the LangGraph conversation flow."""
    conversation_state: ConversationState
    current_input: str
    response: Optional[str]
    extracted_gestures: List[Gesture]
    extracted_citations: List[Citation]
    should_update_progress: bool
    new_location: Optional[BookLocation]


class LiteraryCompanionGraph:
    """
    Main conversation graph for the Literary Companion.
    
    This graph orchestrates:
    - Progress checking and updates
    - Contextual response generation
    - Gesture and citation extraction
    - State management
    """
    
    def __init__(self, llm_provider: str = "anthropic", model_name: str = "claude-3-opus-20240229"):
        """Initialize the graph with specified LLM."""
        # Initialize LLM
        if llm_provider == "anthropic":
            self.llm = ChatAnthropic(model=model_name, temperature=0.7)
        elif llm_provider == "openai":
            self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # Initialize services
        self.book_processor = BookProcessor()
        self.gesture_parser = GestureParser()
        self.citation_extractor = CitationExtractor()
        
        # Load prompts
        self.system_prompt = self._load_prompt("system_prompt.txt")
        self.persona_prompts = {
            "brothers_karamazov": self._load_prompt("brothers_karamazov_persona.txt")
        }
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _load_prompt(self, filename: str) -> str:
        """Load a prompt from the prompts directory."""
        prompt_path = Path(__file__).parent.parent / "prompts" / filename
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            return ""
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation flow."""
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("check_progress", self._check_progress_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("extract_gestures", self._extract_gestures_node)
        workflow.add_node("extract_citations", self._extract_citations_node)
        workflow.add_node("update_state", self._update_state_node)
        
        # Add edges
        workflow.set_entry_point("check_progress")
        workflow.add_edge("check_progress", "generate_response")
        workflow.add_edge("generate_response", "extract_gestures")
        workflow.add_edge("extract_gestures", "extract_citations")
        workflow.add_edge("extract_citations", "update_state")
        workflow.set_finish_point("update_state")
        
        return workflow.compile()
    
    def _check_progress_node(self, state: GraphState) -> GraphState:
        """
        Check if the user is updating their reading progress.
        
        This node analyzes the input to determine if the user is:
        - Reporting new reading progress
        - Asking a question about what they've read
        - Continuing a discussion
        """
        user_input = state["current_input"]
        
        # Patterns for progress updates
        progress_patterns = [
            r"(?i)i(?:'ve| have)?\s+read\s+(?:up\s+)?to\s+chapter\s+(\d+|[IVXLCDM]+)(?:,?\s*page\s+(\d+))?",
            r"(?i)i(?:'m| am)?\s+(?:currently\s+)?(?:on|at)\s+chapter\s+(\d+|[IVXLCDM]+)(?:,?\s*page\s+(\d+))?",
            r"(?i)just\s+finished\s+chapter\s+(\d+|[IVXLCDM]+)",
            r"(?i)(?:i(?:'m| am)?\s+)?(?:on|at)\s+page\s+(\d+)\s+(?:of\s+)?chapter\s+(\d+|[IVXLCDM]+)",
        ]
        
        # Check for progress update
        for pattern in progress_patterns:
            match = re.search(pattern, user_input)
            if match:
                # Extract chapter and page information
                groups = match.groups()
                chapter_num = self._parse_chapter_number(groups[0])
                page_num = int(groups[1]) if len(groups) > 1 and groups[1] else None
                
                # Get chapter title from table of contents
                chapter_title = self._get_chapter_title(
                    state["conversation_state"], 
                    chapter_num
                )
                
                state["should_update_progress"] = True
                state["new_location"] = BookLocation(
                    chapter_number=chapter_num,
                    chapter_title=chapter_title,
                    page_number=page_num or 1
                )
                break
        else:
            state["should_update_progress"] = False
            state["new_location"] = None
        
        return state
    
    def _generate_response_node(self, state: GraphState) -> GraphState:
        """
        Generate the main response using the LLM.
        
        This node:
        - Constructs the appropriate prompt
        - Ensures contextual boundaries
        - Generates a response with embedded gestures
        """
        conv_state = state["conversation_state"]
        
        # Get accessible text based on current reading progress
        if conv_state.current_location and conv_state.full_text:
            accessible_text = self.book_processor.get_text_up_to_location(
                conv_state.current_location
            )
        else:
            accessible_text = ""
        
        # Select appropriate persona
        persona_prompt = self._select_persona_prompt(conv_state.book_title)
        
        # Build message history
        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=persona_prompt)
        ]
        
        # Add conversation context
        if accessible_text and conv_state.current_location:
            context_msg = f"The user has read up to {conv_state.current_location.chapter_title}, page {conv_state.current_location.page_number}. You may reference any content up to this point."
            messages.append(SystemMessage(content=context_msg))
        
        # Add conversation history
        for msg in conv_state.get_conversation_context():
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        # Add current input
        messages.append(HumanMessage(content=state["current_input"]))
        
        # Generate response
        response = self.llm.invoke(messages)
        state["response"] = response.content
        
        return state
    
    def _extract_gestures_node(self, state: GraphState) -> GraphState:
        """Extract gesture annotations from the generated response."""
        response = state["response"]
        gestures = self.gesture_parser.extract_gestures(response)
        state["extracted_gestures"] = gestures
        
        # Clean gestures from response for display
        clean_response = self.gesture_parser.remove_gesture_tags(response)
        state["response"] = clean_response
        
        return state
    
    def _extract_citations_node(self, state: GraphState) -> GraphState:
        """Extract citations from the response."""
        response = state["response"]
        conv_state = state["conversation_state"]
        
        # Extract citations using the citation service
        citations = self.citation_extractor.extract_citations(
            response,
            conv_state.current_location,
            self.book_processor
        )
        
        state["extracted_citations"] = citations
        return state
    
    def _update_state_node(self, state: GraphState) -> GraphState:
        """Update the conversation state with the new message and progress."""
        conv_state = state["conversation_state"]
        
        # Update reading progress if needed
        if state["should_update_progress"] and state["new_location"]:
            conv_state.update_reading_progress(state["new_location"])
        
        # Add user message
        conv_state.add_message(
            role="user",
            content=state["current_input"]
        )
        
        # Add assistant response
        if state["response"]:
            conv_state.add_message(
                role="assistant",
                content=state["response"],
                gestures=state["extracted_gestures"],
                citations=state["extracted_citations"]
            )
        
        return state
    
    def _parse_chapter_number(self, num_str: str) -> int:
        """Convert chapter number string to integer."""
        # Reuse the book processor's method
        return self.book_processor._parse_chapter_number(num_str)
    
    def _get_chapter_title(self, conv_state: ConversationState, 
                          chapter_num: int) -> str:
        """Get chapter title from table of contents."""
        if not conv_state.table_of_contents:
            return f"Chapter {chapter_num}"
        
        for chapter in conv_state.table_of_contents.chapters:
            if chapter.get("number") == chapter_num:
                return chapter.get("title", f"Chapter {chapter_num}")
        
        return f"Chapter {chapter_num}"
    
    def _select_persona_prompt(self, book_title: str) -> str:
        """Select the appropriate persona prompt based on the book."""
        # Normalize book title
        normalized_title = book_title.lower().replace(" ", "_")
        
        # Map common variations
        if "brothers" in normalized_title and "karamazov" in normalized_title:
            return self.persona_prompts.get("brothers_karamazov", "")
        
        # Default to empty if no specific persona
        return ""
    
    async def process_message(self, conversation_state: ConversationState, 
                            user_input: str) -> ConversationState:
        """
        Process a user message through the graph.
        
        Args:
            conversation_state: Current conversation state
            user_input: User's message
            
        Returns:
            Updated conversation state
        """
        # Initialize graph state
        initial_state: GraphState = {
            "conversation_state": conversation_state,
            "current_input": user_input,
            "response": None,
            "extracted_gestures": [],
            "extracted_citations": [],
            "should_update_progress": False,
            "new_location": None
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Return updated conversation state
        return final_state["conversation_state"]