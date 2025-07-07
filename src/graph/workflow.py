"""
Main LangGraph workflow for the Virtual Literature Companion.

This module defines:
- The supervisor agent that coordinates other agents
- The main conversation workflow graph
- Agent routing and orchestration logic
"""

from typing import Dict, List, Any, Optional, Literal, TypedDict
from datetime import datetime
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from ..config import settings, get_prompt_template
from ..models.state import AgentState
from ..models.schemas import (
    TaskAssignment,
    SupervisorDecision,
    EmotionTag,
    ChatRequest,
    ChatResponse
)
from ..agents.pdf_agent import pdf_preprocessor_node
from ..agents.persona_agent import persona_agent_node
from ..voice.tts_engine import EmotionalTTSEngine
from ..memory.honcho_client import HonchoMemoryClient


class SupervisorAgent:
    """
    Supervisor agent that coordinates the multi-agent system.
    
    The supervisor:
    1. Analyzes incoming requests
    2. Determines which agents to invoke
    3. Manages task flow and dependencies
    4. Handles error recovery
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.llm_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
        self.available_agents = ["pdf_preprocessor", "persona", "voice_synthesizer"]
        
    async def route_request(self, state: Dict[str, Any]) -> str:
        """
        Determine the next agent to invoke based on current state.
        
        Args:
            state: Current graph state
            
        Returns:
            Name of the next agent or "end"
        """
        # Check if this is a book processing request
        if state.get("file_path") and not state.get("preprocessing_complete"):
            logger.info("Routing to PDF preprocessor")
            return "pdf_preprocessor"
            
        # Check if this is a chat request
        if state.get("messages") and state.get("book_id"):
            # Ensure book is processed
            if not state.get("processed_chunks"):
                logger.warning("Book not processed, cannot chat")
                return "end"
                
            logger.info("Routing to persona agent")
            return "persona"
            
        # Check if voice synthesis is needed
        if (state.get("generated_response") and 
            state.get("voice_enabled", True) and 
            not state.get("voice_url")):
            logger.info("Routing to voice synthesizer")
            return "voice_synthesizer"
            
        # Default to end
        return "end"
        
    async def create_decision(self, state: Dict[str, Any]) -> SupervisorDecision:
        """
        Create a detailed decision about task execution.
        
        Args:
            state: Current state
            
        Returns:
            SupervisorDecision with task breakdown
        """
        task = state.get("current_task", "Process user request")
        
        # Analyze what needs to be done
        task_assignments = []
        assigned_agents = []
        
        # Book processing task
        if state.get("file_path") and not state.get("preprocessing_complete"):
            task_assignments.append(TaskAssignment(
                assigned_to="pdf_preprocessor",
                task_type="extract_and_analyze",
                parameters={
                    "file_path": state["file_path"],
                    "extract_toc": True,
                    "analyze_themes": True
                },
                priority=1
            ))
            assigned_agents.append("pdf_preprocessor")
            
        # Conversation task
        if state.get("messages"):
            task_assignments.append(TaskAssignment(
                assigned_to="persona",
                task_type="generate_response",
                parameters={
                    "emotion_preference": state.get("emotion_preference"),
                    "include_citations": True
                },
                priority=2
            ))
            assigned_agents.append("persona")
            
            # Voice task (if enabled)
            if state.get("voice_enabled", True):
                task_assignments.append(TaskAssignment(
                    assigned_to="voice_synthesizer",
                    task_type="synthesize_speech",
                    parameters={
                        "use_cache": True
                    },
                    priority=3
                ))
                assigned_agents.append("voice_synthesizer")
                
        return SupervisorDecision(
            input_task=task,
            assigned_agents=assigned_agents,
            task_breakdown=task_assignments,
            execution_order=assigned_agents
        )


# Define the supervisor node
async def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor node that determines routing.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with routing decision
    """
    supervisor = SupervisorAgent()
    
    # Create routing decision
    next_agent = await supervisor.route_request(state)
    state["next_agent"] = next_agent
    
    # Create detailed decision if needed
    if state.get("require_decision_details"):
        decision = await supervisor.create_decision(state)
        state["supervisor_decision"] = decision
        
    logger.info(f"Supervisor routing to: {next_agent}")
    return state


# Voice synthesis node
async def voice_synthesizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Voice synthesis node that generates emotional speech.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with voice URL
    """
    if not state.get("generated_response"):
        logger.warning("No response to synthesize")
        return state
        
    response = state["generated_response"]
    
    # Extract response data
    if isinstance(response, dict):
        text = response.get("text", "")
        emotions = response.get("emotions", [])
        emotion_intensities = response.get("emotion_intensities", {})
    else:
        # Handle EmotionalResponse object
        text = response.text
        emotions = response.emotions
        emotion_intensities = response.emotion_intensities
        
    if not text:
        logger.warning("Empty text for voice synthesis")
        return state
        
    # Convert emotion strings to enums if needed
    if emotions and isinstance(emotions[0], str):
        emotions = [EmotionTag(e) for e in emotions]
        
    try:
        # Synthesize speech
        tts_engine = EmotionalTTSEngine()
        audio_bytes, cache_key = await tts_engine.synthesize_emotional_speech(
            text,
            emotions,
            emotion_intensities
        )
        
        # Save audio and create URL
        # In production, this would upload to cloud storage
        audio_path = settings.voice_cache_dir / f"{cache_key}.mp3"
        voice_url = f"/audio/{cache_key}.mp3"
        
        state["voice_url"] = voice_url
        state["voice_cache_key"] = cache_key
        
        logger.info(f"Generated voice audio: {voice_url}")
        
    except Exception as e:
        logger.error(f"Voice synthesis error: {e}")
        state["voice_error"] = str(e)
        
    return state


# Define routing function
def route_next_agent(state: Dict[str, Any]) -> str:
    """
    Determine the next node based on supervisor decision.
    
    Args:
        state: Current state
        
    Returns:
        Next node name
    """
    next_agent = state.get("next_agent", "end")
    
    if next_agent == "end":
        return END
        
    return next_agent


# Build the main workflow graph
def build_literature_companion_graph() -> StateGraph:
    """
    Build the main LangGraph workflow.
    
    Returns:
        Configured StateGraph
    """
    # Create the graph
    workflow = StateGraph(Dict)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("pdf_preprocessor", pdf_preprocessor_node)
    workflow.add_node("persona", persona_agent_node)
    workflow.add_node("voice_synthesizer", voice_synthesizer_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_next_agent,
        {
            "pdf_preprocessor": "pdf_preprocessor",
            "persona": "persona",
            "voice_synthesizer": "voice_synthesizer",
            END: END
        }
    )
    
    # Add edges back to supervisor from agents
    workflow.add_edge("pdf_preprocessor", "supervisor")
    workflow.add_edge("persona", "supervisor")
    workflow.add_edge("voice_synthesizer", "supervisor")
    
    # Compile the graph
    return workflow.compile()


# Main conversation handler
async def handle_chat_request(request: ChatRequest) -> ChatResponse:
    """
    Handle a chat request through the multi-agent system.
    
    Args:
        request: Chat request
        
    Returns:
        Chat response with emotional content and voice
    """
    # Initialize Honcho client
    async with HonchoMemoryClient() as honcho_client:
        # Create or get user
        await honcho_client.create_or_get_user(request.user_id)
        
        # Create or get session
        if not request.session_id:
            session_id = await honcho_client.create_session(
                request.user_id,
                request.book_id
            )
        else:
            session_id = request.session_id
            
    # Build initial state
    initial_state = {
        "messages": [{"role": "user", "content": request.message}],
        "user_id": request.user_id,
        "book_id": request.book_id,
        "session_id": session_id,
        "voice_enabled": request.include_voice,
        "emotion_preference": request.emotion_preference,
        # Load book chunks - in production this would come from a database
        "processed_chunks": [],  # Would be loaded based on book_id
        "book_metadata": None,   # Would be loaded based on book_id
        "table_of_contents": None  # Would be loaded based on book_id
    }
    
    # Run the workflow
    graph = build_literature_companion_graph()
    final_state = await graph.ainvoke(initial_state)
    
    # Extract response
    response = final_state.get("generated_response")
    if not response:
        raise ValueError("No response generated")
        
    # Build UI animations based on emotions
    ui_animations = {
        "primary_emotion": response.emotions[0].value if response.emotions else "curious",
        "emotion_blend": response.emotion_intensities,
        "animation_duration": len(response.text) * 0.05  # Rough estimate
    }
    
    # Generate follow-up suggestions
    suggested_followups = _generate_followup_suggestions(
        response.text,
        response.emotions,
        final_state.get("relevant_chunks", [])
    )
    
    return ChatResponse(
        session_id=session_id,
        response=response,
        voice_url=final_state.get("voice_url"),
        ui_animations=ui_animations,
        suggested_followups=suggested_followups
    )


def _generate_followup_suggestions(response_text: str, 
                                 emotions: List[EmotionTag],
                                 chunks: List[Any]) -> List[str]:
    """
    Generate suggested follow-up questions based on the response.
    
    Args:
        response_text: Generated response
        emotions: Response emotions
        chunks: Relevant text chunks
        
    Returns:
        List of suggested questions
    """
    suggestions = []
    
    # Emotion-based suggestions
    if EmotionTag.CURIOUS in emotions:
        suggestions.append("What other interpretations have you considered?")
    elif EmotionTag.PASSIONATE in emotions:
        suggestions.append("What makes this theme so compelling to you?")
    elif EmotionTag.MELANCHOLIC in emotions:
        suggestions.append("How does this connect to the broader themes of loss in the book?")
        
    # Content-based suggestions
    if "character" in response_text.lower():
        suggestions.append("How does this character evolve throughout the story?")
    if "theme" in response_text.lower():
        suggestions.append("Where else do we see this theme manifested?")
        
    # Limit to 3 suggestions
    return suggestions[:3]


# Export main components
__all__ = [
    "build_literature_companion_graph",
    "handle_chat_request",
    "SupervisorAgent"
]