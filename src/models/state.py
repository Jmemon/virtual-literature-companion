"""
State management for the LangGraph multi-agent system.

This module defines the shared state that flows through the agent graph,
enabling coordination and communication between different agents.
"""

from typing import List, Dict, Optional, Any, Annotated
from datetime import datetime
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import uuid

from .schemas import (
    BookMetadata,
    TableOfContents,
    TextChunk,
    EmotionalResponse,
    ConversationTurn,
    EmotionTag,
    TaskAssignment,
    PreprocessorResult,
    PersonaState,
    Citation
)


class AgentState(BaseModel):
    """
    Shared state for the multi-agent system.
    
    This state is passed between agents and accumulates information
    as it flows through the graph. Each agent can read from and write
    to specific fields based on their responsibilities.
    """
    
    # Conversation Management
    messages: Annotated[List[Dict[str, Any]], add_messages] = Field(
        default_factory=list,
        description="Conversation messages with metadata"
    )
    
    # Session Information
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier"
    )
    user_id: str = Field(
        ...,
        description="User identifier from the request"
    )
    
    # Book Context
    book_id: Optional[str] = Field(
        None,
        description="Current book being discussed"
    )
    book_metadata: Optional[BookMetadata] = Field(
        None,
        description="Metadata of the current book"
    )
    table_of_contents: Optional[TableOfContents] = Field(
        None,
        description="Table of contents for navigation"
    )
    current_chapter: Optional[int] = Field(
        None,
        description="Chapter currently being discussed"
    )
    
    # Text Processing
    raw_text: Optional[str] = Field(
        None,
        description="Raw text to be processed"
    )
    processed_chunks: List[TextChunk] = Field(
        default_factory=list,
        description="Processed text chunks with metadata"
    )
    relevant_chunks: List[TextChunk] = Field(
        default_factory=list,
        description="Chunks relevant to current query"
    )
    
    # Persona State
    persona_state: PersonaState = Field(
        default_factory=PersonaState,
        description="Current state of the persona agent"
    )
    
    # Emotional Context
    current_emotions: List[EmotionTag] = Field(
        default_factory=list,
        description="Current emotional state"
    )
    emotion_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of emotional states"
    )
    
    # Response Generation
    generated_response: Optional[EmotionalResponse] = Field(
        None,
        description="Generated response from persona"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations to include in response"
    )
    
    # Voice Synthesis
    voice_enabled: bool = Field(
        True,
        description="Whether to generate voice output"
    )
    voice_url: Optional[str] = Field(
        None,
        description="URL of generated voice file"
    )
    voice_parameters: Dict[str, float] = Field(
        default_factory=dict,
        description="Voice modulation parameters"
    )
    
    # UI State
    ui_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current UI state and animations"
    )
    
    # Task Management
    current_task: Optional[str] = Field(
        None,
        description="Current task being processed"
    )
    task_assignments: List[TaskAssignment] = Field(
        default_factory=list,
        description="Tasks assigned by supervisor"
    )
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="List of completed task IDs"
    )
    
    # Agent Coordination
    next_agent: Optional[str] = Field(
        None,
        description="Next agent to invoke"
    )
    agent_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from different agents"
    )
    
    # Error Handling
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered during processing"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="State creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    # Memory Context (from Honcho)
    conversation_history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Retrieved conversation history"
    )
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences and reading habits"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        }
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        
    def add_error(self, agent: str, error: str, details: Optional[Dict] = None):
        """Add an error to the state."""
        error_entry = {
            "agent": agent,
            "error": error,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.errors.append(error_entry)
        
    def set_agent_output(self, agent: str, output: Any):
        """Set the output from a specific agent."""
        self.agent_outputs[agent] = output
        self.updated_at = datetime.utcnow()
        
    def get_agent_output(self, agent: str) -> Optional[Any]:
        """Get the output from a specific agent."""
        return self.agent_outputs.get(agent)
        
    def mark_task_complete(self, task_id: str):
        """Mark a task as complete."""
        self.completed_tasks.append(task_id)
        # Update task status in assignments
        for task in self.task_assignments:
            if task.task_id == task_id:
                task.status = "completed"
                break
                
    def get_latest_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content")
        return None
        
    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context as a string."""
        recent_messages = self.messages[-max_turns * 2:]  # Get last N exchanges
        context_parts = []
        
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"{role.capitalize()}: {content}")
            
        return "\n".join(context_parts)
        
    def update_emotional_state(self, emotions: List[EmotionTag], intensities: Dict[str, float]):
        """Update the current emotional state."""
        self.current_emotions = emotions
        self.emotion_history.append({
            "emotions": [e.value for e in emotions],
            "intensities": intensities,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    def get_emotional_trajectory(self) -> List[Dict[str, Any]]:
        """Get the emotional trajectory over the conversation."""
        return self.emotion_history[-10:]  # Last 10 emotional states
        
    def should_switch_chapter(self) -> bool:
        """Determine if the conversation should move to a different chapter."""
        # Logic to detect when user wants to discuss a different part of the book
        latest_message = self.get_latest_user_message()
        if not latest_message:
            return False
            
        chapter_indicators = [
            "next chapter", "previous chapter", "chapter", 
            "let's move to", "what about when", "later in the book"
        ]
        
        return any(indicator in latest_message.lower() for indicator in chapter_indicators)
        
    def to_checkpoint(self) -> Dict[str, Any]:
        """Create a checkpoint of the current state for persistence."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "book_id": self.book_id,
            "current_chapter": self.current_chapter,
            "messages": self.messages[-20:],  # Keep last 20 messages
            "current_emotions": [e.value for e in self.current_emotions],
            "persona_state": self.persona_state.dict(),
            "updated_at": self.updated_at.isoformat()
        }
        
    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> "AgentState":
        """Restore state from a checkpoint."""
        # Convert emotion strings back to enums
        emotions = [EmotionTag(e) for e in checkpoint.get("current_emotions", [])]
        
        # Restore persona state
        persona_state = PersonaState(**checkpoint.get("persona_state", {}))
        
        return cls(
            session_id=checkpoint["session_id"],
            user_id=checkpoint["user_id"],
            book_id=checkpoint.get("book_id"),
            current_chapter=checkpoint.get("current_chapter"),
            messages=checkpoint.get("messages", []),
            current_emotions=emotions,
            persona_state=persona_state
        )


# Additional state models for specific workflows

class ProcessingState(BaseModel):
    """State for book processing workflow."""
    file_path: str
    file_content: Optional[bytes] = None
    extracted_text: Optional[str] = None
    table_of_contents: Optional[TableOfContents] = None
    chunks: List[TextChunk] = Field(default_factory=list)
    metadata: Optional[BookMetadata] = None
    processing_status: str = "pending"
    progress: float = 0.0
    errors: List[str] = Field(default_factory=list)
    
    def update_progress(self, progress: float, status: Optional[str] = None):
        """Update processing progress."""
        self.progress = min(progress, 100.0)
        if status:
            self.processing_status = status


class VoiceGenerationState(BaseModel):
    """State for voice generation workflow."""
    text: str
    emotions: List[EmotionTag]
    emotion_intensities: Dict[str, float]
    voice_settings: Dict[str, float]
    generated_audio: Optional[bytes] = None
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    generation_status: str = "pending"
    

# Export state models
__all__ = [
    "AgentState",
    "ProcessingState", 
    "VoiceGenerationState"
]