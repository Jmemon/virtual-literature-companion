"""
Data models and schemas for the Virtual Literature Companion.

This module defines all Pydantic models used for:
- Request/response validation
- Internal data structures
- Agent communication
- Database models
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class EmotionTag(str, Enum):
    """Enumeration of available emotion tags."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    CONTEMPLATIVE = "contemplative"
    EXCITED = "excited"
    MELANCHOLIC = "melancholic"
    PASSIONATE = "passionate"
    CURIOUS = "curious"
    SKEPTICAL = "skeptical"
    EMPATHETIC = "empathetic"
    ANALYTICAL = "analytical"


class BookMetadata(BaseModel):
    """Metadata for a processed book."""
    book_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    author: Optional[str] = None
    isbn: Optional[str] = None
    publication_year: Optional[int] = None
    genre: Optional[List[str]] = []
    page_count: Optional[int] = None
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processed_date: Optional[datetime] = None
    file_path: str
    processed_path: Optional[str] = None


class TableOfContents(BaseModel):
    """Structure for book table of contents."""
    
    class Chapter(BaseModel):
        chapter_number: Optional[int] = None
        title: str
        start_page: Optional[int] = None
        end_page: Optional[int] = None
        summary: Optional[str] = None
        themes: List[str] = []
        
    chapters: List[Chapter]
    total_chapters: int
    has_prologue: bool = False
    has_epilogue: bool = False


class TextChunk(BaseModel):
    """A chunk of text with metadata."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    book_id: str
    chapter_number: Optional[int] = None
    page_number: Optional[int] = None
    content: str
    start_char: int
    end_char: int
    themes: List[str] = []
    emotional_tone: Optional[str] = None
    key_quotes: List[str] = []
    character_mentions: List[str] = []
    plot_points: List[Dict[str, Any]] = []
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class Citation(BaseModel):
    """A citation reference within a response."""
    text: str
    chapter: Optional[int] = None
    page: Optional[int] = None
    character_context: Optional[str] = None
    theme_context: Optional[str] = None


class EmotionalResponse(BaseModel):
    """Structure for persona responses with emotional content."""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    emotions: List[EmotionTag]
    emotion_intensities: Dict[str, float] = {}  # emotion -> intensity (0-1)
    citations: List[Citation] = []
    voice_modulation: Dict[str, float] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('emotion_intensities')
    def validate_intensities(cls, v, values):
        """Ensure all emotions have intensities between 0 and 1."""
        if 'emotions' in values:
            for emotion in values['emotions']:
                if emotion.value not in v:
                    v[emotion.value] = 0.5  # Default intensity
                elif not 0 <= v[emotion.value] <= 1:
                    raise ValueError(f'Intensity for {emotion} must be between 0 and 1')
        return v


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_message: str
    assistant_response: EmotionalResponse
    context_used: List[TextChunk] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationSession(BaseModel):
    """A complete conversation session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    book_id: str
    turns: List[ConversationTurn] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class VoiceSettings(BaseModel):
    """Settings for voice synthesis."""
    voice_id: str
    stability: float = Field(0.5, ge=0, le=1)
    similarity_boost: float = Field(0.75, ge=0, le=1)
    style: float = Field(0.5, ge=0, le=1)
    pitch_adjustment: float = Field(1.0, ge=0.5, le=2.0)
    speed_adjustment: float = Field(1.0, ge=0.5, le=2.0)


class AgentMessage(BaseModel):
    """Message format for inter-agent communication."""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    requires_response: bool = True


class TaskAssignment(BaseModel):
    """Task assignment from supervisor to worker agents."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assigned_to: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = Field(1, ge=1, le=5)
    deadline: Optional[datetime] = None
    status: str = "pending"


class ProcessingRequest(BaseModel):
    """Request to process a book."""
    file_path: str
    user_id: str
    extract_toc: bool = True
    analyze_themes: bool = True
    generate_summaries: bool = False
    chunk_size: int = 1000
    chunk_overlap: int = 200


class ChatRequest(BaseModel):
    """Request for a chat interaction."""
    session_id: Optional[str] = None
    user_id: str
    book_id: str
    message: str
    include_voice: bool = True
    emotion_preference: Optional[List[EmotionTag]] = None


class ChatResponse(BaseModel):
    """Response from a chat interaction."""
    session_id: str
    response: EmotionalResponse
    voice_url: Optional[str] = None
    ui_animations: Dict[str, Any] = {}
    suggested_followups: List[str] = []


class UIAnimation(BaseModel):
    """Animation instructions for the UI."""
    element_id: str
    animation_type: str
    duration: float  # seconds
    parameters: Dict[str, Any]
    trigger_time: float = 0  # delay in seconds


class EmotiveUIState(BaseModel):
    """Current state of the emotive UI."""
    primary_emotion: EmotionTag
    emotion_blend: Dict[EmotionTag, float]
    animations: List[UIAnimation]
    color_palette: Dict[str, str]
    particle_effects: Dict[str, Any]
    morphing_shape: Dict[str, Any]


# Agent-specific models

class PreprocessorResult(BaseModel):
    """Result from the preprocessing agent."""
    book_metadata: BookMetadata
    table_of_contents: TableOfContents
    chunks: List[TextChunk]
    processing_time: float
    errors: List[str] = []


class PersonaState(BaseModel):
    """Internal state of the persona agent."""
    character_name: str = "Athena"
    current_mood: EmotionTag = EmotionTag.CURIOUS
    conversation_context: List[ConversationTurn] = []
    reading_progress: Dict[str, Any] = {}
    user_preferences: Dict[str, Any] = {}


class SupervisorDecision(BaseModel):
    """Decision made by the supervisor agent."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_task: str
    assigned_agents: List[str]
    task_breakdown: List[TaskAssignment]
    execution_order: List[str]
    estimated_completion_time: Optional[float] = None


# Validation models for API endpoints

class UploadBookRequest(BaseModel):
    """Request to upload a book."""
    filename: str
    file_content: bytes
    user_id: str
    metadata: Optional[Dict[str, Any]] = {}


class UploadBookResponse(BaseModel):
    """Response after uploading a book."""
    book_id: str
    status: str
    message: str
    processing_job_id: Optional[str] = None


class GetBookResponse(BaseModel):
    """Response when retrieving book information."""
    book: BookMetadata
    table_of_contents: Optional[TableOfContents] = None
    total_chunks: int
    processing_status: str


class SearchBooksRequest(BaseModel):
    """Request to search for books."""
    query: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[List[str]] = None
    year_range: Optional[tuple[int, int]] = None
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)


class SearchBooksResponse(BaseModel):
    """Response from book search."""
    books: List[BookMetadata]
    total_count: int
    has_more: bool


# WebSocket models

class WSMessage(BaseModel):
    """WebSocket message format."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSChatMessage(WSMessage):
    """WebSocket chat message."""
    type: str = "chat"
    data: ChatRequest


class WSEmotionUpdate(WSMessage):
    """WebSocket emotion update."""
    type: str = "emotion_update"
    data: EmotiveUIState


class WSVoiceReady(WSMessage):
    """WebSocket voice ready notification."""
    type: str = "voice_ready"
    data: Dict[str, str]  # {voice_url: str, duration: float}


# Export all models
__all__ = [
    "EmotionTag",
    "BookMetadata",
    "TableOfContents",
    "TextChunk",
    "Citation",
    "EmotionalResponse",
    "ConversationTurn",
    "ConversationSession",
    "VoiceSettings",
    "AgentMessage",
    "TaskAssignment",
    "ProcessingRequest",
    "ChatRequest",
    "ChatResponse",
    "UIAnimation",
    "EmotiveUIState",
    "PreprocessorResult",
    "PersonaState",
    "SupervisorDecision",
    "UploadBookRequest",
    "UploadBookResponse",
    "GetBookResponse",
    "SearchBooksRequest",
    "SearchBooksResponse",
    "WSMessage",
    "WSChatMessage",
    "WSEmotionUpdate",
    "WSVoiceReady"
]