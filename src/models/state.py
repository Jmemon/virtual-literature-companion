"""
State management for the Literary Companion application.

This module defines the core state structure that tracks:
- User's reading progress through the book
- Conversation history and context
- Book metadata and navigation
- Gesture and UI state for the embodied interface
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class GestureType(str, Enum):
    """Enumeration of available gesture types for the embodied UI."""
    LEAN_IN = "LEAN_IN"
    PULL_BACK = "PULL_BACK"
    TREMBLE = "TREMBLE"
    ILLUMINATE = "ILLUMINATE"
    FRAGMENT = "FRAGMENT"
    WHISPER = "WHISPER"
    GRIP = "GRIP"
    SHATTER = "SHATTER"
    BREATHE = "BREATHE"
    REACH = "REACH"
    DANCE = "DANCE"


class Gesture(BaseModel):
    """Represents a single gesture with its parameters."""
    type: GestureType
    parameters: Optional[List[str]] = Field(default_factory=list)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    duration: float = 1.0  # Duration in seconds


class BookLocation(BaseModel):
    """Represents a specific location within the book."""
    chapter_number: int
    chapter_title: str
    page_number: int
    paragraph_index: Optional[int] = None
    sentence_index: Optional[int] = None


class Citation(BaseModel):
    """Represents a citation to a specific passage in the book."""
    location: BookLocation
    text: str
    context: Optional[str] = None  # Surrounding text for context


class Message(BaseModel):
    """Represents a message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    gestures: List[Gesture] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class TableOfContents(BaseModel):
    """Represents the book's table of contents."""
    chapters: List[Dict[str, Any]]  # Each chapter has title, number, start_page, end_page
    total_pages: int
    book_title: str
    author: str


class ConversationState(BaseModel):
    """
    Core state for the Literary Companion conversation.
    
    This state is passed through the LangGraph agent and maintains:
    - The user's current reading progress
    - Complete conversation history
    - Book metadata and navigation info
    - Active gestures and UI state
    """
    
    # Book information
    book_title: str
    author: str
    table_of_contents: Optional[TableOfContents] = None
    full_text: Optional[str] = None  # The complete book text
    
    # User's reading progress
    current_location: Optional[BookLocation] = None
    reading_history: List[BookLocation] = Field(default_factory=list)
    
    # Conversation state
    messages: List[Message] = Field(default_factory=list)
    current_topic: Optional[str] = None
    discussed_themes: List[str] = Field(default_factory=list)
    
    # UI and gesture state
    active_gestures: List[Gesture] = Field(default_factory=list)
    ui_proximity: float = 1.0  # 1.0 = normal, <1 = pulled back, >1 = leaned in
    emotional_intensity: float = 0.5  # 0-1 scale
    
    # Session metadata
    session_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    last_interaction: datetime = Field(default_factory=datetime.now)
    
    def get_accessible_text(self) -> str:
        """
        Returns only the text that the user has read up to their current location.
        
        This is crucial for maintaining the contextual awareness constraint - the AI
        must never reference content beyond what the user has indicated they've read.
        """
        if not self.current_location or not self.full_text:
            return ""
        
        # Extract text up to the current page
        # This would need proper implementation based on how the text is structured
        # For now, returning a placeholder
        return self._extract_text_up_to_location(self.current_location)
    
    def _extract_text_up_to_location(self, location: BookLocation) -> str:
        """Helper method to extract text up to a specific location."""
        # This would be implemented based on the book parsing strategy
        # Would need to handle page boundaries, chapter divisions, etc.
        # For now, returning empty string as placeholder
        return ""
    
    def add_message(self, role: str, content: str, 
                   gestures: Optional[List[Gesture]] = None,
                   citations: Optional[List[Citation]] = None) -> None:
        """Add a new message to the conversation history."""
        message = Message(
            role=role,
            content=content,
            gestures=gestures or [],
            citations=citations or []
        )
        self.messages.append(message)
        self.last_interaction = datetime.now()
        
        # Update active gestures
        if gestures:
            self.active_gestures.extend(gestures)
    
    def update_reading_progress(self, new_location: BookLocation) -> None:
        """Update the user's reading progress."""
        if self.current_location:
            self.reading_history.append(self.current_location)
        self.current_location = new_location
        self.last_interaction = datetime.now()
    
    def get_conversation_context(self, num_messages: int = 10) -> List[Message]:
        """Get recent conversation context."""
        return self.messages[-num_messages:] if self.messages else []
    
    def clear_expired_gestures(self, current_time: float) -> None:
        """Remove gestures that have completed their duration."""
        self.active_gestures = [
            g for g in self.active_gestures 
            if current_time - g.timestamp < g.duration
        ]