"""
State management for the Wild Genius Professor agent.
Tracks conversation flow, emotional states, and user progress through the text.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class EmotionalState(str, Enum):
    """Available emotional states for the professor."""
    WONDER = "WONDER"
    ANGUISH = "ANGUISH"
    ECSTASY = "ECSTASY"
    CONTEMPLATION = "CONTEMPLATION"
    MELANCHOLY = "MELANCHOLY"
    FERVOR = "FERVOR"
    SERENITY = "SERENITY"
    TURMOIL = "TURMOIL"
    RAPTURE = "RAPTURE"
    DESPAIR = "DESPAIR"


class ReadingProgress(BaseModel):
    """Tracks user's progress through the book."""
    book_id: str = "brothers_karamazov"
    current_book: int = Field(default=1, ge=1, le=12)
    current_chapter: int = Field(default=1, ge=1)
    last_updated: datetime = Field(default_factory=datetime.now)
    completed_sections: List[str] = Field(default_factory=list)
    
    def can_discuss(self, book: int, chapter: int) -> bool:
        """Check if a section can be discussed without spoilers."""
        if book < self.current_book:
            return True
        elif book == self.current_book:
            return chapter <= self.current_chapter
        return False


class Citation(BaseModel):
    """Represents a text citation."""
    book: int
    chapter: int
    page: Optional[int] = None
    quote: Optional[str] = None
    context: str
    

class ConversationContext(BaseModel):
    """Maintains context about the ongoing conversation."""
    themes_discussed: List[str] = Field(default_factory=list)
    characters_mentioned: List[str] = Field(default_factory=list)
    questions_asked: List[str] = Field(default_factory=list)
    insights_discovered: List[str] = Field(default_factory=list)
    emotional_journey: List[EmotionalState] = Field(default_factory=list)
    
    
class Message(BaseModel):
    """Enhanced message with emotional metadata."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotions: List[EmotionalState] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    ui_directives: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    """
    The main state for the Wild Genius Professor agent.
    This is what gets passed between nodes in the LangGraph.
    """
    # Core conversation state
    messages: List[Message]
    
    # User tracking
    user_id: str
    session_id: str
    reading_progress: ReadingProgress
    
    # Emotional and thematic tracking
    current_emotion: EmotionalState
    emotional_intensity: float  # 0.0 to 1.0
    conversation_context: ConversationContext
    
    # Agent decision tracking
    next_action: Literal["respond", "clarify_progress", "cite_text", "generate_image", "deepen_inquiry"]
    pending_citations: List[Citation]
    
    # UI communication
    ui_updates: Dict[str, Any]  # Commands for UI changes
    image_prompt: Optional[str]  # If image should be generated
    music_directive: Optional[str]  # Musical mood change
    
    # Memory integration (Honcho)
    honcho_session_id: Optional[str]
    user_profile: Dict[str, Any]  # Built from Honcho insights
    
    # Control flow
    should_end: bool
    error: Optional[str]


class ProfessorMemory(BaseModel):
    """
    Long-term memory structure for Honcho integration.
    Tracks patterns across conversations.
    """
    user_id: str
    
    # Learning style insights
    prefers_concrete_examples: bool = False
    responds_to_emotional_appeals: bool = True
    enjoys_philosophical_depth: bool = True
    needs_more_guidance: bool = False
    
    # Thematic interests
    favorite_themes: List[str] = Field(default_factory=list)
    avoided_topics: List[str] = Field(default_factory=list)
    recurring_questions: List[str] = Field(default_factory=list)
    
    # Engagement patterns
    average_session_length: int = 0
    preferred_discussion_pace: Literal["fast", "moderate", "slow"] = "moderate"
    responds_best_to_emotions: List[EmotionalState] = Field(default_factory=list)
    
    # Progress patterns
    chapters_discussed: List[str] = Field(default_factory=list)
    breakthrough_moments: List[Dict[str, Any]] = Field(default_factory=list)
    struggling_concepts: List[str] = Field(default_factory=list)


def create_initial_state(user_id: str, session_id: str) -> AgentState:
    """Create a fresh agent state for a new conversation."""
    return {
        "messages": [],
        "user_id": user_id,
        "session_id": session_id,
        "reading_progress": ReadingProgress(),
        "current_emotion": EmotionalState.CONTEMPLATION,
        "emotional_intensity": 0.5,
        "conversation_context": ConversationContext(),
        "next_action": "respond",
        "pending_citations": [],
        "ui_updates": {},
        "image_prompt": None,
        "music_directive": None,
        "honcho_session_id": None,
        "user_profile": {},
        "should_end": False,
        "error": None
    }


def extract_emotions_from_text(text: str) -> List[EmotionalState]:
    """Extract emotion tags from professor's response."""
    emotions = []
    for emotion in EmotionalState:
        if f"[{emotion.value}]" in text:
            emotions.append(emotion)
    return emotions


def calculate_emotional_intensity(emotions: List[EmotionalState], text: str) -> float:
    """
    Calculate the intensity of emotional expression.
    Based on number of emotions, punctuation, and key phrases.
    """
    intensity = 0.3  # Base intensity
    
    # More emotions = higher intensity
    intensity += len(emotions) * 0.1
    
    # Exclamation marks increase intensity
    intensity += text.count("!") * 0.05
    
    # Question marks (especially multiple) show turmoil
    intensity += text.count("?") * 0.03
    
    # Ellipses show contemplation/trailing thought
    intensity -= text.count("...") * 0.02
    
    # Intense emotions boost intensity
    intense_emotions = {EmotionalState.ECSTASY, EmotionalState.ANGUISH, 
                       EmotionalState.RAPTURE, EmotionalState.DESPAIR}
    if any(e in intense_emotions for e in emotions):
        intensity += 0.2
    
    # Cap between 0 and 1
    return max(0.0, min(1.0, intensity))


def update_conversation_context(
    context: ConversationContext, 
    message: Message
) -> ConversationContext:
    """Update conversation context based on new message."""
    # This would use NLP to extract themes, characters, etc.
    # For now, we'll update emotional journey
    if message.emotions:
        context.emotional_journey.extend(message.emotions)
    
    return context