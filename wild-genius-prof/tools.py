"""
Tools for the Wild Genius Professor agent.
These tools enable citation lookup, emotion expression, and text analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
import json
import re
from datetime import datetime

from .state import Citation, EmotionalState, ReadingProgress


class CitationSearchInput(BaseModel):
    """Input for citation search tool."""
    query: str = Field(..., description="Search query for finding relevant passages")
    book_number: Optional[int] = Field(None, description="Specific book to search in")
    chapter_number: Optional[int] = Field(None, description="Specific chapter to search in")
    

class EmotionAnalysisInput(BaseModel):
    """Input for emotion analysis tool."""
    text: str = Field(..., description="Text to analyze for emotional content")
    context: str = Field(..., description="Context of the discussion")


class ProgressCheckInput(BaseModel):
    """Input for checking reading progress."""
    book: int = Field(..., description="Book number to check")
    chapter: int = Field(..., description="Chapter number to check")


class ImageGenerationInput(BaseModel):
    """Input for image generation planning."""
    theme: str = Field(..., description="Theme or subject for the image")
    emotion: EmotionalState = Field(..., description="Current emotional state")
    style_notes: str = Field(..., description="Style consistency notes")


@tool
def search_citations(
    query: str,
    book_number: Optional[int] = None,
    chapter_number: Optional[int] = None,
    reading_progress: Optional[Dict[str, Any]] = None
) -> List[Citation]:
    """
    Search for relevant citations in The Brothers Karamazov.
    Respects user's reading progress to avoid spoilers.
    
    Args:
        query: Search query for finding passages
        book_number: Optional specific book to search
        chapter_number: Optional specific chapter
        reading_progress: User's current reading progress
        
    Returns:
        List of relevant citations
    """
    # This would connect to a vector database with chunked text
    # For now, return mock citations
    mock_citations = [
        Citation(
            book=5,
            chapter=5,
            quote="If God does not exist, then everything is permitted.",
            context="Ivan's philosophical argument about morality and faith"
        ),
        Citation(
            book=6,
            chapter=3,
            quote="Love all God's creation, the whole and every grain of sand in it.",
            context="Father Zosima's teachings on universal love"
        )
    ]
    
    # Filter based on reading progress
    if reading_progress:
        progress = ReadingProgress(**reading_progress)
        mock_citations = [
            c for c in mock_citations 
            if progress.can_discuss(c.book, c.chapter)
        ]
    
    return mock_citations


@tool 
def analyze_user_emotion(text: str, context: str) -> Dict[str, Any]:
    """
    Analyze the emotional tone of user's message to calibrate response.
    
    Args:
        text: User's message
        context: Current conversation context
        
    Returns:
        Dictionary with emotional analysis
    """
    # Simple keyword-based analysis
    # In production, this would use sentiment analysis
    
    emotions_detected = []
    intensity = 0.3
    
    # Check for question marks - indicates curiosity
    if "?" in text:
        emotions_detected.append("curious")
        intensity += 0.1
        
    # Check for confusion indicators
    confusion_words = ["confused", "don't understand", "lost", "unclear"]
    if any(word in text.lower() for word in confusion_words):
        emotions_detected.append("confused")
        intensity += 0.2
        
    # Check for excitement
    if "!" in text or any(word in text.lower() for word in ["amazing", "wonderful", "love"]):
        emotions_detected.append("excited")
        intensity += 0.3
        
    # Check for frustration
    frustration_words = ["frustrated", "annoying", "difficult", "hate"]
    if any(word in text.lower() for word in frustration_words):
        emotions_detected.append("frustrated")
        intensity += 0.2
    
    return {
        "emotions": emotions_detected,
        "intensity": min(intensity, 1.0),
        "suggested_response_emotion": _suggest_response_emotion(emotions_detected)
    }


def _suggest_response_emotion(user_emotions: List[str]) -> EmotionalState:
    """Suggest appropriate emotional response based on user's state."""
    if "confused" in user_emotions:
        return EmotionalState.SERENITY  # Calm guidance
    elif "excited" in user_emotions:
        return EmotionalState.FERVOR  # Match their energy
    elif "frustrated" in user_emotions:
        return EmotionalState.CONTEMPLATION  # Thoughtful support
    else:
        return EmotionalState.WONDER  # Default to curious exploration


@tool
def check_spoiler_safety(
    book: int, 
    chapter: int,
    reading_progress: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if discussing a particular section would spoil the story.
    
    Args:
        book: Book number to check
        chapter: Chapter number to check  
        reading_progress: User's current progress
        
    Returns:
        Safety check results
    """
    progress = ReadingProgress(**reading_progress)
    is_safe = progress.can_discuss(book, chapter)
    
    result = {
        "is_safe": is_safe,
        "user_progress": f"Book {progress.current_book}, Chapter {progress.current_chapter}",
        "requested_section": f"Book {book}, Chapter {chapter}"
    }
    
    if not is_safe:
        result["warning"] = "This section contains spoilers for content the user hasn't read yet."
        result["suggestion"] = "Redirect the conversation to explored sections or ask about their current reading."
        
    return result


@tool
def generate_image_prompt(
    theme: str,
    emotion: str,
    style_notes: str = "Russian literary realism with expressionist touches"
) -> Dict[str, str]:
    """
    Generate a prompt for image creation based on current discussion.
    
    Args:
        theme: Main theme or subject
        emotion: Current emotional state
        style_notes: Consistent style guidelines
        
    Returns:
        Image generation specifications
    """
    # Map emotions to visual characteristics
    emotion_visuals = {
        "WONDER": "ethereal lighting, soft focus, ascending composition",
        "ANGUISH": "sharp contrasts, dark shadows, fractured elements",
        "ECSTASY": "vibrant colors, swirling movement, luminous effects",
        "CONTEMPLATION": "muted palette, centered composition, subtle details",
        "MELANCHOLY": "blue-grey tones, rain or mist, solitary figures",
        "FERVOR": "warm colors, dynamic angles, intense expressions",
        "SERENITY": "balanced composition, natural light, peaceful scenes",
        "TURMOIL": "chaotic elements, multiple perspectives, storm imagery",
        "RAPTURE": "golden light, upward movement, transcendent imagery",
        "DESPAIR": "void spaces, downward composition, absence of light"
    }
    
    visual_style = emotion_visuals.get(emotion, "atmospheric and evocative")
    
    prompt = f"{theme}, {visual_style}, {style_notes}"
    
    return {
        "prompt": prompt,
        "emotion": emotion,
        "timestamp": datetime.now().isoformat(),
        "style_consistency": style_notes
    }


@tool
def extract_themes(text: str) -> List[str]:
    """
    Extract major themes from a text passage for tracking.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of identified themes
    """
    # Major themes in Brothers Karamazov
    theme_keywords = {
        "faith": ["god", "faith", "belief", "religion", "prayer", "soul"],
        "doubt": ["doubt", "atheism", "question", "uncertainty", "skeptic"],
        "love": ["love", "heart", "passion", "devotion", "affection"],
        "suffering": ["suffer", "pain", "anguish", "torment", "agony"],
        "freedom": ["free", "liberty", "choice", "will", "decision"],
        "guilt": ["guilt", "conscience", "shame", "responsibility", "blame"],
        "family": ["father", "brother", "son", "family", "blood"],
        "morality": ["good", "evil", "right", "wrong", "moral", "ethics"],
        "redemption": ["forgive", "redeem", "salvation", "grace", "mercy"],
        "madness": ["mad", "insane", "crazy", "hysteria", "breakdown"]
    }
    
    text_lower = text.lower()
    themes = []
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            themes.append(theme)
            
    return themes


@tool
def format_socratic_question(
    topic: str,
    user_level: str = "intermediate",
    previous_questions: List[str] = []
) -> str:
    """
    Generate a Socratic question based on topic and user engagement.
    
    Args:
        topic: Current topic of discussion
        user_level: User's demonstrated understanding level
        previous_questions: Recently asked questions to avoid repetition
        
    Returns:
        A well-crafted Socratic question
    """
    question_templates = {
        "beginner": [
            "What do you think {topic} means in this context?",
            "How does {topic} make you feel as you read?",
            "Can you describe {topic} in your own words?"
        ],
        "intermediate": [
            "How might {topic} connect to the larger themes we've discussed?",
            "What would happen if {topic} were different?",
            "Why do you think Dostoevsky chose to present {topic} this way?"
        ],
        "advanced": [
            "How does {topic} challenge our conventional understanding?",
            "What paradox does {topic} reveal about human nature?",
            "In what ways does {topic} transcend its immediate context?"
        ]
    }
    
    templates = question_templates.get(user_level, question_templates["intermediate"])
    
    # Select a template not recently used
    for template in templates:
        question = template.format(topic=topic)
        if question not in previous_questions:
            return question
            
    # Fallback
    return f"What new insight about {topic} emerges as we discuss it?"


@tool
def plan_ui_update(
    emotion: str,
    intensity: float,
    action: str = "transition"
) -> Dict[str, Any]:
    """
    Plan UI updates based on emotional state.
    
    Args:
        emotion: Current emotional state
        intensity: Emotional intensity (0-1)
        action: Type of UI action
        
    Returns:
        UI update directives
    """
    ui_themes = {
        "WONDER": {
            "primary_color": "#E6E6FA",  # Lavender
            "secondary_color": "#B19CD9",  # Purple
            "animation": "float",
            "particle_effect": "stardust",
            "shape_morph": "expanding_sphere"
        },
        "ANGUISH": {
            "primary_color": "#8B0000",  # Dark red
            "secondary_color": "#2F4F4F",  # Dark grey
            "animation": "shake",
            "particle_effect": "shatter",
            "shape_morph": "jagged_fragments"
        },
        "ECSTASY": {
            "primary_color": "#FFD700",  # Gold
            "secondary_color": "#FF69B4",  # Hot pink
            "animation": "pulse_rapid",
            "particle_effect": "fireworks",
            "shape_morph": "explosive_bloom"
        },
        "CONTEMPLATION": {
            "primary_color": "#708090",  # Slate grey
            "secondary_color": "#F5DEB3",  # Wheat
            "animation": "breathe",
            "particle_effect": "gentle_motes",
            "shape_morph": "slow_rotation"
        }
    }
    
    theme = ui_themes.get(emotion, ui_themes["CONTEMPLATION"])
    
    return {
        "action": action,
        "theme": theme,
        "intensity": intensity,
        "duration": 2000 if action == "transition" else 500,
        "easing": "ease-in-out" if intensity < 0.7 else "ease-out"
    }