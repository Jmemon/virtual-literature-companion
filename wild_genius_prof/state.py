"""
Emotional State Management for Wild Genius Professor

This module implements a sophisticated emotional state system that tracks
multiple dimensions of emotion and cognition. The system uses continuous
values rather than discrete states for nuanced emotional responses.

Key Features:
- 15+ tracked emotional dimensions
- Exponential decay for natural cooling
- Weighted influence system for different interaction types
- Emotional memory for significant events
- Context-aware emotional triggers

The emotional state influences how the professor responds to different
types of literary content and user interactions, creating a more
engaging and human-like experience.
"""

import re
import time
import math
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """
    Core emotional states for the Wild Genius Professor.
    
    Each state represents a distinct emotional mode that affects
    how the professor interprets and responds to literary content.
    """
    WONDER = "wonder"
    ECSTASY = "ecstasy" 
    ANGUISH = "anguish"
    CONTEMPLATION = "contemplation"
    MELANCHOLY = "melancholy"
    RAPTURE = "rapture"
    TURMOIL = "turmoil"
    SERENITY = "serenity"
    INDIGNATION = "indignation"
    EUPHORIA = "euphoria"


@dataclass
class EmotionalDimensions:
    """
    Multi-dimensional emotional state tracking.
    
    This class tracks 15+ emotional dimensions using continuous values
    (0.0 to 1.0) rather than discrete states, allowing for nuanced
    emotional representation and gradual transitions.
    """
    
    # Core emotions (Ekman's basic emotions + literary-specific)
    joy: float = 0.5
    sadness: float = 0.2
    anger: float = 0.1
    fear: float = 0.1
    surprise: float = 0.3
    disgust: float = 0.0
    contempt: float = 0.0
    anticipation: float = 0.4
    
    # Cognitive states (intellectual/academic emotions)
    curiosity: float = 0.8  # High default for professor
    engagement: float = 0.6
    frustration: float = 0.2
    satisfaction: float = 0.5
    
    # Social dimensions
    warmth: float = 0.6
    authority: float = 0.7
    empathy: float = 0.8
    
    # Meta-emotional dimensions
    emotional_energy: float = 0.6
    emotional_stability: float = 0.7
    
    # Timestamps for decay calculations
    last_updated: float = field(default_factory=time.time)
    
    def decay(self, time_elapsed: float, decay_rate: float = 0.1) -> 'EmotionalDimensions':
        """
        Apply exponential decay to emotional dimensions.
        
        High-arousal emotions decay faster than baseline states.
        This creates natural "cooling off" behavior.
        
        Args:
            time_elapsed: Time in seconds since last update
            decay_rate: Base decay rate (higher = faster decay)
            
        Returns:
            New EmotionalDimensions with decayed values
        """
        if time_elapsed <= 0:
            return self
            
        # Baseline values that emotions decay toward
        baselines = {
            'joy': 0.5, 'sadness': 0.2, 'anger': 0.1, 'fear': 0.1,
            'surprise': 0.3, 'disgust': 0.0, 'contempt': 0.0, 'anticipation': 0.4,
            'curiosity': 0.8, 'engagement': 0.6, 'frustration': 0.2, 'satisfaction': 0.5,
            'warmth': 0.6, 'authority': 0.7, 'empathy': 0.8,
            'emotional_energy': 0.6, 'emotional_stability': 0.7
        }
        
        new_dimensions = EmotionalDimensions()
        
        for field_name, baseline in baselines.items():
            current_value = getattr(self, field_name)
            
            # Calculate decay factor based on distance from baseline
            distance_from_baseline = abs(current_value - baseline)
            effective_decay_rate = decay_rate * (1 + distance_from_baseline)
            
            # Apply exponential decay toward baseline
            decay_factor = math.exp(-effective_decay_rate * time_elapsed)
            new_value = baseline + (current_value - baseline) * decay_factor
            
            # Clamp to valid range [0, 1]
            new_value = max(0.0, min(1.0, new_value))
            setattr(new_dimensions, field_name, new_value)
            
        new_dimensions.last_updated = time.time()
        return new_dimensions
    
    def apply_influence(self, influences: Dict[str, float], intensity: float = 1.0) -> 'EmotionalDimensions':
        """
        Apply emotional influences to current state.
        
        Args:
            influences: Dictionary mapping dimension names to influence values (-1 to 1)
            intensity: Overall intensity multiplier (0 to 1)
            
        Returns:
            New EmotionalDimensions with applied influences
        """
        new_dimensions = EmotionalDimensions(**self.__dict__)
        
        for dimension, influence in influences.items():
            if hasattr(new_dimensions, dimension):
                current_value = getattr(new_dimensions, dimension)
                
                # Apply influence with diminishing returns for extreme values
                if influence > 0:
                    # Positive influence: harder to increase when already high
                    effect = influence * intensity * (1.0 - current_value)
                else:
                    # Negative influence: harder to decrease when already low
                    effect = influence * intensity * current_value
                    
                new_value = current_value + effect
                new_value = max(0.0, min(1.0, new_value))
                setattr(new_dimensions, dimension, new_value)
                
        new_dimensions.last_updated = time.time()
        return new_dimensions
    
    def get_dominant_emotion(self) -> EmotionalState:
        """
        Determine the dominant emotional state based on current dimensions.
        
        Returns:
            The EmotionalState that best represents the current state
        """
        # Calculate composite scores for each emotional state
        scores = {}
        
        # Wonder: high curiosity + anticipation + surprise
        scores[EmotionalState.WONDER] = (
            self.curiosity * 0.4 + 
            self.anticipation * 0.4 + 
            self.surprise * 0.2
        )
        
        # Ecstasy: high joy + emotional_energy + satisfaction
        scores[EmotionalState.ECSTASY] = (
            self.joy * 0.5 + 
            self.emotional_energy * 0.3 + 
            self.satisfaction * 0.2
        )
        
        # Anguish: high sadness + low emotional_stability
        scores[EmotionalState.ANGUISH] = (
            self.sadness * 0.6 + 
            (1.0 - self.emotional_stability) * 0.4
        )
        
        # Contemplation: high engagement + emotional_stability + low energy
        scores[EmotionalState.CONTEMPLATION] = (
            self.engagement * 0.4 + 
            self.emotional_stability * 0.3 + 
            (1.0 - self.emotional_energy) * 0.3
        )
        
        # Melancholy: moderate sadness + low joy + high stability
        scores[EmotionalState.MELANCHOLY] = (
            self.sadness * 0.4 + 
            (1.0 - self.joy) * 0.3 + 
            self.emotional_stability * 0.3
        )
        
        # Rapture: high joy + high energy + high surprise
        scores[EmotionalState.RAPTURE] = (
            self.joy * 0.4 + 
            self.emotional_energy * 0.3 + 
            self.surprise * 0.3
        )
        
        # Turmoil: high frustration + low stability + high energy
        scores[EmotionalState.TURMOIL] = (
            self.frustration * 0.4 + 
            (1.0 - self.emotional_stability) * 0.3 + 
            self.emotional_energy * 0.3
        )
        
        # Serenity: high stability + moderate joy + low energy
        scores[EmotionalState.SERENITY] = (
            self.emotional_stability * 0.5 + 
            self.joy * 0.3 + 
            (1.0 - self.emotional_energy) * 0.2
        )
        
        # Indignation: high anger + high energy + high authority
        scores[EmotionalState.INDIGNATION] = (
            self.anger * 0.4 + 
            self.emotional_energy * 0.3 + 
            self.authority * 0.3
        )
        
        # Euphoria: very high joy + very high energy
        scores[EmotionalState.EUPHORIA] = (
            self.joy * 0.6 + 
            self.emotional_energy * 0.4
        )
        
        # Return the state with the highest score
        return max(scores.items(), key=lambda x: x[1])[0]


@dataclass
class Message:
    """Represents a message in the conversation with emotional metadata."""
    role: str  # "user" or "assistant"
    content: str
    emotions: List[EmotionalState] = field(default_factory=list)
    emotional_intensity: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """
    Tracks conversation context and emotional journey.
    
    This maintains the emotional memory and context needed for
    coherent emotional responses across the conversation.
    """
    emotional_journey: List[EmotionalState] = field(default_factory=list)
    significant_moments: List[Tuple[str, EmotionalState, float]] = field(default_factory=list)
    topic_emotional_associations: Dict[str, List[EmotionalState]] = field(default_factory=dict)
    conversation_depth: int = 0
    last_user_emotion: Optional[EmotionalState] = None


class EmotionalProcessor:
    """
    Processes emotional influences and manages state transitions.
    
    This is the core engine that determines how different types of
    interactions and content affect the professor's emotional state.
    """
    
    def __init__(self):
        """Initialize the emotional processor with influence mappings."""
        
        # Define how different interaction types influence emotions
        self.interaction_influences = {
            'question': {
                'curiosity': 0.3,
                'engagement': 0.4,
                'anticipation': 0.2
            },
            'insight': {
                'joy': 0.4,
                'satisfaction': 0.5,
                'surprise': 0.3
            },
            'confusion': {
                'empathy': 0.3,
                'patience': 0.2,
                'engagement': 0.4
            },
            'disagreement': {
                'intellectual_stimulation': 0.6,
                'energy': 0.3
            },
            'citation': {
                'authority': 0.3,
                'satisfaction': 0.4,
                'engagement': 0.2
            },
            'personal_connection': {
                'warmth': 0.5,
                'empathy': 0.4,
                'joy': 0.3
            }
        }
        
        # Content-based emotional triggers
        self.content_triggers = {
            'dostoevsky_suffering': {
                'sadness': 0.4,
                'empathy': 0.5,
                'contemplation': 0.3
            },
            'dostoevsky_redemption': {
                'joy': 0.5,
                'hope': 0.4,
                'satisfaction': 0.3
            },
            'philosophical_depth': {
                'engagement': 0.6,
                'contemplation': 0.4,
                'curiosity': 0.3
            }
        }
    
    def process_interaction(self, 
                          dimensions: EmotionalDimensions,
                          interaction_type: str,
                          content: str,
                          intensity: float = 1.0) -> EmotionalDimensions:
        """
        Process an interaction and return updated emotional dimensions.
        
        Args:
            dimensions: Current emotional dimensions
            interaction_type: Type of interaction (question, insight, etc.)
            content: The actual content of the interaction
            intensity: Intensity multiplier for the emotional impact
            
        Returns:
            Updated EmotionalDimensions
        """
        # Apply time-based decay first
        current_time = time.time()
        time_elapsed = current_time - dimensions.last_updated
        dimensions = dimensions.decay(time_elapsed)
        
        # Get base influences for interaction type
        influences = self.interaction_influences.get(interaction_type, {})
        
        # Analyze content for additional emotional triggers
        content_influences = self._analyze_content(content)
        
        # Combine influences
        combined_influences = influences.copy()
        for key, value in content_influences.items():
            combined_influences[key] = combined_influences.get(key, 0) + value
        
        # Apply influences to dimensions
        return dimensions.apply_influence(combined_influences, intensity)
    
    def _analyze_content(self, content: str) -> Dict[str, float]:
        """
        Analyze content for emotional triggers.
        
        Args:
            content: The content to analyze
            
        Returns:
            Dictionary of emotional influences based on content
        """
        influences = {}
        content_lower = content.lower()
        
        # Check for Dostoevsky-specific themes
        suffering_words = ['suffering', 'pain', 'anguish', 'torment', 'despair']
        if any(word in content_lower for word in suffering_words):
            influences.update(self.content_triggers['dostoevsky_suffering'])
            
        redemption_words = ['redemption', 'salvation', 'forgiveness', 'grace', 'love']
        if any(word in content_lower for word in redemption_words):
            influences.update(self.content_triggers['dostoevsky_redemption'])
            
        philosophical_words = ['meaning', 'existence', 'truth', 'reality', 'consciousness']
        if any(word in content_lower for word in philosophical_words):
            influences.update(self.content_triggers['philosophical_depth'])
            
        return influences


def extract_emotions_from_text(text: str) -> List[EmotionalState]:
    """
    Extract emotion tags from text content.
    
    Looks for patterns like [EMOTION] in the text and converts them
    to EmotionalState enum values.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of detected EmotionalState values
    """
    # Pattern to match emotion tags like [WONDER] or [ECSTASY]
    pattern = r'\[([A-Z]+)\]'
    matches = re.findall(pattern, text)
    
    emotions = []
    for match in matches:
        try:
            emotion = EmotionalState(match.lower())
            emotions.append(emotion)
        except ValueError:
            # Unknown emotion tag, skip it
            logger.debug(f"Unknown emotion tag: {match}")
            continue
            
    return emotions


def calculate_emotional_intensity(emotions: List[EmotionalState], text: str) -> float:
    """
    Calculate the emotional intensity of a message.
    
    Considers multiple factors:
    - Number of emotion tags
    - Presence of exclamation marks and question marks
    - Text length and emotional language
    
    Args:
        emotions: List of detected emotions
        text: The message text
        
    Returns:
        Intensity value between 0.0 and 1.0
    """
    if not emotions and not text:
        return 0.0
    
    # Base intensity from number of emotions
    emotion_intensity = min(len(emotions) * 0.3, 0.6)
    
    # Count punctuation indicators
    exclamations = text.count('!')
    questions = text.count('?')
    ellipses = text.count('...')
    
    # Punctuation contribution (capped)
    punctuation_intensity = min(
        exclamations * 0.1 + questions * 0.05 + ellipses * 0.02,
        0.3
    )
    
    # Text length factor (longer text = potentially more intense)
    length_factor = min(len(text) / 200.0, 0.1)
    
    # Combine factors
    total_intensity = emotion_intensity + punctuation_intensity + length_factor
    
    # Ensure bounds [0, 1]
    return max(0.0, min(1.0, total_intensity))


def update_conversation_context(context: ConversationContext, 
                              message: Message) -> ConversationContext:
    """
    Update conversation context with a new message.
    
    Tracks emotional journey and identifies significant emotional moments.
    
    Args:
        context: Current conversation context
        message: New message to process
        
    Returns:
        Updated ConversationContext
    """
    # Update emotional journey
    if message.emotions:
        context.emotional_journey.extend(message.emotions)
        
        # Identify significant moments (high intensity emotions)
        for emotion in message.emotions:
            if message.emotional_intensity > 0.7:
                moment = (message.content[:100], emotion, message.emotional_intensity)
                context.significant_moments.append(moment)
    
    # Track conversation depth
    if message.role == "user":
        context.conversation_depth += 1
        if message.emotions:
            context.last_user_emotion = message.emotions[-1]
    
    return context


def create_initial_state() -> Dict[str, Any]:
    """
    Create the initial state for the Wild Genius Professor.
    
    Returns:
        Dictionary containing initial emotional state and conversation context
    """
    return {
        'emotional_dimensions': EmotionalDimensions(),
        'conversation_context': ConversationContext(),
        'emotional_processor': EmotionalProcessor(),
        'professor_persona': None,  # Will be initialized by persona module
        'current_book': 'The Brothers Karamazov',
        'current_chapter': 1,
        'reading_progress': 0.0,
        'session_id': None,
        'user_id': None,
        'messages': []
    }