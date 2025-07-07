"""
Advanced Emotional Intelligence System for the Wild Genius Professor

This module implements a sophisticated emotional state management system with:
- Multi-dimensional continuous emotional states (15+ tracked dimensions)
- Exponential decay for natural emotional cooling over time
- Weighted influence system for different interaction types
- Emotional memory for significant events
- Adaptive response generation based on emotional context
- Integration with existing persona and state management

The system tracks core emotions (joy, sadness, anger, fear, etc.), cognitive states
(curiosity, engagement, frustration), social dimensions (warmth, authority, empathy),
and energy/stability metrics to provide nuanced emotional responses.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field

from .state import EmotionalState as LegacyEmotionalState


class InteractionType(str, Enum):
    """Types of interactions that influence emotional state."""
    QUESTION = "question"
    INSIGHT = "insight"
    CONFUSION = "confusion"
    DISAGREEMENT = "disagreement"
    ENTHUSIASM = "enthusiasm"
    CITATION_REQUEST = "citation_request"
    THEME_EXPLORATION = "theme_exploration"
    PERSONAL_REFLECTION = "personal_reflection"
    BREAKTHROUGH = "breakthrough"
    FRUSTRATION = "frustration"


@dataclass
class EmotionalInfluence:
    """Represents an influence on emotional state from an interaction."""
    interaction_type: InteractionType
    intensity: float  # 0.0 to 1.0
    emotional_weights: Dict[str, float]  # emotion_name -> influence_strength
    timestamp: datetime = field(default_factory=datetime.now)
    context: str = ""
    
    def apply_decay(self, time_elapsed: float) -> float:
        """Apply exponential decay based on time elapsed (in minutes)."""
        # Decay half-life of 30 minutes for most emotions
        half_life = 30.0
        decay_factor = math.exp(-time_elapsed * math.log(2) / half_life)
        return decay_factor


class EmotionalMemory(BaseModel):
    """Tracks significant emotional events for contextual awareness."""
    significant_events: List[Dict[str, Any]] = Field(default_factory=list)
    emotional_patterns: Dict[str, float] = Field(default_factory=dict)
    baseline_drift: Dict[str, float] = Field(default_factory=dict)
    adaptation_rate: float = 0.1
    
    def record_significant_event(
        self, 
        emotion_name: str, 
        intensity: float, 
        context: str,
        interaction_type: InteractionType
    ):
        """Record a significant emotional event."""
        if intensity > 0.7:  # Only record high-intensity events
            event = {
                "timestamp": datetime.now().isoformat(),
                "emotion": emotion_name,
                "intensity": intensity,
                "context": context,
                "interaction_type": interaction_type.value
            }
            self.significant_events.append(event)
            
            # Keep only last 50 significant events
            if len(self.significant_events) > 50:
                self.significant_events = self.significant_events[-50:]
    
    def update_patterns(self, current_state: Dict[str, float]):
        """Update long-term emotional patterns."""
        for emotion, value in current_state.items():
            if emotion not in self.emotional_patterns:
                self.emotional_patterns[emotion] = value
            else:
                # Exponential moving average
                self.emotional_patterns[emotion] = (
                    (1 - self.adaptation_rate) * self.emotional_patterns[emotion] +
                    self.adaptation_rate * value
                )


class AdvancedEmotionalState(BaseModel):
    """
    Sophisticated multi-dimensional emotional state system.
    
    Tracks 15+ emotional dimensions with continuous values, implementing
    exponential decay, weighted influences, and adaptive thresholds.
    """
    
    # Core emotions (Plutchik's model extended)
    joy: float = Field(default=0.3, ge=0.0, le=1.0)
    sadness: float = Field(default=0.2, ge=0.0, le=1.0)
    anger: float = Field(default=0.1, ge=0.0, le=1.0)
    fear: float = Field(default=0.1, ge=0.0, le=1.0)
    surprise: float = Field(default=0.2, ge=0.0, le=1.0)
    disgust: float = Field(default=0.05, ge=0.0, le=1.0)
    contempt: float = Field(default=0.1, ge=0.0, le=1.0)
    anticipation: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # Cognitive states
    curiosity: float = Field(default=0.7, ge=0.0, le=1.0)
    engagement: float = Field(default=0.5, ge=0.0, le=1.0)
    frustration: float = Field(default=0.1, ge=0.0, le=1.0)
    satisfaction: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # Social dimensions
    warmth: float = Field(default=0.6, ge=0.0, le=1.0)
    authority: float = Field(default=0.7, ge=0.0, le=1.0)
    empathy: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Energy and stability
    energy: float = Field(default=0.6, ge=0.0, le=1.0)
    stability: float = Field(default=0.7, ge=0.0, le=1.0)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Metadata
    last_update: datetime = Field(default_factory=datetime.now)
    active_influences: List[EmotionalInfluence] = Field(default_factory=list)
    memory: EmotionalMemory = Field(default_factory=EmotionalMemory)
    
    # Configuration
    decay_rates: Dict[str, float] = Field(default_factory=lambda: {
        # Fast-changing emotions
        "surprise": 0.8,
        "anger": 0.7,
        "fear": 0.6,
        
        # Medium-changing emotions  
        "joy": 0.5,
        "sadness": 0.4,
        "frustration": 0.6,
        "satisfaction": 0.5,
        
        # Slow-changing states
        "curiosity": 0.3,
        "engagement": 0.3,
        "warmth": 0.2,
        "authority": 0.1,
        "empathy": 0.2,
        "energy": 0.4,
        "stability": 0.2
    })
    
    def apply_decay(self) -> "AdvancedEmotionalState":
        """Apply exponential decay to all emotional dimensions."""
        now = datetime.now()
        time_elapsed = (now - self.last_update).total_seconds() / 60.0  # minutes
        
        if time_elapsed < 0.1:  # Less than 6 seconds, skip decay
            return self
        
        # Apply decay to each dimension
        emotional_state = self.dict()
        for dimension, value in emotional_state.items():
            if isinstance(value, float) and dimension in self.decay_rates:
                decay_rate = self.decay_rates[dimension]
                half_life = 30.0 / decay_rate  # Adjusted half-life
                decay_factor = math.exp(-time_elapsed * math.log(2) / half_life)
                
                # Decay towards baseline (different for each emotion)
                baseline = self._get_baseline(dimension)
                decayed_value = baseline + (value - baseline) * decay_factor
                emotional_state[dimension] = max(0.0, min(1.0, decayed_value))
        
        # Update timestamp
        emotional_state["last_update"] = now
        
        # Clean up old influences
        self._cleanup_influences()
        
        return AdvancedEmotionalState(**emotional_state)
    
    def _get_baseline(self, dimension: str) -> float:
        """Get the baseline/neutral value for each emotional dimension."""
        baselines = {
            # Core emotions - generally low baselines
            "joy": 0.3, "sadness": 0.2, "anger": 0.1, "fear": 0.1,
            "surprise": 0.1, "disgust": 0.05, "contempt": 0.1, "anticipation": 0.3,
            
            # Cognitive states - professor's natural curiosity and engagement
            "curiosity": 0.7, "engagement": 0.5, "frustration": 0.1, "satisfaction": 0.4,
            
            # Social dimensions - professor's characteristic warmth and authority
            "warmth": 0.6, "authority": 0.7, "empathy": 0.8,
            
            # Energy and stability - moderate defaults
            "energy": 0.6, "stability": 0.7, "intensity": 0.3
        }
        return baselines.get(dimension, 0.5)
    
    def apply_influence(self, influence: EmotionalInfluence) -> "AdvancedEmotionalState":
        """Apply an emotional influence to the current state."""
        # First apply any pending decay
        current_state = self.apply_decay()
        
        # Apply the influence
        emotional_state = current_state.dict()
        
        for emotion_name, weight in influence.emotional_weights.items():
            if emotion_name in emotional_state and isinstance(emotional_state[emotion_name], float):
                # Calculate influence magnitude
                influence_magnitude = influence.intensity * weight
                
                # Apply with diminishing returns for extreme values
                current_value = emotional_state[emotion_name]
                if influence_magnitude > 0:
                    # Positive influence - diminishing returns as we approach 1.0
                    new_value = current_value + influence_magnitude * (1.0 - current_value)
                else:
                    # Negative influence - diminishing returns as we approach 0.0
                    new_value = current_value + influence_magnitude * current_value
                
                emotional_state[emotion_name] = max(0.0, min(1.0, new_value))
        
        # Update intensity based on overall emotional activation
        total_activation = sum(
            emotional_state[dim] for dim in ["joy", "sadness", "anger", "fear", "surprise"]
        ) / 5.0
        emotional_state["intensity"] = min(1.0, total_activation * 1.2)
        
        # Record significant events
        for emotion_name, weight in influence.emotional_weights.items():
            if abs(weight) * influence.intensity > 0.5:  # Significant influence
                current_state.memory.record_significant_event(
                    emotion_name=emotion_name,
                    intensity=abs(weight) * influence.intensity,
                    context=influence.context,
                    interaction_type=influence.interaction_type
                )
        
        # Add to active influences
        emotional_state["active_influences"].append(influence)
        emotional_state["last_update"] = datetime.now()
        
        # Update memory patterns
        emotion_values = {k: v for k, v in emotional_state.items() 
                         if isinstance(v, float) and k in self.decay_rates}
        current_state.memory.update_patterns(emotion_values)
        
        return AdvancedEmotionalState(**emotional_state)
    
    def _cleanup_influences(self):
        """Remove influences older than 2 hours."""
        cutoff_time = datetime.now() - timedelta(hours=2)
        self.active_influences = [
            inf for inf in self.active_influences 
            if inf.timestamp > cutoff_time
        ]
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the currently dominant emotion and its intensity."""
        core_emotions = {
            "joy": self.joy,
            "sadness": self.sadness, 
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "disgust": self.disgust,
            "contempt": self.contempt,
            "anticipation": self.anticipation
        }
        
        dominant_emotion = max(core_emotions.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def get_cognitive_state(self) -> Dict[str, float]:
        """Get current cognitive state dimensions."""
        return {
            "curiosity": self.curiosity,
            "engagement": self.engagement,
            "frustration": self.frustration,
            "satisfaction": self.satisfaction
        }
    
    def get_social_state(self) -> Dict[str, float]:
        """Get current social interaction dimensions."""
        return {
            "warmth": self.warmth,
            "authority": self.authority,
            "empathy": self.empathy
        }
    
    def to_legacy_emotion(self) -> LegacyEmotionalState:
        """Convert to legacy emotional state for backwards compatibility."""
        dominant_emotion, intensity = self.get_dominant_emotion()
        
        # Map to legacy emotions with some logic
        if self.curiosity > 0.6 and dominant_emotion in ["anticipation", "joy"]:
            return LegacyEmotionalState.WONDER
        elif dominant_emotion == "joy" and intensity > 0.7:
            return LegacyEmotionalState.ECSTASY
        elif dominant_emotion == "sadness" and intensity > 0.6:
            if self.empathy > 0.7:
                return LegacyEmotionalState.MELANCHOLY
            else:
                return LegacyEmotionalState.DESPAIR
        elif dominant_emotion == "anger" or self.frustration > 0.6:
            return LegacyEmotionalState.TURMOIL
        elif self.engagement > 0.7 and self.curiosity > 0.6:
            return LegacyEmotionalState.FERVOR
        elif self.stability > 0.7 and intensity < 0.4:
            return LegacyEmotionalState.SERENITY
        elif dominant_emotion == "fear" or self.stability < 0.3:
            return LegacyEmotionalState.ANGUISH
        elif dominant_emotion == "joy" and self.satisfaction > 0.8:
            return LegacyEmotionalState.RAPTURE
        else:
            return LegacyEmotionalState.CONTEMPLATION
    
    def generate_emotional_context(self) -> str:
        """Generate a description of current emotional state for prompt context."""
        dominant_emotion, intensity = self.get_dominant_emotion()
        cognitive = self.get_cognitive_state()
        social = self.get_social_state()
        
        context_parts = []
        
        # Primary emotional state
        if intensity > 0.6:
            context_parts.append(f"Experiencing strong {dominant_emotion} (intensity: {intensity:.1f})")
        else:
            context_parts.append(f"Mild {dominant_emotion} present")
        
        # Cognitive state
        if cognitive["curiosity"] > 0.7:
            context_parts.append("high intellectual curiosity")
        if cognitive["engagement"] > 0.7:
            context_parts.append("deeply engaged")
        if cognitive["frustration"] > 0.5:
            context_parts.append("some frustration present")
        
        # Social dimensions
        if social["warmth"] > 0.7:
            context_parts.append("warm and approachable")
        if social["authority"] > 0.8:
            context_parts.append("authoritative presence")
        if social["empathy"] > 0.8:
            context_parts.append("highly empathetic")
        
        # Energy and stability
        if self.energy > 0.7:
            context_parts.append("high energy")
        elif self.energy < 0.3:
            context_parts.append("low energy")
        
        if self.stability < 0.4:
            context_parts.append("emotionally unstable")
        
        return "; ".join(context_parts)


class EmotionalProcessor:
    """
    Processes interactions and determines their emotional influence.
    
    This class analyzes user messages and system events to determine
    how they should influence the professor's emotional state.
    """
    
    def __init__(self):
        self.influence_mappings = self._build_influence_mappings()
    
    def _build_influence_mappings(self) -> Dict[InteractionType, Dict[str, float]]:
        """Build mappings from interaction types to emotional influences."""
        return {
            InteractionType.QUESTION: {
                "curiosity": 0.3,
                "engagement": 0.2,
                "anticipation": 0.2,
                "warmth": 0.1
            },
            
            InteractionType.INSIGHT: {
                "joy": 0.4,
                "satisfaction": 0.5,
                "curiosity": 0.3,
                "engagement": 0.4,
                "warmth": 0.2
            },
            
            InteractionType.CONFUSION: {
                "empathy": 0.4,
                "warmth": 0.3,
                "engagement": 0.2,
                "authority": -0.1,
                "frustration": 0.2
            },
            
            InteractionType.DISAGREEMENT: {
                "anger": 0.2,
                "frustration": 0.3,
                "authority": 0.3,
                "empathy": 0.2,
                "engagement": 0.2
            },
            
            InteractionType.ENTHUSIASM: {
                "joy": 0.5,
                "energy": 0.4,
                "engagement": 0.6,
                "warmth": 0.3,
                "satisfaction": 0.3
            },
            
            InteractionType.CITATION_REQUEST: {
                "authority": 0.4,
                "satisfaction": 0.3,
                "engagement": 0.2,
                "curiosity": 0.1
            },
            
            InteractionType.THEME_EXPLORATION: {
                "curiosity": 0.5,
                "engagement": 0.6,
                "satisfaction": 0.4,
                "anticipation": 0.3,
                "joy": 0.2
            },
            
            InteractionType.PERSONAL_REFLECTION: {
                "empathy": 0.6,
                "warmth": 0.5,
                "satisfaction": 0.3,
                "engagement": 0.4,
                "sadness": 0.1  # Sympathetic response
            },
            
            InteractionType.BREAKTHROUGH: {
                "joy": 0.8,
                "satisfaction": 0.9,
                "energy": 0.6,
                "warmth": 0.4,
                "engagement": 0.7
            },
            
            InteractionType.FRUSTRATION: {
                "frustration": 0.6,
                "empathy": 0.5,
                "warmth": 0.4,
                "engagement": -0.2,
                "satisfaction": -0.3
            }
        }
    
    def analyze_user_message(self, message: str, context: str = "") -> Tuple[InteractionType, float]:
        """
        Analyze a user message to determine interaction type and intensity.
        
        Args:
            message: The user's message content
            context: Additional context about the conversation
            
        Returns:
            Tuple of (interaction_type, intensity)
        """
        message_lower = message.lower()
        
        # Question detection
        if "?" in message:
            if "why" in message_lower or "how" in message_lower:
                return InteractionType.QUESTION, 0.6
            else:
                return InteractionType.QUESTION, 0.4
        
        # Enthusiasm detection
        enthusiasm_indicators = ["!", "amazing", "incredible", "wow", "brilliant", "love"]
        enthusiasm_count = sum(1 for indicator in enthusiasm_indicators if indicator in message_lower)
        if enthusiasm_count >= 2:
            return InteractionType.ENTHUSIASM, min(0.8, 0.3 + enthusiasm_count * 0.1)
        
        # Confusion detection  
        confusion_indicators = ["confused", "don't understand", "unclear", "lost", "help"]
        if any(indicator in message_lower for indicator in confusion_indicators):
            return InteractionType.CONFUSION, 0.5
        
        # Disagreement detection
        disagreement_indicators = ["disagree", "wrong", "but", "however", "actually"]
        if any(indicator in message_lower for indicator in disagreement_indicators):
            return InteractionType.DISAGREEMENT, 0.4
        
        # Insight detection
        insight_indicators = ["realize", "understand", "see", "insight", "aha", "connects"]
        if any(indicator in message_lower for indicator in insight_indicators):
            return InteractionType.INSIGHT, 0.7
        
        # Personal reflection detection
        reflection_indicators = ["feel", "reminds me", "personal", "experience", "relate"]
        if any(indicator in message_lower for indicator in reflection_indicators):
            return InteractionType.PERSONAL_REFLECTION, 0.6
        
        # Theme exploration (default for substantive messages)
        if len(message.strip()) > 50:  # Longer messages likely exploring themes
            return InteractionType.THEME_EXPLORATION, 0.5
        
        # Default to question for shorter messages
        return InteractionType.QUESTION, 0.3
    
    def create_influence(
        self, 
        interaction_type: InteractionType, 
        intensity: float,
        context: str = ""
    ) -> EmotionalInfluence:
        """Create an emotional influence from an interaction."""
        base_weights = self.influence_mappings.get(interaction_type, {})
        
        # Scale weights by intensity
        scaled_weights = {emotion: weight * intensity for emotion, weight in base_weights.items()}
        
        return EmotionalInfluence(
            interaction_type=interaction_type,
            intensity=intensity,
            emotional_weights=scaled_weights,
            context=context
        )
    
    def process_system_event(self, event_type: str, context: Dict[str, Any]) -> Optional[EmotionalInfluence]:
        """Process system events that might influence emotional state."""
        if event_type == "citation_found":
            return self.create_influence(
                InteractionType.CITATION_REQUEST, 
                0.4, 
                f"Found citation for {context.get('topic', 'discussion')}"
            )
        
        elif event_type == "breakthrough_detected":
            return self.create_influence(
                InteractionType.BREAKTHROUGH,
                0.8,
                f"Student breakthrough on {context.get('theme', 'topic')}"
            )
        
        elif event_type == "session_start":
            return EmotionalInfluence(
                interaction_type=InteractionType.THEME_EXPLORATION,
                intensity=0.3,
                emotional_weights={
                    "anticipation": 0.4,
                    "curiosity": 0.3,
                    "warmth": 0.3,
                    "energy": 0.2
                },
                context="New conversation session beginning"
            )
        
        return None


def create_professor_emotional_state() -> AdvancedEmotionalState:
    """Create initial emotional state for the Wild Genius Professor."""
    return AdvancedEmotionalState(
        # Set professor's characteristic emotional baseline
        curiosity=0.8,      # Naturally very curious
        engagement=0.6,     # Generally engaged
        warmth=0.7,         # Warm personality
        authority=0.8,      # Strong academic authority
        empathy=0.9,        # Highly empathetic
        energy=0.7,         # Energetic personality  
        anticipation=0.5,   # Moderate anticipation
        joy=0.4,           # Moderately joyful default
        stability=0.8       # Generally stable
    )