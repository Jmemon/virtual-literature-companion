"""
Virtual Literature Companion - Wild Genius Professor Module

This module implements an emotionally-aware AI literature professor persona
that can engage in deep, Socratic dialogue about literary works while
maintaining and evolving emotional states based on interactions.

Core Components:
- EmotionalState: Advanced emotional state management with 15+ dimensions
- ProfessorPersona: The main professor personality with emotional awareness
- WildGeniusProfessorGraph: LangGraph workflow with emotional processing
- State management for conversation context and emotional memory

Key Features:
- Multi-dimensional emotional tracking (core emotions, cognitive states, social dimensions)
- Exponential decay for natural emotional cooling
- Contextual emotional triggers based on interaction types
- Emotional memory for significant events
- Adaptive response generation based on current emotional state
"""

from .state import (
    EmotionalState,
    Message,
    ConversationContext,
    EmotionalProcessor,
    extract_emotions_from_text,
    calculate_emotional_intensity,
    update_conversation_context,
    create_initial_state
)

from .persona import ProfessorPersona
from .graph import WildGeniusProfessorGraph
from .config import config
from .tools import literature_tools

__version__ = "0.1.0"

__all__ = [
    "EmotionalState",
    "Message", 
    "ConversationContext",
    "EmotionalProcessor",
    "extract_emotions_from_text",
    "calculate_emotional_intensity", 
    "update_conversation_context",
    "create_initial_state",
    "ProfessorPersona",
    "WildGeniusProfessorGraph",
    "config",
    "literature_tools"
]