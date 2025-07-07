"""
Wild Genius Professor - Core AI persona for the Virtual Literature Companion.
"""

from .graph import WildGeniusProfessorGraph
from .persona import WildGeniusProfessorPersona
from .state import (
    AgentState,
    EmotionalState,
    Message,
    Citation,
    ReadingProgress,
    ConversationContext,
    ProfessorMemory,
    create_initial_state
)
from .config import config, Config

__all__ = [
    "WildGeniusProfessorGraph",
    "WildGeniusProfessorPersona",
    "AgentState",
    "EmotionalState", 
    "Message",
    "Citation",
    "ReadingProgress",
    "ConversationContext",
    "ProfessorMemory",
    "create_initial_state",
    "config",
    "Config"
]

__version__ = "0.1.0"