"""
Configuration management for Wild Genius Professor.

This module provides centralized configuration management with
environment variable support and sensible defaults for the
emotional AI literature companion.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class Config:
    """
    Configuration class for the Wild Genius Professor system.
    
    Manages all configurable parameters including LLM settings,
    emotional processing parameters, and system behavior.
    """
    
    # System Configuration
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # LLM Configuration
    default_llm_provider: str = "openai"
    default_model: str = "gpt-4-turbo-preview"
    max_tokens: int = 2000
    temperature: float = 0.8  # Higher for creative responses
    
    # Emotional Processing Parameters
    emotional_decay_rate: float = 0.1
    emotional_intensity_threshold: float = 0.7
    max_emotional_memory: int = 50
    
    # Professor Personality Parameters
    base_curiosity: float = 0.8
    base_warmth: float = 0.6
    base_authority: float = 0.7
    base_empathy: float = 0.8
    
    # Literature-specific Configuration
    current_focus_book: str = "The Brothers Karamazov"
    supported_books: Optional[List[str]] = None
    citation_style: str = "academic"
    
    # Memory and Persistence
    honcho_enabled: bool = os.getenv("HONCHO_API_KEY") is not None
    session_timeout: int = 3600  # 1 hour
    
    # Response Generation
    use_emotional_prefixes: bool = True
    include_citations: bool = True
    socratic_mode: bool = True
    
    def __post_init__(self):
        """Initialize computed properties after dataclass creation."""
        if self.supported_books is None:
            self.supported_books = [
                "The Brothers Karamazov",
                "Crime and Punishment", 
                "The Idiot",
                "Demons",
                "Notes from Underground"
            ]
    
    def get_emotional_config(self) -> Dict[str, float]:
        """
        Get emotional processing configuration.
        
        Returns:
            Dictionary of emotional processing parameters
        """
        return {
            'decay_rate': self.emotional_decay_rate,
            'intensity_threshold': self.emotional_intensity_threshold,
            'base_curiosity': self.base_curiosity,
            'base_warmth': self.base_warmth,
            'base_authority': self.base_authority,
            'base_empathy': self.base_empathy
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for model initialization.
        
        Returns:
            Dictionary of LLM parameters
        """
        return {
            'provider': self.default_llm_provider,
            'model': self.default_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        
        # Update LLM settings
        llm_provider = os.getenv("LLM_PROVIDER")
        if llm_provider:
            self.default_llm_provider = llm_provider
            
        llm_model = os.getenv("LLM_MODEL")
        if llm_model:
            self.default_model = llm_model
            
        llm_temp = os.getenv("LLM_TEMPERATURE")
        if llm_temp:
            try:
                self.temperature = float(llm_temp)
            except ValueError:
                logging.warning("Invalid LLM_TEMPERATURE value, using default")
        
        # Update emotional parameters
        decay_rate = os.getenv("EMOTIONAL_DECAY_RATE")
        if decay_rate:
            try:
                self.emotional_decay_rate = float(decay_rate)
            except ValueError:
                logging.warning("Invalid EMOTIONAL_DECAY_RATE value, using default")
                
        # Update book focus
        focus_book = os.getenv("FOCUS_BOOK")
        if focus_book:
            self.current_focus_book = focus_book


# Global configuration instance
config = Config()

# Update from environment on import
config.update_from_env()

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)