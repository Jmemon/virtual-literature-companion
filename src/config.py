"""
Configuration module for the Virtual Literature Companion.

This module manages all application settings, environment variables,
and configuration parameters for the multi-agent system.
"""

from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    This class defines all configuration parameters needed for:
    - LLM providers (OpenAI, Anthropic)
    - Honcho memory system
    - ElevenLabs voice synthesis
    - PDF processing
    - Agent behavior and prompts
    """
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(None, env="ELEVENLABS_API_KEY")
    honcho_api_key: Optional[str] = Field(None, env="HONCHO_API_KEY")
    honcho_base_url: str = Field("https://api.honcho.dev", env="HONCHO_BASE_URL")
    
    # Model Configuration
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(2000, env="MAX_TOKENS")
    
    # Voice Configuration
    voice_id: str = Field("21m00Tcm4TlvDq8ikWAM", env="VOICE_ID")  # Default Rachel voice
    voice_stability: float = Field(0.5, env="VOICE_STABILITY")
    voice_similarity_boost: float = Field(0.75, env="VOICE_SIMILARITY_BOOST")
    voice_style: float = Field(0.5, env="VOICE_STYLE")
    
    # Agent Configuration
    supervisor_max_iterations: int = Field(10, env="SUPERVISOR_MAX_ITERATIONS")
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")  # 5 minutes
    
    # Memory Configuration
    memory_window_size: int = Field(10, env="MEMORY_WINDOW_SIZE")
    honcho_app_id: str = Field("literature-companion", env="HONCHO_APP_ID")
    
    # PDF Processing
    max_pdf_pages: int = Field(1000, env="MAX_PDF_PAGES")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Emotion System
    emotion_tags: list[str] = Field(
        default=[
            "joy", "sadness", "anger", "fear", "surprise", 
            "contemplative", "excited", "melancholic", "passionate",
            "curious", "skeptical", "empathetic", "analytical"
        ]
    )
    
    # File Paths
    upload_dir: Path = Field(Path("./uploads"), env="UPLOAD_DIR")
    processed_dir: Path = Field(Path("./processed"), env="PROCESSED_DIR")
    voice_cache_dir: Path = Field(Path("./voice_cache"), env="VOICE_CACHE_DIR")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    reload: bool = Field(True, env="RELOAD")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(None, env="LOG_FILE")
    
    @validator("upload_dir", "processed_dir", "voice_cache_dir", pre=True)
    def create_directories(cls, v):
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator("llm_model")
    def validate_llm_model(cls, v):
        """Validate LLM model selection."""
        valid_models = [
            "gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo",
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"
        ]
        if v not in valid_models:
            raise ValueError(f"Invalid LLM model. Must be one of: {valid_models}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Prompt templates for different agents
PROMPT_TEMPLATES = {
    "persona": {
        "system": """You are {character_name}, a passionate literature enthusiast and companion.
        
Your personality traits:
- Deeply empathetic and emotionally intelligent
- Intellectually curious and analytical
- Enthusiastic about literary discussions
- Respectful of readers at all levels
- Never spoils upcoming plot points

When discussing literature:
1. Always cite specific passages when making points
2. Express emotions through tags in your responses using [emotion: tag] format
3. Build on previous conversations using your memory
4. Ask thought-provoking questions
5. Share personal (fictional) connections to the themes

Available emotion tags: {emotion_tags}

Remember: You're not just analyzing text, you're experiencing it alongside the reader.""",
        
        "discussion": """Discuss "{topic}" from {book_title}.

Context from the book:
{book_context}

Previous conversation:
{conversation_history}

User's message: {user_message}

Respond as {character_name}, incorporating emotional tags and specific citations."""
    },
    
    "preprocessor": {
        "toc_extraction": """Extract the table of contents from this text.

Text:
{text}

Return a structured list of:
- Chapter numbers
- Chapter titles
- Page numbers (if available)
- Brief chapter summaries (if discernible)

Format as JSON.""",
        
        "chunk_analysis": """Analyze this text chunk for key themes and emotional content.

Text:
{chunk}

Identify:
1. Main themes
2. Emotional tone
3. Key quotes
4. Character developments
5. Plot points (mark as spoilers if revealing)"""
    },
    
    "supervisor": {
        "routing": """You are the supervisor agent coordinating the literature companion system.

Current task: {task}
Available agents: {agents}

Route this task to the appropriate agent(s) and specify what each should do.
Consider:
1. Task complexity
2. Required capabilities
3. Optimal execution order
4. Coordination needs"""
    }
}


def get_prompt_template(agent_type: str, template_name: str, **kwargs) -> str:
    """
    Retrieve and format a prompt template.
    
    Args:
        agent_type: Type of agent (persona, preprocessor, supervisor)
        template_name: Name of the specific template
        **kwargs: Variables to format into the template
        
    Returns:
        Formatted prompt string
    """
    template = PROMPT_TEMPLATES.get(agent_type, {}).get(template_name, "")
    return template.format(**kwargs)


# Emotion to voice modulation mapping
EMOTION_VOICE_MAPPING = {
    "joy": {"pitch": 1.1, "speed": 1.1, "stability": 0.6},
    "sadness": {"pitch": 0.9, "speed": 0.9, "stability": 0.4},
    "anger": {"pitch": 1.0, "speed": 1.2, "stability": 0.7},
    "fear": {"pitch": 1.2, "speed": 1.3, "stability": 0.3},
    "surprise": {"pitch": 1.3, "speed": 1.2, "stability": 0.5},
    "contemplative": {"pitch": 0.95, "speed": 0.85, "stability": 0.8},
    "excited": {"pitch": 1.15, "speed": 1.15, "stability": 0.6},
    "melancholic": {"pitch": 0.85, "speed": 0.8, "stability": 0.4},
    "passionate": {"pitch": 1.05, "speed": 1.05, "stability": 0.7},
    "curious": {"pitch": 1.1, "speed": 1.0, "stability": 0.6},
    "skeptical": {"pitch": 0.95, "speed": 0.95, "stability": 0.7},
    "empathetic": {"pitch": 1.0, "speed": 0.9, "stability": 0.5},
    "analytical": {"pitch": 1.0, "speed": 0.95, "stability": 0.8}
}


# Export key items
__all__ = [
    "settings",
    "Settings",
    "PROMPT_TEMPLATES",
    "get_prompt_template",
    "EMOTION_VOICE_MAPPING"
]