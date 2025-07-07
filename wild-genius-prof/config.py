"""
Configuration for the Wild Genius Professor agent.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.8
    max_tokens: int = 2000
    streaming: bool = True
    

@dataclass 
class EmotionConfig:
    """Emotion system configuration."""
    # Emotion intensity thresholds
    high_intensity_threshold: float = 0.7
    low_intensity_threshold: float = 0.3
    
    # UI transition timings (ms)
    transition_duration: int = 2000
    rapid_transition_duration: int = 500
    
    # Image generation triggers
    intensity_trigger: float = 0.7
    message_interval_trigger: int = 5
    

@dataclass
class HonchoConfig:
    """Honcho integration configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("HONCHO_API_KEY"))
    app_name: str = "wild-genius-professor"
    enable_insights: bool = True
    insight_query_timeout: int = 5000  # ms
    

@dataclass
class BookConfig:
    """Book-specific configuration."""
    current_book: str = "brothers_karamazov"
    total_books: int = 12  # Brothers K has 12 books
    
    # Visual style per book (can be extended for other books)
    visual_styles: Dict[str, str] = field(default_factory=lambda: {
        "brothers_karamazov": "Russian literary realism with expressionist touches, "
                             "dark orthodox iconography, snow-covered landscapes, "
                             "psychological portraits in the style of Ilya Repin"
    })
    
    # Key themes per book
    major_themes: Dict[str, list] = field(default_factory=lambda: {
        "brothers_karamazov": [
            "faith_and_doubt", "patricide", "free_will", "suffering",
            "redemption", "family", "morality", "madness", "love", "justice"
        ]
    })
    

@dataclass
class UIConfig:
    """UI configuration."""
    # WebSocket settings
    ws_heartbeat_interval: int = 30000  # ms
    ws_reconnect_delay: int = 3000  # ms
    
    # Animation settings
    shape_complexity: int = 3  # 1-5, affects polygon count
    particle_count: int = 100
    enable_3d: bool = True
    
    # Theme transition settings  
    color_transition_easing: str = "ease-in-out"
    shape_morph_duration: int = 3000  # ms
    

@dataclass
class GenerationConfig:
    """Multimedia generation configuration."""
    # Image generation
    image_provider: str = "dalle3"  # "dalle3", "stability", "midjourney"
    image_size: str = "1024x1024"
    image_quality: str = "hd"
    
    # Voice synthesis
    voice_provider: str = "elevenlabs"  # "elevenlabs", "azure", "google"
    voice_id: str = "wild_genius_professor"
    voice_stability: float = 0.75
    voice_clarity: float = 0.75
    
    # Music generation
    music_enabled: bool = True
    music_volume: float = 0.3  # 0-1
    music_fade_duration: int = 2000  # ms
    

@dataclass
class Config:
    """Main configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    honcho: HonchoConfig = field(default_factory=HonchoConfig)
    book: BookConfig = field(default_factory=BookConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Global settings
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # Override with env vars if present
        llm_provider = os.getenv("LLM_PROVIDER")
        if llm_provider:
            config.llm.provider = llm_provider
        llm_model = os.getenv("LLM_MODEL")
        if llm_model:
            config.llm.model = llm_model
        llm_temp = os.getenv("LLM_TEMPERATURE")
        if llm_temp:
            config.llm.temperature = float(llm_temp)
            
        image_provider = os.getenv("IMAGE_PROVIDER")
        if image_provider:
            config.generation.image_provider = image_provider
        voice_provider = os.getenv("VOICE_PROVIDER")
        if voice_provider:
            config.generation.voice_provider = voice_provider
            
        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "streaming": self.llm.streaming
            },
            "emotion": {
                "high_intensity_threshold": self.emotion.high_intensity_threshold,
                "low_intensity_threshold": self.emotion.low_intensity_threshold,
                "transition_duration": self.emotion.transition_duration
            },
            "honcho": {
                "app_name": self.honcho.app_name,
                "enable_insights": self.honcho.enable_insights
            },
            "book": {
                "current_book": self.book.current_book,
                "visual_style": self.book.visual_styles.get(self.book.current_book, "")
            },
            "ui": {
                "shape_complexity": self.ui.shape_complexity,
                "particle_count": self.ui.particle_count,
                "enable_3d": self.ui.enable_3d
            },
            "generation": {
                "image_provider": self.generation.image_provider,
                "voice_provider": self.generation.voice_provider,
                "music_enabled": self.generation.music_enabled
            },
            "debug": self.debug,
            "log_level": self.log_level
        }


# Global config instance
config = Config.from_env()