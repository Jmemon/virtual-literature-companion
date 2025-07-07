"""
Text-to-Speech engine with emotional voice modulation.

This module handles:
- Voice synthesis using ElevenLabs API
- Emotional voice modulation based on persona emotions
- Audio caching and optimization
- Real-time streaming support
"""

from typing import Dict, List, Optional, Any, Tuple
import asyncio
import hashlib
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
from elevenlabs import AsyncElevenLabs, Voice, VoiceSettings
from elevenlabs.types import Model
import aiofiles
import numpy as np
from pydub import AudioSegment
import io

from ..config import settings, EMOTION_VOICE_MAPPING
from ..models.schemas import EmotionTag, VoiceSettings as VoiceSettingsSchema


class EmotionalTTSEngine:
    """
    Text-to-Speech engine with emotional awareness.
    
    This engine:
    1. Synthesizes speech using ElevenLabs API
    2. Modulates voice parameters based on emotions
    3. Caches generated audio for efficiency
    4. Provides streaming support for real-time playback
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.elevenlabs_api_key
        if not self.api_key:
            logger.warning("ElevenLabs API key not provided. Voice features disabled.")
            self.client = None
        else:
            self.client = AsyncElevenLabs(api_key=self.api_key)
            
        self.cache_dir = settings.voice_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default voice settings
        self.default_voice_id = settings.voice_id
        self.default_settings = VoiceSettingsSchema(
            voice_id=self.default_voice_id,
            stability=settings.voice_stability,
            similarity_boost=settings.voice_similarity_boost,
            style=settings.voice_style
        )
        
    async def synthesize_emotional_speech(self, 
                                        text: str,
                                        emotions: List[EmotionTag],
                                        emotion_intensities: Dict[str, float],
                                        voice_modulation: Optional[Dict[str, float]] = None,
                                        use_cache: bool = True) -> Tuple[bytes, str]:
        """
        Synthesize speech with emotional modulation.
        
        Args:
            text: Text to synthesize
            emotions: List of emotions
            emotion_intensities: Emotion intensity mapping
            voice_modulation: Optional pre-calculated voice parameters
            use_cache: Whether to use cached audio
            
        Returns:
            Tuple of (audio_bytes, cache_key)
        """
        if not self.client:
            logger.error("TTS client not initialized")
            return b"", ""
            
        # Generate cache key
        cache_key = self._generate_cache_key(text, emotions, emotion_intensities)
        
        # Check cache
        if use_cache:
            cached_audio = await self._get_cached_audio(cache_key)
            if cached_audio:
                logger.info(f"Using cached audio for key: {cache_key}")
                return cached_audio, cache_key
                
        # Calculate voice modulation if not provided
        if not voice_modulation:
            voice_modulation = self._calculate_voice_modulation(emotions, emotion_intensities)
            
        # Apply emotional modulation to voice settings
        voice_settings = self._apply_emotional_modulation(voice_modulation)
        
        try:
            # Generate audio with ElevenLabs
            logger.info(f"Generating speech with emotions: {[e.value for e in emotions]}")
            
            audio_bytes = await self._generate_audio(text, voice_settings)
            
            # Apply post-processing for emotional effects
            processed_audio = await self._apply_emotional_effects(
                audio_bytes,
                emotions,
                emotion_intensities
            )
            
            # Cache the result
            if use_cache:
                await self._cache_audio(cache_key, processed_audio)
                
            return processed_audio, cache_key
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return b"", ""
            
    async def _generate_audio(self, text: str, 
                            voice_settings: VoiceSettingsSchema) -> bytes:
        """
        Generate audio using ElevenLabs API.
        
        Args:
            text: Text to synthesize
            voice_settings: Voice configuration
            
        Returns:
            Audio bytes
        """
        # Create voice settings for API
        settings_obj = VoiceSettings(
            stability=voice_settings.stability,
            similarity_boost=voice_settings.similarity_boost,
            style=voice_settings.style,
            use_speaker_boost=True
        )
        
        # Generate audio
        if not self.client:
            raise ValueError("TTS client not initialized")
            
        audio_generator = await self.client.generate(
            text=text,
            voice=Voice(
                voice_id=voice_settings.voice_id,
                settings=settings_obj
            ),
            model="eleven_monolingual_v1"
        )
        
        # Collect audio chunks
        audio_chunks = []
        async for chunk in audio_generator:
            audio_chunks.append(chunk)
            
        return b"".join(audio_chunks)
        
    def _calculate_voice_modulation(self, emotions: List[EmotionTag],
                                  emotion_intensities: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate voice modulation parameters from emotions.
        
        Args:
            emotions: List of emotions
            emotion_intensities: Emotion intensities
            
        Returns:
            Voice modulation parameters
        """
        # Start with neutral baseline
        modulation = {
            "pitch": 1.0,
            "speed": 1.0,
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5
        }
        
        if not emotions:
            return modulation
            
        # Calculate weighted average of emotion settings
        total_weight = 0
        weighted_params = {key: 0.0 for key in modulation.keys()}
        
        for emotion in emotions:
            weight = emotion_intensities.get(emotion.value, 0.5)
            total_weight += weight
            
            emotion_mapping = EMOTION_VOICE_MAPPING.get(emotion.value, {})
            for param, value in emotion_mapping.items():
                if param in weighted_params:
                    weighted_params[param] += value * weight
                    
        # Normalize by total weight
        if total_weight > 0:
            for param in weighted_params:
                # Blend with baseline
                blend_factor = 0.7  # How much emotion affects the voice
                modulation[param] = (
                    modulation[param] * (1 - blend_factor) + 
                    (weighted_params[param] / total_weight) * blend_factor
                )
                
        return modulation
        
    def _apply_emotional_modulation(self, 
                                  voice_modulation: Dict[str, float]) -> VoiceSettingsSchema:
        """
        Apply emotional modulation to voice settings.
        
        Args:
            voice_modulation: Modulation parameters
            
        Returns:
            Modified voice settings
        """
        # Start with default settings
        settings = VoiceSettingsSchema(
            voice_id=self.default_voice_id,
            stability=self.default_settings.stability,
            similarity_boost=self.default_settings.similarity_boost,
            style=self.default_settings.style
        )
        
        # Apply modulation
        if "stability" in voice_modulation:
            settings.stability = max(0, min(1, voice_modulation["stability"]))
            
        if "similarity_boost" in voice_modulation:
            settings.similarity_boost = max(0, min(1, voice_modulation["similarity_boost"]))
            
        if "style" in voice_modulation:
            settings.style = max(0, min(1, voice_modulation["style"]))
            
        # Note: Pitch and speed adjustments would be applied in post-processing
        # as ElevenLabs doesn't directly support these parameters
        
        return settings
        
    async def _apply_emotional_effects(self, audio_bytes: bytes,
                                     emotions: List[EmotionTag],
                                     emotion_intensities: Dict[str, float]) -> bytes:
        """
        Apply post-processing effects based on emotions.
        
        Args:
            audio_bytes: Raw audio data
            emotions: List of emotions
            emotion_intensities: Emotion intensities
            
        Returns:
            Processed audio bytes
        """
        try:
            # Convert bytes to AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            
            # Calculate overall emotional parameters
            pitch_shift = 0
            speed_change = 1.0
            
            for emotion in emotions:
                intensity = emotion_intensities.get(emotion.value, 0.5)
                emotion_mapping = EMOTION_VOICE_MAPPING.get(emotion.value, {})
                
                if "pitch" in emotion_mapping:
                    pitch_shift += (emotion_mapping["pitch"] - 1.0) * intensity
                    
                if "speed" in emotion_mapping:
                    speed_change *= emotion_mapping["speed"] ** intensity
                    
            # Apply pitch shifting (in semitones)
            if abs(pitch_shift) > 0.01:
                # Convert pitch multiplier to semitones
                semitones = 12 * np.log2(1 + pitch_shift)
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * (2.0 ** (semitones / 12.0)))
                }).set_frame_rate(audio.frame_rate)
                
            # Apply speed change
            if abs(speed_change - 1.0) > 0.01:
                # Speed up or slow down
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed_change)
                }).set_frame_rate(audio.frame_rate)
                
            # Add subtle effects based on emotions
            if EmotionTag.MELANCHOLIC in emotions:
                # Add slight reverb effect for melancholic mood
                audio = self._add_reverb(audio, 0.1)
                
            if EmotionTag.EXCITED in emotions:
                # Add slight compression for excited speech
                audio = self._add_compression(audio, 0.8)
                
            # Export back to bytes
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3")
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"Error applying emotional effects: {e}")
            return audio_bytes  # Return original if processing fails
            
    def _add_reverb(self, audio: AudioSegment, amount: float) -> AudioSegment:
        """Add simple reverb effect."""
        # Simplified reverb - mix with delayed version
        delay_ms = 50
        delayed = AudioSegment.silent(duration=delay_ms) + audio
        return audio.overlay(delayed - 10)  # Mix at -10dB
        
    def _add_compression(self, audio: AudioSegment, ratio: float) -> AudioSegment:
        """Add dynamic compression."""
        # Simplified compression - normalize dynamics
        return audio.compress_dynamic_range()
        
    def _generate_cache_key(self, text: str, emotions: List[EmotionTag],
                          emotion_intensities: Dict[str, float]) -> str:
        """Generate unique cache key for audio."""
        # Create deterministic string representation
        emotion_str = "-".join(sorted([e.value for e in emotions]))
        intensity_str = "-".join([
            f"{k}:{v:.2f}" for k, v in sorted(emotion_intensities.items())
        ])
        
        cache_string = f"{text}|{emotion_str}|{intensity_str}|{self.default_voice_id}"
        return hashlib.md5(cache_string.encode()).hexdigest()
        
    async def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Retrieve cached audio if available."""
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        
        if cache_path.exists():
            try:
                async with aiofiles.open(cache_path, "rb") as f:
                    return await f.read()
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
                
        return None
        
    async def _cache_audio(self, cache_key: str, audio_bytes: bytes):
        """Cache audio for future use."""
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        
        try:
            async with aiofiles.open(cache_path, "wb") as f:
                await f.write(audio_bytes)
                
            # Also save metadata
            metadata_path = self.cache_dir / f"{cache_key}.json"
            metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "size_bytes": len(audio_bytes)
            }
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(json.dumps(metadata))
                
        except Exception as e:
            logger.warning(f"Error caching audio: {e}")
            
    async def clear_cache(self, older_than_days: int = 7):
        """Clear old cached audio files."""
        cutoff_time = datetime.utcnow().timestamp() - (older_than_days * 24 * 60 * 60)
        
        for cache_file in self.cache_dir.glob("*.mp3"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    # Also remove metadata
                    metadata_file = cache_file.with_suffix(".json")
                    if metadata_file.exists():
                        metadata_file.unlink()
            except Exception as e:
                logger.warning(f"Error clearing cache file {cache_file}: {e}")


# Convenience function
async def synthesize_emotional_speech(text: str, emotions: List[EmotionTag],
                                    emotion_intensities: Dict[str, float]) -> Tuple[bytes, str]:
    """
    Synthesize emotional speech using the default engine.
    
    Args:
        text: Text to synthesize
        emotions: List of emotions
        emotion_intensities: Emotion intensities
        
    Returns:
        Tuple of (audio_bytes, cache_key)
    """
    engine = EmotionalTTSEngine()
    return await engine.synthesize_emotional_speech(text, emotions, emotion_intensities)