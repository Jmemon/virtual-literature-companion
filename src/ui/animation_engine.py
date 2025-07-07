"""
Animation engine for the emotive UI.

This module generates dynamic UI states and animations
based on the emotional content of the persona's responses.
"""

from typing import Dict, List, Any, Optional, Tuple
import math
import random
from datetime import datetime
from dataclasses import dataclass, field
import colorsys

from ..models.schemas import (
    EmotionTag,
    UIAnimation,
    EmotiveUIState
)
from ..config import settings


@dataclass
class EmotionColorPalette:
    """Color palette associated with an emotion."""
    primary: str
    secondary: str
    accent: str
    gradient_start: str
    gradient_end: str


@dataclass
class ParticleEffect:
    """Configuration for particle effects."""
    count: int
    speed: float
    size_range: Tuple[float, float]
    color: str
    behavior: str  # "float", "burst", "orbit", "rain"
    lifetime: float


@dataclass
class MorphingShape:
    """Configuration for morphing shapes."""
    base_shape: str  # "circle", "blob", "polygon"
    vertices: int
    roughness: float
    scale: float
    rotation_speed: float
    pulse_frequency: float
    distortion_amount: float


class EmotiveAnimationEngine:
    """
    Engine for generating emotive UI animations.
    
    This engine creates dynamic visual representations
    of emotions through colors, particles, and morphing shapes.
    """
    
    def __init__(self):
        self.emotion_colors = self._init_emotion_colors()
        self.emotion_particles = self._init_emotion_particles()
        self.emotion_shapes = self._init_emotion_shapes()
        
    def generate_ui_state(self, 
                         emotions: List[EmotionTag],
                         emotion_intensities: Dict[str, float],
                         text_length: int = 100) -> EmotiveUIState:
        """
        Generate a complete UI state based on emotions.
        
        Args:
            emotions: List of current emotions
            emotion_intensities: Intensity mapping for emotions
            text_length: Length of response text (affects animation duration)
            
        Returns:
            Complete EmotiveUIState configuration
        """
        if not emotions:
            emotions = [EmotionTag.CURIOUS]
            emotion_intensities = {"curious": 0.5}
            
        # Determine primary emotion
        primary_emotion = self._get_primary_emotion(emotions, emotion_intensities)
        
        # Generate color palette
        color_palette = self._generate_color_palette(emotions, emotion_intensities)
        
        # Generate particle effects
        particle_effects = self._generate_particle_effects(emotions, emotion_intensities)
        
        # Generate morphing shape
        morphing_shape = self._generate_morphing_shape(emotions, emotion_intensities)
        
        # Generate animations
        animations = self._generate_animations(emotions, emotion_intensities, text_length)
        
        # Create emotion blend
        emotion_blend = {
            emotion: emotion_intensities.get(emotion.value, 0.5)
            for emotion in emotions
        }
        
        return EmotiveUIState(
            primary_emotion=primary_emotion,
            emotion_blend=emotion_blend,
            animations=animations,
            color_palette=color_palette,
            particle_effects=particle_effects,
            morphing_shape=morphing_shape
        )
        
    def _init_emotion_colors(self) -> Dict[str, EmotionColorPalette]:
        """Initialize color palettes for each emotion."""
        return {
            "joy": EmotionColorPalette(
                primary="#FFD700",      # Gold
                secondary="#FFA500",    # Orange
                accent="#FF6347",       # Tomato
                gradient_start="#FFE4B5",
                gradient_end="#FF8C00"
            ),
            "sadness": EmotionColorPalette(
                primary="#4682B4",      # Steel Blue
                secondary="#6495ED",    # Cornflower Blue
                accent="#191970",       # Midnight Blue
                gradient_start="#87CEEB",
                gradient_end="#000080"
            ),
            "anger": EmotionColorPalette(
                primary="#DC143C",      # Crimson
                secondary="#FF0000",    # Red
                accent="#8B0000",       # Dark Red
                gradient_start="#FF6B6B",
                gradient_end="#660000"
            ),
            "fear": EmotionColorPalette(
                primary="#8B008B",      # Dark Magenta
                secondary="#9370DB",    # Medium Purple
                accent="#4B0082",       # Indigo
                gradient_start="#DDA0DD",
                gradient_end="#301934"
            ),
            "surprise": EmotionColorPalette(
                primary="#FF1493",      # Deep Pink
                secondary="#FF69B4",    # Hot Pink
                accent="#FFB6C1",       # Light Pink
                gradient_start="#FFC0CB",
                gradient_end="#C71585"
            ),
            "contemplative": EmotionColorPalette(
                primary="#708090",      # Slate Gray
                secondary="#778899",    # Light Slate Gray
                accent="#2F4F4F",       # Dark Slate Gray
                gradient_start="#B0C4DE",
                gradient_end="#36454F"
            ),
            "excited": EmotionColorPalette(
                primary="#00CED1",      # Dark Turquoise
                secondary="#00FFFF",    # Cyan
                accent="#00BFFF",       # Deep Sky Blue
                gradient_start="#AFEEEE",
                gradient_end="#006B6B"
            ),
            "melancholic": EmotionColorPalette(
                primary="#483D8B",      # Dark Slate Blue
                secondary="#6A5ACD",    # Slate Blue
                accent="#7B68EE",       # Medium Slate Blue
                gradient_start="#9999CC",
                gradient_end="#2C2C54"
            ),
            "passionate": EmotionColorPalette(
                primary="#B22222",      # Fire Brick
                secondary="#FF4500",    # Orange Red
                accent="#FF6347",       # Tomato
                gradient_start="#FA8072",
                gradient_end="#8B1A1A"
            ),
            "curious": EmotionColorPalette(
                primary="#32CD32",      # Lime Green
                secondary="#00FF00",    # Lime
                accent="#228B22",       # Forest Green
                gradient_start="#90EE90",
                gradient_end="#006400"
            ),
            "skeptical": EmotionColorPalette(
                primary="#D2691E",      # Chocolate
                secondary="#A0522D",    # Sienna
                accent="#8B4513",       # Saddle Brown
                gradient_start="#DEB887",
                gradient_end="#654321"
            ),
            "empathetic": EmotionColorPalette(
                primary="#DA70D6",      # Orchid
                secondary="#BA55D3",    # Medium Orchid
                accent="#9932CC",       # Dark Orchid
                gradient_start="#E6E6FA",
                gradient_end="#663399"
            ),
            "analytical": EmotionColorPalette(
                primary="#4169E1",      # Royal Blue
                secondary="#0000FF",    # Blue
                accent="#000080",       # Navy
                gradient_start="#6495ED",
                gradient_end="#191970"
            )
        }
        
    def _init_emotion_particles(self) -> Dict[str, ParticleEffect]:
        """Initialize particle effects for each emotion."""
        return {
            "joy": ParticleEffect(
                count=50,
                speed=2.0,
                size_range=(2, 8),
                color="#FFD700",
                behavior="float",
                lifetime=3.0
            ),
            "sadness": ParticleEffect(
                count=30,
                speed=0.5,
                size_range=(3, 6),
                color="#4682B4",
                behavior="rain",
                lifetime=4.0
            ),
            "anger": ParticleEffect(
                count=60,
                speed=3.0,
                size_range=(1, 5),
                color="#FF0000",
                behavior="burst",
                lifetime=1.5
            ),
            "fear": ParticleEffect(
                count=40,
                speed=1.5,
                size_range=(2, 4),
                color="#8B008B",
                behavior="orbit",
                lifetime=2.5
            ),
            "curious": ParticleEffect(
                count=45,
                speed=1.8,
                size_range=(2, 6),
                color="#32CD32",
                behavior="orbit",
                lifetime=3.5
            )
        }
        
    def _init_emotion_shapes(self) -> Dict[str, MorphingShape]:
        """Initialize morphing shapes for each emotion."""
        return {
            "joy": MorphingShape(
                base_shape="blob",
                vertices=8,
                roughness=0.3,
                scale=1.2,
                rotation_speed=0.5,
                pulse_frequency=1.0,
                distortion_amount=0.2
            ),
            "sadness": MorphingShape(
                base_shape="blob",
                vertices=6,
                roughness=0.1,
                scale=0.8,
                rotation_speed=0.1,
                pulse_frequency=0.3,
                distortion_amount=0.4
            ),
            "anger": MorphingShape(
                base_shape="polygon",
                vertices=5,
                roughness=0.8,
                scale=1.3,
                rotation_speed=2.0,
                pulse_frequency=2.0,
                distortion_amount=0.5
            ),
            "contemplative": MorphingShape(
                base_shape="circle",
                vertices=12,
                roughness=0.05,
                scale=1.0,
                rotation_speed=0.2,
                pulse_frequency=0.5,
                distortion_amount=0.1
            ),
            "curious": MorphingShape(
                base_shape="blob",
                vertices=7,
                roughness=0.4,
                scale=1.1,
                rotation_speed=0.8,
                pulse_frequency=1.2,
                distortion_amount=0.3
            )
        }
        
    def _get_primary_emotion(self, emotions: List[EmotionTag],
                           intensities: Dict[str, float]) -> EmotionTag:
        """Determine the primary emotion based on intensities."""
        if not emotions:
            return EmotionTag.CURIOUS
            
        # Sort by intensity
        sorted_emotions = sorted(
            emotions,
            key=lambda e: intensities.get(e.value, 0.5),
            reverse=True
        )
        
        return sorted_emotions[0]
        
    def _generate_color_palette(self, emotions: List[EmotionTag],
                              intensities: Dict[str, float]) -> Dict[str, str]:
        """Generate a blended color palette based on emotions."""
        if not emotions:
            base_palette = self.emotion_colors["curious"]
            return {
                "primary": base_palette.primary,
                "secondary": base_palette.secondary,
                "accent": base_palette.accent,
                "background_gradient_start": base_palette.gradient_start,
                "background_gradient_end": base_palette.gradient_end
            }
            
        # Blend colors based on intensities
        blended_colors = self._blend_colors(emotions, intensities)
        
        return {
            "primary": blended_colors["primary"],
            "secondary": blended_colors["secondary"],
            "accent": blended_colors["accent"],
            "background_gradient_start": blended_colors["gradient_start"],
            "background_gradient_end": blended_colors["gradient_end"],
            "text": self._get_text_color(blended_colors["primary"]),
            "shadow": self._darken_color(blended_colors["primary"], 0.3)
        }
        
    def _blend_colors(self, emotions: List[EmotionTag],
                     intensities: Dict[str, float]) -> Dict[str, str]:
        """Blend multiple emotion colors based on intensities."""
        # Get color components for each emotion
        color_components = {
            "primary": [],
            "secondary": [],
            "accent": [],
            "gradient_start": [],
            "gradient_end": []
        }
        
        total_weight = 0
        
        for emotion in emotions:
            weight = intensities.get(emotion.value, 0.5)
            total_weight += weight
            
            palette = self.emotion_colors.get(emotion.value)
            if palette:
                for component in color_components:
                    color = getattr(palette, component)
                    rgb = self._hex_to_rgb(color)
                    color_components[component].append((rgb, weight))
                    
        # Blend colors
        blended = {}
        for component, colors_weights in color_components.items():
            if colors_weights:
                blended_rgb = self._weighted_average_color(colors_weights, total_weight)
                blended[component] = self._rgb_to_hex(blended_rgb)
            else:
                # Fallback
                blended[component] = "#808080"
                
        return blended
        
    def _weighted_average_color(self, colors_weights: List[Tuple[Tuple[int, int, int], float]],
                               total_weight: float) -> Tuple[int, int, int]:
        """Calculate weighted average of colors."""
        r_total = g_total = b_total = 0
        
        for (r, g, b), weight in colors_weights:
            r_total += r * weight
            g_total += g * weight
            b_total += b * weight
            
        if total_weight > 0:
            return (
                int(r_total / total_weight),
                int(g_total / total_weight),
                int(b_total / total_weight)
            )
        return (128, 128, 128)  # Gray fallback
        
    def _generate_particle_effects(self, emotions: List[EmotionTag],
                                 intensities: Dict[str, float]) -> Dict[str, Any]:
        """Generate particle effect configuration."""
        if not emotions:
            return {}
            
        # Get primary emotion particle effect
        primary_emotion = self._get_primary_emotion(emotions, intensities)
        base_effect = self.emotion_particles.get(
            primary_emotion.value,
            self.emotion_particles["curious"]
        )
        
        # Modify based on intensity
        primary_intensity = intensities.get(primary_emotion.value, 0.5)
        
        return {
            "enabled": True,
            "count": int(base_effect.count * (0.5 + primary_intensity)),
            "speed": base_effect.speed * (0.5 + primary_intensity),
            "sizeRange": list(base_effect.size_range),
            "color": base_effect.color,
            "behavior": base_effect.behavior,
            "lifetime": base_effect.lifetime,
            "opacity": 0.3 + (primary_intensity * 0.4)
        }
        
    def _generate_morphing_shape(self, emotions: List[EmotionTag],
                               intensities: Dict[str, float]) -> Dict[str, Any]:
        """Generate morphing shape configuration."""
        if not emotions:
            emotions = [EmotionTag.CURIOUS]
            
        # Blend shape properties from multiple emotions
        blended_shape = self._blend_shapes(emotions, intensities)
        
        return {
            "type": blended_shape["base_shape"],
            "vertices": blended_shape["vertices"],
            "roughness": blended_shape["roughness"],
            "scale": blended_shape["scale"],
            "rotationSpeed": blended_shape["rotation_speed"],
            "pulseFrequency": blended_shape["pulse_frequency"],
            "distortionAmount": blended_shape["distortion_amount"],
            "morphSpeed": 0.5,
            "complexity": self._calculate_shape_complexity(emotions)
        }
        
    def _blend_shapes(self, emotions: List[EmotionTag],
                     intensities: Dict[str, float]) -> Dict[str, Any]:
        """Blend shape properties from multiple emotions."""
        # Start with primary emotion shape
        primary_emotion = self._get_primary_emotion(emotions, intensities)
        base_shape = self.emotion_shapes.get(
            primary_emotion.value,
            self.emotion_shapes["curious"]
        )
        
        # If only one emotion, return its shape
        if len(emotions) == 1:
            return {
                "base_shape": base_shape.base_shape,
                "vertices": base_shape.vertices,
                "roughness": base_shape.roughness,
                "scale": base_shape.scale,
                "rotation_speed": base_shape.rotation_speed,
                "pulse_frequency": base_shape.pulse_frequency,
                "distortion_amount": base_shape.distortion_amount
            }
            
        # Blend numeric properties
        total_weight = sum(intensities.get(e.value, 0.5) for e in emotions)
        
        blended = {
            "base_shape": base_shape.base_shape,  # Keep primary shape type
            "vertices": 0,
            "roughness": 0,
            "scale": 0,
            "rotation_speed": 0,
            "pulse_frequency": 0,
            "distortion_amount": 0
        }
        
        for emotion in emotions:
            weight = intensities.get(emotion.value, 0.5) / total_weight
            shape = self.emotion_shapes.get(emotion.value)
            
            if shape:
                blended["vertices"] += int(shape.vertices * weight)
                blended["roughness"] += shape.roughness * weight
                blended["scale"] += shape.scale * weight
                blended["rotation_speed"] += shape.rotation_speed * weight
                blended["pulse_frequency"] += shape.pulse_frequency * weight
                blended["distortion_amount"] += shape.distortion_amount * weight
                
        # Ensure vertices is at least 3
        blended["vertices"] = max(3, blended["vertices"])
        
        return blended
        
    def _generate_animations(self, emotions: List[EmotionTag],
                           intensities: Dict[str, float],
                           text_length: int) -> List[UIAnimation]:
        """Generate UI animations based on emotions."""
        animations = []
        
        # Base duration scaled by text length
        base_duration = min(text_length * 0.05, 10.0)  # Max 10 seconds
        
        # Primary emotion animation
        primary_emotion = self._get_primary_emotion(emotions, intensities)
        primary_intensity = intensities.get(primary_emotion.value, 0.5)
        
        # Shape morph animation
        animations.append(UIAnimation(
            element_id="emotion-shape",
            animation_type="morph",
            duration=base_duration,
            parameters={
                "easing": "easeInOutSine",
                "loop": True,
                "intensity": primary_intensity
            }
        ))
        
        # Color transition animation
        animations.append(UIAnimation(
            element_id="background",
            animation_type="gradient-shift",
            duration=base_duration * 1.5,
            parameters={
                "easing": "linear",
                "loop": True
            }
        ))
        
        # Particle animation
        animations.append(UIAnimation(
            element_id="particles",
            animation_type="particle-flow",
            duration=base_duration * 2,
            parameters={
                "continuous": True
            }
        ))
        
        # Add emotion-specific animations
        if EmotionTag.JOY in emotions:
            animations.append(UIAnimation(
                element_id="emotion-shape",
                animation_type="bounce",
                duration=1.0,
                parameters={"repeat": 3, "height": 20}
            ))
            
        if EmotionTag.SADNESS in emotions:
            animations.append(UIAnimation(
                element_id="emotion-shape",
                animation_type="drip",
                duration=2.0,
                parameters={"drops": 5}
            ))
            
        if EmotionTag.EXCITED in emotions:
            animations.append(UIAnimation(
                element_id="emotion-shape",
                animation_type="vibrate",
                duration=0.5,
                parameters={"intensity": primary_intensity * 10}
            ))
            
        return animations
        
    def _calculate_shape_complexity(self, emotions: List[EmotionTag]) -> float:
        """Calculate shape complexity based on emotion mix."""
        # More emotions = more complex shape
        base_complexity = len(emotions) / len(EmotionTag)
        
        # Certain emotions add complexity
        complex_emotions = [EmotionTag.CONTEMPLATIVE, EmotionTag.ANALYTICAL, EmotionTag.MELANCHOLIC]
        complexity_bonus = sum(0.1 for e in emotions if e in complex_emotions)
        
        return min(base_complexity + complexity_bonus, 1.0)
        
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return (rgb[0], rgb[1], rgb[2])
        
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
    def _get_text_color(self, background_color: str) -> str:
        """Get appropriate text color for given background."""
        r, g, b = self._hex_to_rgb(background_color)
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        # Return white for dark backgrounds, black for light
        return "#FFFFFF" if luminance < 0.5 else "#000000"
        
    def _darken_color(self, color: str, factor: float) -> str:
        """Darken a color by given factor."""
        r, g, b = self._hex_to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        l = max(0, l - factor)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return self._rgb_to_hex((int(r*255), int(g*255), int(b*255)))