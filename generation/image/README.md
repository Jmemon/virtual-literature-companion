# Image Generation Service

This module handles AI-generated imagery that reflects the themes, emotions, and content of literary discussions.

## Overview

The image generation service creates visually cohesive artwork that:
- Maintains a consistent artistic style throughout a book discussion
- Reflects the current emotional state of the conversation
- Illustrates themes and concepts being explored
- Enhances the immersive experience without distracting from the text

## Architecture

### Components

1. **Style Manager** (`style_manager.py`)
   - Maintains visual consistency across all generated images
   - Adapts base style to current emotional state
   - Manages style parameters per book

2. **Prompt Builder** (`prompt_builder.py`)
   - Constructs detailed prompts from:
     - Current discussion theme
     - Emotional context
     - Book-specific visual style
     - Previous image history

3. **Generation Engine** (to be implemented)
   - Interfaces with image generation APIs (DALL-E 3, Stable Diffusion, Midjourney)
   - Handles rate limiting and error recovery
   - Manages image caching and storage

## Visual Styles

### The Brothers Karamazov
- **Base Style**: Russian literary realism with expressionist touches
- **Color Palette**: Deep reds, orthodox golds, winter whites, charcoal blacks
- **Influences**: Ilya Repin, Orthodox iconography, Russian winter landscapes
- **Mood**: Psychological intensity, spiritual questioning, human suffering

### Emotional Modifiers

Each emotion applies specific visual modifications:

- **[WONDER]**: Ethereal lighting, soft focus, upward composition
- **[ANGUISH]**: Sharp contrasts, fractured elements, dark shadows
- **[ECSTASY]**: Vibrant colors, swirling movement, light overflow
- **[CONTEMPLATION]**: Muted tones, centered composition, subtle details
- **[MELANCHOLY]**: Blue-grey palette, rain/mist, solitary figures

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Implement style manager with emotion mappings
- [ ] Build prompt construction pipeline
- [ ] Create provider abstraction layer

### Phase 2: Provider Integration
- [ ] DALL-E 3 integration (primary)
- [ ] Stable Diffusion fallback
- [ ] Midjourney support (optional)

### Phase 3: Optimization
- [ ] Implement intelligent caching
- [ ] Add prompt refinement based on results
- [ ] Build style evolution tracking

## API Design

```python
class ImageGenerator:
    async def generate(
        theme: str,
        emotion: EmotionalState,
        intensity: float,
        context: ConversationContext
    ) -> ImageResult:
        """Generate an image based on current discussion."""
        
    async def get_style_preview(
        book: str,
        emotion: EmotionalState
    ) -> StylePreview:
        """Preview the visual style for a book/emotion combo."""
```

## Configuration

```yaml
image_generation:
  provider: "dalle3"
  model: "dall-e-3"
  size: "1024x1024"
  quality: "hd"
  style_consistency: 0.8
  
  providers:
    dalle3:
      api_key: ${OPENAI_API_KEY}
      rate_limit: 50/min
      
    stable_diffusion:
      endpoint: "https://api.stability.ai"
      api_key: ${STABILITY_API_KEY}
      model: "stable-diffusion-xl-1024-v1-0"
```

## Style Evolution

The visual style should subtly evolve as the user progresses through the book:

1. **Early chapters**: More literal, grounded imagery
2. **Middle chapters**: Increasing psychological abstraction
3. **Final chapters**: Full expressionist/symbolic treatment

## Performance Considerations

- Generate images asynchronously during conversation
- Pre-generate common theme/emotion combinations
- Cache generated images with semantic keys
- Compress and optimize for web delivery

## Future Enhancements

1. **User Preference Learning**
   - Track which images resonate with users
   - Adapt style based on engagement

2. **Multi-Modal Integration**
   - Sync image transitions with music
   - Coordinate with UI color changes

3. **Interactive Elements**
   - Allow users to save favorite images
   - Create visual journey gallery