# Voice Synthesis Service

This module handles emotionally-aware text-to-speech synthesis for the Wild Genius Professor persona.

## Overview

The voice synthesis service creates dynamic vocal performances that:
- Express the professor's current emotional state through vocal qualities
- Maintain character consistency while allowing emotional range
- Adapt pacing and emphasis based on content
- Create an intimate, engaging listening experience

## Architecture

### Components

1. **Emotion Mapper** (`emotion_mapper.py`)
   - Maps emotional states to voice parameters
   - Calculates transitions between emotional states
   - Manages intensity scaling

2. **TTS Engine** (`tts_engine.py`)
   - Interfaces with voice synthesis providers
   - Handles real-time parameter adjustments
   - Manages audio streaming and buffering

3. **Voice Director** (to be implemented)
   - Analyzes text for emphasis points
   - Adds dramatic pauses and pacing
   - Coordinates with emotion system

## Voice Character Profile

### Wild Genius Professor Voice
- **Age**: Late 50s to early 60s
- **Gender**: Ambiguous/flexible
- **Accent**: Slight Eastern European inflection
- **Base Qualities**: Warm, scholarly, passionate
- **Range**: From whispered contemplation to thunderous fervor

## Emotional Voice Mappings

### [WONDER]
- **Pitch**: Slightly elevated, breathy
- **Speed**: Slower, with pauses
- **Qualities**: Awe, breathlessness, discovery
- **Example**: "Oh... do you see what Dostoevsky has done here?"

### [ANGUISH]
- **Pitch**: Lower, strained
- **Speed**: Variable, broken rhythm
- **Qualities**: Pain, tension, struggle
- **Example**: "The weight of Ivan's argument... it crushes..."

### [ECSTASY]
- **Pitch**: High variation, musical
- **Speed**: Rapid, breathless
- **Qualities**: Joy, overflow, transcendence
- **Example**: "YES! The moment when Alyosha understands!"

### [CONTEMPLATION]
- **Pitch**: Steady, measured
- **Speed**: Deliberate, thoughtful
- **Qualities**: Calm, focused, precise
- **Example**: "Let us consider carefully what this means..."

### [MELANCHOLY]
- **Pitch**: Lower, minor key
- **Speed**: Slow, flowing
- **Qualities**: Wistful, nostalgic, gentle sadness
- **Example**: "There's something so beautifully tragic here..."

## Implementation Plan

### Phase 1: Core Voice System
- [ ] Implement emotion-to-parameter mapping
- [ ] Create base voice profile
- [ ] Build provider abstraction

### Phase 2: Provider Integration
- [ ] ElevenLabs integration (primary)
- [ ] Azure Neural TTS (fallback)
- [ ] Google Cloud TTS (budget option)

### Phase 3: Advanced Features
- [ ] Real-time emotion transitions
- [ ] Dynamic emphasis detection
- [ ] Breathing and pause modeling

## API Design

```python
class VoiceSynthesizer:
    async def synthesize(
        text: str,
        emotion: EmotionalState,
        intensity: float,
        previous_emotion: Optional[EmotionalState] = None
    ) -> AudioStream:
        """Generate speech with emotional expression."""
        
    def get_voice_preview(
        emotion: EmotionalState,
        sample_text: str = None
    ) -> AudioClip:
        """Preview voice in specific emotional state."""
```

## Voice Parameters

```yaml
voice_synthesis:
  provider: "elevenlabs"
  voice_id: "wild_genius_professor"
  model: "eleven_multilingual_v2"
  
  base_parameters:
    stability: 0.75
    similarity_boost: 0.75
    style: 0.5
    use_speaker_boost: true
    
  emotion_modifiers:
    wonder:
      pitch_shift: +5%
      speed: 0.9
      breathiness: 0.3
      
    anguish:
      pitch_shift: -10%
      speed: 0.85
      tension: 0.7
      
    ecstasy:
      pitch_shift: +15%
      speed: 1.2
      variability: 0.8
```

## Performance Considerations

### Streaming Strategy
1. Begin synthesis during response generation
2. Stream audio in chunks for immediate playback
3. Pre-synthesize common phrases
4. Cache emotional transitions

### Quality vs Latency
- High emotion moments: Prioritize quality
- Rapid exchanges: Optimize for speed
- Background pre-generation for predictable responses

## Prosody Markup

Support SSML-like markup for fine control:

```xml
<speak emotion="wonder" intensity="0.8">
  <pause time="500ms"/>
  But consider this...
  <emphasis level="strong">what if</emphasis>
  <pause time="300ms"/>
  everything we believed was wrong?
</speak>
```

## Transition Handling

Smooth emotional transitions are crucial:

1. **Emotion Blending**: Interpolate parameters over 500-1000ms
2. **Natural Breaks**: Transition at sentence boundaries
3. **Intensity Curves**: Use easing functions for natural feel

## Future Enhancements

### Advanced Emotion Model
- Compound emotions (wonder + melancholy)
- Micro-expressions in voice
- Subtext and irony detection

### Interactive Features
- User interruption handling
- Real-time emotion feedback
- Voice mirroring for rapport

### Accessibility
- Adjustable speaking rate
- Clear articulation mode
- Emotion indicators for deaf users