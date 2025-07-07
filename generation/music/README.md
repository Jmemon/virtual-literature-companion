# Music Generation Service

This module creates adaptive, emotionally-responsive musical accompaniment for literary discussions.

## Overview

The music generation service provides:
- Real-time generative music that reflects emotional states
- Smooth transitions between emotional themes
- Non-intrusive ambient soundscapes
- Dynamic intensity based on conversation flow

## Architecture

### Components

1. **Emotion Themes** (`emotion_themes.py`)
   - Defines musical characteristics per emotion
   - Manages theme transitions
   - Controls dynamic parameters

2. **Generator** (`generator.py`)
   - Real-time music synthesis engine
   - Layered composition system
   - Effect processing pipeline

3. **Orchestrator** (to be implemented)
   - Coordinates with conversation flow
   - Manages musical narrative arc
   - Handles silence and space

## Musical Design Philosophy

### Core Principles
1. **Subtlety First**: Music enhances, never dominates
2. **Emotional Truth**: Each emotion has authentic musical expression
3. **Literary Harmony**: Reflects the book's cultural/historical context
4. **Dynamic Response**: Reacts to conversation intensity and pacing

### The Brothers Karamazov Soundworld
- **Instrumentation**: Orthodox choir samples, solo cello, piano, bells
- **Harmonic Language**: Modal scales, Russian orthodox modes
- **Texture**: From sparse single notes to rich polyphony
- **Cultural Elements**: Church bells, folk melodies, winter ambience

## Emotional Musical Mappings

### [WONDER]
- **Key**: Lydian mode (raised 4th creates "floating" quality)
- **Tempo**: 60-70 BPM, rubato
- **Instruments**: Celeste, harp glissandos, high strings
- **Texture**: Sparse, ascending arpeggios, shimmer effects
- **Dynamics**: pp to mp, gentle swells

### [ANGUISH]
- **Key**: Phrygian mode (dark, tense)
- **Tempo**: 50-60 BPM, irregular
- **Instruments**: Low cello, distorted piano, distant choir
- **Texture**: Dissonant clusters, grinding bass, silence breaks
- **Dynamics**: Sudden forte stabs in pianissimo texture

### [ECSTASY]
- **Key**: Major with added 6ths and 9ths
- **Tempo**: 120-140 BPM, driving
- **Instruments**: Full strings, brass swells, bells
- **Texture**: Cascading runs, polyrhythms, dense harmonies
- **Dynamics**: Building from mf to fff

### [CONTEMPLATION]
- **Key**: Natural minor, stable tonic
- **Tempo**: 70-80 BPM, steady
- **Instruments**: Solo piano, subtle strings
- **Texture**: Simple counterpoint, breathing space
- **Dynamics**: Consistent mp, minimal variation

### [MELANCHOLY]
- **Key**: Dorian mode (bittersweet quality)
- **Tempo**: 55-65 BPM, flowing
- **Instruments**: Solo cello, piano, rain sounds
- **Texture**: Descending lines, suspensions, pedal tones
- **Dynamics**: p to mp, gentle dynamics

## Technical Implementation

### Synthesis Approach
```python
class MusicGenerator:
    def __init__(self):
        self.layers = {
            'harmonic': HarmonicLayer(),      # Chord progressions
            'melodic': MelodicLayer(),        # Theme fragments
            'textural': TexturalLayer(),      # Ambient textures
            'rhythmic': RhythmicLayer()       # Pulse and time
        }
        
    async def generate_stream(
        self,
        emotion: EmotionalState,
        intensity: float,
        transition_time: float = 2000
    ) -> AudioStream:
        """Generate continuous music stream."""
```

### Layer System

1. **Harmonic Foundation**
   - Chord progressions based on emotion
   - Voice leading between transitions
   - Pedal points for stability

2. **Melodic Fragments**
   - Leitmotifs for each emotion
   - Procedural variation
   - Call and response patterns

3. **Textural Elements**
   - Environmental sounds (wind, bells)
   - Reverb and space
   - Granular synthesis for transitions

4. **Rhythmic Pulse**
   - Adaptive tempo based on conversation pace
   - Polyrhythms for complexity
   - Silence as rhythmic element

## Transition Strategies

### Smooth Morphing
```
CONTEMPLATION → WONDER:
- Gradually introduce Lydian notes
- Increase reverb and high frequencies
- Slow tempo slightly
- Add celeste/harp colors
```

### Dramatic Shifts
```
SERENITY → ANGUISH:
- Sudden silence (500ms)
- Low cluster attack
- Immediate mode shift
- Textural disruption
```

## Performance Optimization

### Real-time Generation
- Pre-calculate common progressions
- Use efficient synthesis algorithms
- Layer management for CPU efficiency
- Dynamic quality adjustment

### Memory Management
- Stream audio in small buffers
- Garbage collect unused layers
- Cache frequently used samples
- Compress ambient textures

## Configuration

```yaml
music_generation:
  enabled: true
  base_volume: 0.3
  fade_duration: 2000
  
  synthesis:
    sample_rate: 44100
    buffer_size: 512
    latency_mode: "low"
    
  emotions:
    wonder:
      bpm_range: [60, 70]
      key_center: "F"
      mode: "lydian"
      
    anguish:
      bpm_range: [50, 60]
      key_center: "E"
      mode: "phrygian"
```

## Future Enhancements

### Adaptive Composition
- Learn user's musical preferences
- Develop themes across conversations
- Create "memory motifs" for key insights

### Advanced Synthesis
- Neural audio synthesis for unique timbres
- Physical modeling for realistic instruments
- Spatial audio for immersive experience

### Integration Features
- Sync with image generation timing
- Musical punctuation for UI events
- Collaborative composition with user input

## Audio Examples

(In production, this would include audio file links)

- Wonder Theme: Ascending lydian arpeggios with celeste
- Anguish Theme: Dissonant cello clusters with choir
- Transition: Contemplation to Ecstasy buildup
- Full Emotional Journey: 5-minute composition