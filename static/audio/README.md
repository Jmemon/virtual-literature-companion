# Audio Files for Gesture System

This directory should contain the audio files for each gesture type. The audio files should be in MP3 format and named according to the gesture they represent.

## Required Audio Files

Place the following audio files in this directory:

- `lean-in.mp3` - Rising tension chord
- `pull-back.mp3` - Descending, spacious tone
- `tremble.mp3` - Tremolo effect
- `illuminate.mp3` - Crystalline chimes
- `fragment.mp3` - Glitchy, breaking sound
- `whisper.mp3` - Pink noise filter effect
- `grip.mp3` - Tightening, contracting sound
- `shatter.mp3` - Glass breaking with reverb
- `breathe.mp3` - Low sine wave matching breath rhythm
- `reach.mp3` - Connecting, bridging sound
- `dance.mp3` - Joyful, spiraling melody

## Audio Creation Guidelines

Each audio file should:
- Be 1-4 seconds in duration
- Fade in and out smoothly
- Match the emotional quality of the gesture
- Be normalized to consistent volume levels
- Be compressed to reasonable file sizes (< 100KB per file)

## Alternative: Web Audio API

If audio files are not available, the application can be modified to generate sounds using the Web Audio API. See `static/js/audio-generator.js` for an example implementation.