# Virtual Literature Companion 📚🎭

An emotionally-aware AI companion for deep literary discussions. This system combines advanced language models with emotional intelligence to create a living, breathing companion that engages with literature on both intellectual and emotional levels.

## 🌟 Vision

Imagine a literature companion that doesn't just analyze text, but *feels* it alongside you. Our Virtual Literature Companion:

- **Expresses genuine emotions** through dynamic UI morphing and voice modulation
- **Remembers your reading journey** using persistent memory (Honcho)
- **Cites specific passages** to ground discussions in the text
- **Never spoils** upcoming plot points
- **Adapts its personality** based on the emotional tone of your conversations

## 🏗️ Architecture

### Multi-Agent System (LangGraph)

The system uses a supervisor pattern with specialized agents:

```
┌─────────────────┐
│   Supervisor    │ ← Orchestrates all agents
└────────┬────────┘
         │
    ┌────┴────┬──────────┬─────────┐
    ▼         ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│  PDF   │ │Persona │ │ Voice  │ │   UI   │
│ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │
└────────┘ └────────┘ └────────┘ └────────┘
```

1. **PDF Preprocessing Agent**: Extracts table of contents, chunks text, analyzes themes
2. **Persona Agent**: Engages in emotional discussions with memory (powered by Honcho)
3. **Voice Synthesis Agent**: Renders responses with emotional voice modulation (ElevenLabs)
4. **UI Animation Agent**: Generates dynamic visual states based on emotions

### Emotional System

Emotions are expressed through multiple channels:

- **Text**: Emotion tags embedded in responses `[emotion: joy]`
- **Voice**: Pitch, speed, and timbre modulation based on emotional state
- **Visual**: Morphing shapes, particle effects, and color transitions

### Supported Emotions

- Core: joy, sadness, anger, fear, surprise
- Literary: contemplative, melancholic, passionate, curious
- Analytical: skeptical, empathetic, analytical

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PDF books for processing
- API Keys:
  - OpenAI or Anthropic (for LLMs)
  - ElevenLabs (for voice synthesis)
  - Honcho (for memory - optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/virtual-literature-companion.git
cd virtual-literature-companion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

1. Start the API server:
```bash
python api/main.py
```

2. The API will be available at `http://localhost:8000`
3. Interactive API docs at `http://localhost:8000/docs`

## 📖 Usage

### 1. Upload a Book

```bash
curl -X POST "http://localhost:8000/api/books/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/book.pdf" \
  -F "user_id=your_user_id"
```

### 2. Check Processing Status

```bash
curl -X GET "http://localhost:8000/api/books/processing/{job_id}"
```

### 3. Start a Conversation

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "book_id": "book_id_from_upload",
    "message": "What did you think about the opening chapter?",
    "include_voice": true
  }'
```

### 4. WebSocket Connection (Real-time)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/your_user_id');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'chat_response') {
    // Handle text response with emotions
  } else if (data.type === 'emotion_update') {
    // Update UI with animation state
  } else if (data.type === 'voice_ready') {
    // Play emotional voice
  }
};

// Send a message
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    user_id: 'your_user_id',
    book_id: 'book_id',
    message: 'Tell me about the protagonist'
  }
}));
```

## 🎨 Emotive UI System

The UI dynamically morphs based on the companion's emotional state:

### Color Palettes
- **Joy**: Golden yellows and warm oranges
- **Sadness**: Deep blues and steel grays
- **Contemplative**: Muted grays and soft blues
- **Passionate**: Fiery reds and vibrant oranges

### Animations
- **Shape Morphing**: Blob-like forms that pulse and distort
- **Particle Effects**: Floating, bursting, or raining particles
- **Gradient Shifts**: Smooth color transitions in the background

### Example UI State
```json
{
  "primary_emotion": "contemplative",
  "emotion_blend": {
    "contemplative": 0.7,
    "curious": 0.3
  },
  "color_palette": {
    "primary": "#708090",
    "background_gradient_start": "#B0C4DE",
    "background_gradient_end": "#36454F"
  },
  "morphing_shape": {
    "type": "circle",
    "vertices": 12,
    "roughness": 0.05,
    "pulse_frequency": 0.5
  }
}
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

## 📁 Project Structure

```
virtual-literature-companion/
├── api/                    # FastAPI application
│   └── main.py            # Main API endpoints
├── src/
│   ├── agents/            # LangGraph agents
│   │   ├── pdf_agent.py   # PDF preprocessing
│   │   └── persona_agent.py # Emotional persona
│   ├── config.py          # Configuration management
│   ├── graph/             # LangGraph workflow
│   │   └── workflow.py    # Supervisor orchestration
│   ├── memory/            # Honcho integration
│   │   └── honcho_client.py
│   ├── models/            # Data models
│   │   ├── schemas.py     # Pydantic schemas
│   │   └── state.py       # Agent state
│   ├── processors/        # Text processing
│   │   └── pdf_extractor.py
│   ├── ui/                # UI generation
│   │   └── animation_engine.py
│   └── voice/             # Voice synthesis
│       └── tts_engine.py
├── frontend/              # React frontend (optional)
├── tests/                 # Test suite
└── requirements.txt       # Dependencies
```

## 🔧 Configuration

Key configuration options in `.env`:

```env
# LLM Configuration
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

# Voice Configuration
VOICE_ID=21m00Tcm4TlvDq8ikWAM  # ElevenLabs voice ID
VOICE_STABILITY=0.5
VOICE_SIMILARITY_BOOST=0.75

# Memory Configuration
MEMORY_WINDOW_SIZE=10
HONCHO_APP_ID=literature-companion

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run code formatting:
```bash
black src/ tests/
ruff check src/ tests/
```

3. Run type checking:
```bash
mypy src/
```

## 📈 Future Enhancements

- **Multi-modal Understanding**: Analyze book illustrations and maps
- **Reading Groups**: Shared discussions with multiple users
- **Voice Cloning**: Custom companion voices
- **AR/VR Integration**: Immersive literary experiences
- **Dynamic UI Experiments**: 
  - Rorschach-like morphing blobs
  - Generative art based on emotional state
  - 3D particle systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain/LangGraph for the agent framework
- Honcho for conversation memory
- ElevenLabs for voice synthesis
- The open-source community for inspiration

---

*"A book is a dream you hold in your hands." - Neil Gaiman*

Let's make those dreams come alive. 📚✨