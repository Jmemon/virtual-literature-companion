# Virtual Literature Companion 📚✨

An AI-driven literature companion that creates emotionally expressive, intellectually engaging discussions about classic literature through a unique multimedia experience. The AI persona doesn't just discuss books - it lives them, expressing emotions through dynamic UI, generated imagery, voice modulation, and adaptive music.

![Virtual Literature Companion Demo](docs/images/demo.gif)

## 🌟 Features

### Core Experience
- **Emotionally Aware AI Persona**: The "Wild Genius Professor" - an eccentric, passionate literature scholar who experiences texts viscerally
- **Socratic Teaching Method**: Guides discovery through thoughtful questions rather than providing answers
- **Spoiler Protection**: Never reveals content beyond your current reading position
- **Persistent Memory**: Remembers your conversations and builds a psychological profile using Honcho

### Multimedia Expression
- **Dynamic UI**: Interface morphs and changes color based on emotional state
- **AI-Generated Imagery**: Creates thematically appropriate artwork reflecting discussion topics
- **Emotive Voice Synthesis**: Professor's voice changes with emotional intensity
- **Adaptive Music**: Real-time generative soundscapes that reflect the conversation's mood

### Emotional States
The professor expresses ten distinct emotional states, each with unique multimedia characteristics:
- `[WONDER]` - Breathless discovery with ethereal visuals and floating UI elements
- `[ANGUISH]` - Sharp contrasts and fragmented animations expressing pain
- `[ECSTASY]` - Psychedelic colors and rapid pulsing for moments of revelation
- `[CONTEMPLATION]` - Muted earth tones with slow, focused transitions
- And more...

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for UI)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/virtual-literature-companion.git
cd virtual-literature-companion
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Install UI dependencies**
```bash
cd ui
npm install
```

### Running the Application

1. **Start the backend**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start the frontend** (in a new terminal)
```bash
cd ui
npm run dev
```

3. **Access the application**
Open your browser to `http://localhost:3000`

## 🏗️ Architecture

### Project Structure
```
virtual-literature-companion/
├── wild-genius-prof/        # Core AI persona agent
│   ├── prompts/            # System prompts and templates
│   ├── graph.py            # LangGraph conversation flow
│   ├── state.py            # State management
│   ├── tools.py            # Agent tools
│   └── persona.py          # Honcho integration
├── processors/             # Text processing pipeline
├── generation/             # Multimedia generation services
│   ├── image/             # AI image generation
│   ├── voice/             # Emotional voice synthesis
│   └── music/             # Adaptive music generation
├── api/                   # FastAPI backend
├── ui/                    # React frontend
└── data/                  # Processed books and conversations
```

### Technology Stack
- **Backend**: FastAPI, LangChain, LangGraph
- **AI/ML**: OpenAI/Anthropic, Honcho (memory), ChromaDB (vectors)
- **Frontend**: React, Three.js, Framer Motion
- **Multimedia**: DALL-E 3, ElevenLabs, Custom music engine
- **Infrastructure**: Docker, WebSockets, Redis

## 📖 Usage Guide

### Starting a Conversation

1. **Upload or Select a Book**: Currently optimized for "The Brothers Karamazov"
2. **Set Your Reading Progress**: Tell the professor where you are in the book
3. **Begin Discussion**: Ask questions, share thoughts, or request guidance

### Example Interactions

```
You: I just finished the Grand Inquisitor chapter. I'm struggling with Ivan's argument.

Professor: [TURMOIL] Ah... yes... the Grand Inquisitor. *shifts uncomfortably* 
This chapter has haunted me for decades. Tell me, what specific aspect of 
Ivan's argument disturbs you most? Is it his claim about human freedom, or 
perhaps the image of Christ's silence?

[The UI darkens, swirling with conflicted colors as unsettling music plays]
```

## 🛠️ Configuration

### Emotion Intensity Thresholds
Edit `wild-genius-prof/config.py`:
```python
emotion_config = EmotionConfig(
    high_intensity_threshold=0.7,  # Triggers image generation
    transition_duration=2000,      # UI transition time (ms)
)
```

### Visual Styles
Customize book-specific aesthetics in `config.py`:
```python
visual_styles = {
    "brothers_karamazov": "Russian literary realism with expressionist touches..."
}
```

## 🧪 Development

### Running Tests
```bash
pytest tests/ -v --cov=wild_genius_prof
```

### Code Formatting
```bash
black .
flake8 .
mypy .
```

### Adding New Books
1. Create processor for the book's PDF
2. Define visual style and themes
3. Update professor's prompts with book-specific knowledge
4. Test spoiler protection thoroughly

## 🎨 Customization

### Adding Emotional States
1. Define the emotion in `state.py`
2. Create prompt guidance in `emotional_tags.txt`
3. Map to UI colors/animations
4. Define voice parameters
5. Create musical theme

### Extending Multimedia
- **Images**: Modify style prompts in `generation/image/`
- **Voice**: Adjust parameters in `generation/voice/`
- **Music**: Edit emotion themes in `generation/music/`

## 📊 Performance Optimization

- **Streaming Responses**: LLM responses stream in real-time
- **Async Generation**: Multimedia generates in parallel
- **Smart Caching**: Common responses and media cached
- **Progressive Enhancement**: Core chat works without multimedia

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Priority Areas
- [ ] Support for more books
- [ ] Additional language support
- [ ] Mobile app development
- [ ] Accessibility improvements
- [ ] Educational analytics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the depth and complexity of "The Brothers Karamazov"
- Built with love for literature and AI
- Special thanks to the Honcho team for memory infrastructure

---

**Note**: This is an experimental project pushing the boundaries of AI-human interaction in literature education. The goal is not to replace human discussion but to create a unique, emotionally resonant companion for solo readers.

*"The mystery of human existence lies not in just staying alive, but in finding something to live for."* - Fyodor Dostoevsky
