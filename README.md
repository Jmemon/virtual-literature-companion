# Virtual Literature Companion

An AI-powered interactive literary companion designed to foster deeper thought and understanding of literary texts through contextually-aware dialogue and embodied gestural expression.

## Overview

The Virtual Literature Companion is a sophisticated application that serves as an intelligent reading partner, engaging users in thoughtful discussions about literature while respecting their reading progress. The system features a unique gestural vocabulary that creates an emotionally resonant and physically expressive interface.

### Key Features

- **Contextual Awareness**: Strictly adheres to user's reading progress, never revealing information beyond what has been read
- **Dynamic Persona**: AI personality tailored to match the philosophical and emotional depth of the source material
- **Embodied UI**: Rich gestural vocabulary with synchronized visual and audio effects
- **Citation System**: Precise text references with chapter and page citations
- **Table of Contents Navigation**: Automatic extraction and interactive navigation
- **Real-time Communication**: WebSocket-based conversation with typing indicators

## Gestural Vocabulary

The application uses an innovative gesture system where the AI can express emotions and emphasize concepts through UI transformations:

- **[GESTURE:LEAN_IN]**: Interface moves closer, creating intimacy
- **[GESTURE:PULL_BACK]**: Creates contemplative space after heavy revelations
- **[GESTURE:TREMBLE]**: Individual letters vibrate with overwhelming emotion
- **[GESTURE:ILLUMINATE:keyword1,keyword2]**: Key words pulse with golden significance
- **[GESTURE:FRAGMENT:word]**: Words break apart and reform to show shattered understanding
- **[GESTURE:WHISPER]**: Text fades to near-transparency, requiring focused attention
- **[GESTURE:GRIP]**: Borders thicken, container contracts to seize attention
- **[GESTURE:SHATTER]**: The entire interface cracks like breaking glass
- **[GESTURE:BREATHE]**: Everything expands/contracts in meditative rhythm
- **[GESTURE:REACH]**: Words create bridges to connect ideas visually
- **[GESTURE:DANCE]**: Elements spiral in synchronized joy

## Technical Architecture

### Core Components

- **LangGraph Agent**: Orchestrates conversation flow and state management
- **Book Processor**: Handles text ingestion, chapter extraction, and navigation
- **Gesture System**: Manages UI animations and audio synchronization
- **Citation Extractor**: Identifies and validates text references
- **WebSocket Server**: Enables real-time bidirectional communication

### Technology Stack

- **Backend**: Python 3.11+, FastAPI, LangGraph, LangChain
- **Frontend**: Vanilla JavaScript with modern ES6+ features
- **Styling**: Custom CSS with CSS Grid and Flexbox
- **Real-time**: WebSockets for live communication
- **AI Models**: Support for Anthropic Claude and OpenAI GPT models

## Installation

### Prerequisites

- Python 3.11 or higher
- Node.js (for development tools)
- An API key for either Anthropic or OpenAI

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/virtual-literature-companion.git
   cd virtual-literature-companion
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. **Run the application**
   ```bash
   python -m src.main
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

## Usage

### Starting a Reading Session

1. **Upload Book Text**: Click "Upload Book Text File" and select a `.txt` file containing the full text of your book
2. **Enter Book Details**: Provide the title and author
3. **Begin Session**: Click "Begin Reading Session" to start

### During Your Reading Session

1. **Report Progress**: Tell the AI what chapter and page you've read up to (e.g., "I've read up to Chapter 3, page 45")
2. **Ask Questions**: Engage with questions about themes, characters, or passages you've read
3. **Experience Gestures**: Watch as the AI uses gestural expressions to emphasize emotional and philosophical points
4. **Navigate Chapters**: Use the table of contents sidebar to quickly update your reading progress

### Example Interactions

```
User: I've read up to Chapter 2, page 30

AI: [GESTURE:LEAN_IN] Ah, so you've just encountered the pivotal conversation 
between Ivan and Alyosha. The way Dostoevsky [GESTURE:ILLUMINATE:faith,doubt] 
interweaves their opposing worldviews creates a fascinating tension...

User: What do you think Ivan really means when he talks about returning his ticket?

AI: [GESTURE:TREMBLE] Ivan's metaphor of "returning his ticket" carries such 
profound weight... [GESTURE:GRIP] He's essentially rejecting the entire premise 
of a world where innocent suffering exists...
```

## Docker Deployment

### Using Docker Compose

1. **Build and run**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   Navigate to `http://localhost:8000`

### Manual Docker Build

```bash
# Build the image
docker build -t literary-companion .

# Run the container
docker run -p 8000:8000 --env-file .env literary-companion
```

## Development

### Project Structure

```
virtual-literature-companion/
├── src/
│   ├── agents/          # LangGraph agents and conversation flow
│   ├── models/          # Pydantic models and state management
│   ├── prompts/         # AI persona and system prompts
│   ├── services/        # Gesture and citation services
│   ├── tools/           # Book processing and analysis tools
│   └── main.py          # FastAPI application entry point
├── static/
│   ├── css/            # Stylesheets and animations
│   ├── js/             # Frontend JavaScript
│   └── audio/          # Gesture sound effects
├── tests/              # Test suite
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Docker orchestration
└── README.md          # This file
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

The project uses:
- **Black** for Python code formatting
- **Ruff** for linting
- **Type hints** throughout the codebase

## API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | AI provider (anthropic/openai) | anthropic |
| `MODEL_NAME` | Specific model to use | claude-3-opus-20240229 |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `APP_HOST` | Application host | 0.0.0.0 |
| `APP_PORT` | Application port | 8000 |
| `LOG_LEVEL` | Logging level | info |

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Ensure the backend is running
   - Check browser console for errors
   - Verify firewall settings

2. **Gestures Not Working**
   - Check browser compatibility (Chrome/Firefox recommended)
   - Ensure JavaScript is enabled
   - Look for console errors

3. **Book Processing Issues**
   - Verify text file is UTF-8 encoded
   - Check for proper chapter formatting
   - Ensure file isn't too large (>10MB may cause issues)

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Support for additional book formats (EPUB, PDF)
- [ ] Multi-language support
- [ ] Book-specific gesture vocabularies
- [ ] Collaborative reading sessions
- [ ] Mobile-responsive design improvements
- [ ] Integration with e-reader applications
- [ ] Persistent conversation history
- [ ] Advanced citation management

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the philosophical depths of "The Brothers Karamazov"
- Built with LangGraph and LangChain for advanced AI orchestration
- Gesture system designed to embody the emotional resonance of literature

---

**Note**: This application requires significant computational resources and API access. Ensure you have appropriate API limits and budget for extended use.