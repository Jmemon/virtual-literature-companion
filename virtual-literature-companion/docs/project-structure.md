# Virtual Literature Companion - Project Structure

```
virtual-literature-companion/
│
├── docs/                           # Documentation
│   ├── project-structure.md       # This file
│   ├── requirements.md            # Project requirements document
│   └── architecture.md            # System architecture overview
│
├── wild-genius-prof/              # Core LLM-driven persona agent
│   ├── prompts/                   # Prompt templates
│   │   ├── system.txt            # System prompt defining persona
│   │   ├── emotional_tags.txt    # Emotion tag definitions and usage
│   │   ├── socratic.txt          # Socratic questioning templates
│   │   └── citation.txt          # Text citation format templates
│   │
│   ├── graph.py                   # LangGraph implementation
│   ├── state.py                   # Agent state management
│   ├── tools.py                   # Agent tools (citations, emotion, etc.)
│   ├── persona.py                 # Persona management with Honcho
│   └── config.py                  # Agent configuration
│
├── processors/                    # Text processing modules
│   ├── pdf_extractor.py          # PDF text extraction
│   ├── toc_parser.py             # Table of contents parsing
│   ├── text_chunker.py           # Intelligent text chunking
│   ├── citation_index.py         # Build searchable citation index
│   └── spoiler_guard.py          # Prevent spoilers based on reading progress
│
├── ui/                           # Frontend application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx          # Main chat interface
│   │   │   ├── EmotiveShape.tsx # Abstract shape that morphs with emotion
│   │   │   ├── ThemeEngine.tsx  # Dynamic theming based on emotion
│   │   │   └── ImageGallery.tsx # Display generated images
│   │   ├── hooks/
│   │   │   ├── useEmotion.ts    # Handle emotion state changes
│   │   │   └── useWebSocket.ts  # Real-time communication
│   │   └── lib/
│   │       ├── emotions.ts       # Emotion definitions and mappings
│   │       └── animations.ts     # UI animation configurations
│   │
│   ├── public/
│   └── package.json
│
├── generation/                    # Multimedia generation services
│   ├── image/
│   │   ├── README.md             # Image generation overview
│   │   ├── style_manager.py      # Maintain consistent style
│   │   └── prompt_builder.py     # Build image prompts from context
│   │
│   ├── voice/
│   │   ├── README.md             # Voice synthesis overview
│   │   ├── emotion_mapper.py     # Map emotions to voice parameters
│   │   └── tts_engine.py         # Text-to-speech with emotion
│   │
│   └── music/
│       ├── README.md             # Music generation overview
│       ├── emotion_themes.py     # Define musical themes per emotion
│       └── generator.py          # Procedural music generation
│
├── api/                          # Backend API
│   ├── main.py                   # FastAPI application
│   ├── routers/
│   │   ├── chat.py              # Chat endpoints
│   │   ├── books.py             # Book management
│   │   └── generation.py        # Multimedia generation endpoints
│   ├── services/
│   │   ├── agent_service.py     # Interface with wild-genius-prof
│   │   └── book_service.py      # Book processing and storage
│   └── models/
│       ├── chat.py              # Chat data models
│       └── emotion.py           # Emotion state models
│
├── data/                         # Data storage
│   ├── books/                    # Processed book data
│   │   └── brothers_karamazov/
│   │       ├── text_chunks.json
│   │       ├── toc.json
│   │       └── metadata.json
│   └── conversations/            # Conversation history
│
├── tests/                        # Test suite
│   ├── test_persona.py
│   ├── test_processors.py
│   └── test_emotion_flow.py
│
├── docker-compose.yml            # Multi-service orchestration
├── Dockerfile                    # Container configuration
├── pyproject.toml               # Python project configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # Project overview