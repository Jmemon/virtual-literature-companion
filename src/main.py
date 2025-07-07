#!/usr/bin/env python3
"""
Main entry point for the Virtual Literature Companion.

This script starts the FastAPI server and initializes all components.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main entry point."""
    # Import here to ensure path is set
    import uvicorn
    from api.main import app
    from src.config import settings
    
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║   Virtual Literature Companion 📚🎭          ║
    ║   An emotional AI for literary discussions   ║
    ╚══════════════════════════════════════════════╝
    
    Starting server at http://{settings.host}:{settings.port}
    API documentation at http://{settings.host}:{settings.port}/docs
    
    Press CTRL+C to stop
    """)
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()