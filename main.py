#!/usr/bin/env python3
"""
Virtual Literature Companion - CLI Entry Point

This script provides a command-line interface to test the Wild Genius Professor agent.
Use this to interact with the AI literature companion until the full web interface is built.

Usage:
    python main.py
    
Requirements:
    - Set OPENAI_API_KEY or ANTHROPIC_API_KEY in environment
    - Optionally set HONCHO_API_KEY for persistent memory
"""

import os
import sys
import logging
import asyncio
from typing import Optional
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the wild_genius_prof module
from wild_genius_prof import (
    WildGeniusProfessorGraph,
    create_initial_state,
    config
)

# Configure comprehensive logging
def setup_logging():
    """Set up detailed logging for debugging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler("logs/debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logger = logging.getLogger(__name__)
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("🎭 VIRTUAL LITERATURE COMPANION STARTING")
    logger.info("=" * 80)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Config debug mode: {config.debug}")
    logger.info(f"Log level: {config.log_level}")
    
    return logger


def check_environment():
    """Check that required environment variables are set."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Checking environment variables...")
    
    # Check for LLM API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    honcho_key = os.getenv("HONCHO_API_KEY")
    
    logger.debug(f"OPENAI_API_KEY present: {bool(openai_key)}")
    logger.debug(f"ANTHROPIC_API_KEY present: {bool(anthropic_key)}")
    logger.debug(f"HONCHO_API_KEY present: {bool(honcho_key)}")
    
    if not openai_key and not anthropic_key:
        logger.error("❌ No LLM API key found!")
        logger.error("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        logger.error("Example: export OPENAI_API_KEY='your-key-here'")
        return False
    
    if not honcho_key:
        logger.warning("⚠️  HONCHO_API_KEY not set - memory features will be limited")
    
    logger.info("✅ Environment check complete")
    return True


def display_welcome():
    """Display welcome message and instructions."""
    print("\n" + "=" * 80)
    print("🎭✨ WELCOME TO THE VIRTUAL LITERATURE COMPANION ✨📚")
    print("=" * 80)
    print()
    print("Meet the Wild Genius Professor - an eccentric AI literature scholar")
    print("who experiences texts viscerally and guides you through Socratic dialogue.")
    print()
    print("📖 Currently focused on: The Brothers Karamazov by Dostoevsky")
    print()
    print("Commands:")
    print("  /progress <book> <chapter>  - Set your reading progress")
    print("  /emotion                    - Show current professor emotion")
    print("  /debug                      - Toggle debug output")
    print("  /quit                       - Exit the conversation")
    print()
    print("Simply type your thoughts or questions about the book to begin!")
    print("=" * 80)
    print()


class LiteratureCompanionCLI:
    """Command-line interface for the Wild Genius Professor."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("🏗️  Initializing Literature Companion CLI")
        
        self.debug_mode = config.debug
        self.graph: Optional[WildGeniusProfessorGraph] = None
        self.user_id = "cli_user"
        self.session_id = "cli_session_001"
        
        self.logger.debug(f"Debug mode: {self.debug_mode}")
        self.logger.debug(f"User ID: {self.user_id}")
        self.logger.debug(f"Session ID: {self.session_id}")
    
    def initialize_agent(self):
        """Initialize the Wild Genius Professor agent."""
        self.logger.info("🤖 Initializing Wild Genius Professor agent...")
        
        try:
            # Determine which LLM to use
            if os.getenv("OPENAI_API_KEY"):
                llm_provider = "openai"
                model_name = "gpt-4-turbo-preview"
            else:
                llm_provider = "anthropic"
                model_name = "claude-3-opus-20240229"
                
            self.logger.info(f"Using LLM provider: {llm_provider}")
            self.logger.info(f"Model: {model_name}")
            
            # Initialize the agent with comprehensive error handling
            self.graph = WildGeniusProfessorGraph(
                llm_provider=llm_provider,
                model_name=model_name,
                honcho_api_key=os.getenv("HONCHO_API_KEY")
            )
            
            self.logger.info("✅ Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False
    
    def process_command(self, user_input: str) -> bool:
        """Process special commands. Returns True if command was handled."""
        self.logger.debug(f"Checking if input is command: {user_input}")
        
        if not user_input.startswith('/'):
            return False
            
        parts = user_input.split()
        command = parts[0]
        
        self.logger.info(f"Processing command: {command}")
        
        if command == '/quit':
            print("\n👋 Farewell! May the texts continue to inspire you.")
            self.logger.info("User quit the application")
            return True
            
        elif command == '/debug':
            self.debug_mode = not self.debug_mode
            print(f"\n🔧 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            self.logger.info(f"Debug mode toggled to: {self.debug_mode}")
            
        elif command == '/progress':
            if len(parts) >= 3:
                try:
                    book = int(parts[1])
                    chapter = int(parts[2])
                    print(f"\n📍 Setting progress to Book {book}, Chapter {chapter}")
                    self.logger.info(f"User set progress: Book {book}, Chapter {chapter}")
                    # Note: In a full implementation, this would update the state
                except ValueError:
                    print("\n❌ Usage: /progress <book_number> <chapter_number>")
                    self.logger.warning("Invalid progress command format")
            else:
                print("\n❌ Usage: /progress <book_number> <chapter_number>")
                
        elif command == '/emotion':
            print("\n😊 Current professor emotion: CONTEMPLATION")
            print("(This would show the actual emotional state in full implementation)")
            self.logger.info("User requested emotion status")
            
        else:
            print(f"\n❓ Unknown command: {command}")
            self.logger.warning(f"Unknown command attempted: {command}")
            
        return False  # Continue conversation unless /quit
    
    async def chat_loop(self):
        """Main conversation loop."""
        self.logger.info("🗣️  Starting chat loop")
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("\n💭 You: ").strip()
                    self.logger.debug(f"User input received: '{user_input}'")
                    
                    if not user_input:
                        continue
                        
                    # Process commands
                    if self.process_command(user_input):
                        if user_input == '/quit':
                            break
                        continue
                    
                    # Process with the agent
                    self.logger.info("📤 Sending message to Wild Genius Professor")
                    
                    if self.debug_mode:
                        print(f"\n🔧 DEBUG: Processing message through LangGraph...")
                    
                    # This is where we'd call the agent
                    # For now, show a mock response until full implementation
                    response = await self.get_professor_response(user_input)
                    
                    print(f"\n🎭 Professor: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted by user. Goodbye!")
                    self.logger.info("User interrupted with Ctrl+C")
                    break
                except Exception as e:
                    self.logger.error(f"Error in chat loop: {e}")
                    self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                    print(f"\n❌ An error occurred: {e}")
                    if self.debug_mode:
                        print(f"Full traceback:\n{traceback.format_exc()}")
                        
        except Exception as e:
            self.logger.error(f"Fatal error in chat loop: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
    async def get_professor_response(self, user_input: str) -> str:
        """Get response from the Wild Genius Professor."""
        self.logger.debug("Generating professor response")
        
        try:
            if self.graph:
                # This would use the actual graph when fully implemented
                response_data = self.graph.run(
                    user_message=user_input,
                    user_id=self.user_id, 
                    session_id=self.session_id
                )
                return response_data.get("response", "I seem to be at a loss for words...")
            else:
                # Mock response for now
                return ("[CONTEMPLATION] Ah, a fascinating observation... "
                       "Tell me, what draws you to this particular aspect? "
                       "I find myself wondering about the deeper currents here... 🤔")
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return (f"[TURMOIL] Forgive me, I seem to be experiencing some internal turbulence... "
                   f"Perhaps we could try rephrasing that thought? 😵")


async def main():
    """Main entry point."""
    logger = setup_logging()
    
    try:
        logger.info("🚀 Starting Virtual Literature Companion")
        
        # Check environment
        if not check_environment():
            logger.error("Environment check failed, exiting")
            sys.exit(1)
            
        # Display welcome
        display_welcome()
        
        # Initialize CLI
        cli = LiteratureCompanionCLI()
        
        # Initialize agent (this may fail gracefully)
        if not cli.initialize_agent():
            logger.warning("Agent initialization failed, running in mock mode")
            print("⚠️  Running in mock mode - some features may be limited")
            
        # Start chat loop
        await cli.chat_loop()
        
        logger.info("🏁 Virtual Literature Companion shutting down")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"\n💥 Fatal error: {e}")
        if config.debug:
            print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())