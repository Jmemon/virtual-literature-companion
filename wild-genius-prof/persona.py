"""
Persona management using Honcho for persistent memory and psychological modeling.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from honcho import Honcho
from honcho.client import HonchoClient

from .state import ProfessorMemory, EmotionalState, ConversationContext


class WildGeniusProfessorPersona:
    """
    Manages the Wild Genius Professor's personality, memory, and psychological insights
    using Honcho for persistence and theory of mind modeling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize persona with Honcho connection."""
        self.api_key = api_key or os.getenv("HONCHO_API_KEY")
        if not self.api_key:
            raise ValueError("Honcho API key required")
            
        self.client = HonchoClient(api_key=self.api_key)
        self.app_name = "wild-genius-professor"
        
        # Create or get the app
        self._setup_app()
        
    def _setup_app(self):
        """Create or retrieve the Honcho app."""
        try:
            # Get existing app or create new one
            apps = self.client.apps.list()
            self.app = next((app for app in apps if app.name == self.app_name), None)
            
            if not self.app:
                self.app = self.client.apps.create(name=self.app_name)
        except Exception as e:
            print(f"Error setting up Honcho app: {e}")
            raise
            
    def get_or_create_user(self, user_id: str) -> Any:
        """Get or create a Honcho user."""
        try:
            # Try to get existing user
            users = self.client.apps.users.list(app_id=self.app.id)
            user = next((u for u in users if u.name == user_id), None)
            
            if not user:
                # Create new user
                user = self.client.apps.users.create(
                    app_id=self.app.id,
                    name=user_id,
                    metadata={"created_at": datetime.now().isoformat()}
                )
                
            return user
        except Exception as e:
            print(f"Error managing user: {e}")
            raise
            
    def create_session(self, user_id: str, book: str = "brothers_karamazov") -> Any:
        """Create a new conversation session."""
        user = self.get_or_create_user(user_id)
        
        session = self.client.apps.users.sessions.create(
            app_id=self.app.id,
            user_id=user.id,
            metadata={
                "book": book,
                "started_at": datetime.now().isoformat(),
                "persona": "wild-genius-professor"
            }
        )
        
        return session
        
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        emotions: Optional[List[EmotionalState]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the conversation."""
        message_metadata = metadata or {}
        
        if emotions:
            message_metadata["emotions"] = [e.value for e in emotions]
            
        self.client.apps.users.sessions.messages.create(
            app_id=self.app.id,
            session_id=session_id,
            role=role,
            content=content,
            metadata=message_metadata
        )
        
    def get_user_insights(self, user_id: str, query: str) -> str:
        """
        Query Honcho's dialectic API for psychological insights about the user.
        
        Args:
            user_id: The user to analyze
            query: Natural language question about the user
            
        Returns:
            Insight from Honcho's theory of mind model
        """
        user = self.get_or_create_user(user_id)
        
        # Use Honcho's dialectic endpoint to get insights
        response = self.client.apps.users.sessions.dialectic(
            app_id=self.app.id,
            user_id=user.id,
            query=query
        )
        
        return response.content
        
    def analyze_learning_style(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's learning style based on conversation history."""
        insights = {}
        
        # Query different aspects of learning style
        queries = {
            "abstract_vs_concrete": "Does this user prefer abstract philosophical discussions or concrete examples?",
            "pace_preference": "Does this user prefer fast-paced exchanges or slow, contemplative discussion?",
            "emotional_engagement": "How does this user respond to emotional content versus analytical content?",
            "question_style": "What types of questions engage this user most effectively?",
            "depth_preference": "Does this user prefer to explore topics deeply or cover more ground?"
        }
        
        for aspect, query in queries.items():
            try:
                insight = self.get_user_insights(user_id, query)
                insights[aspect] = insight
            except Exception as e:
                insights[aspect] = f"Unable to determine: {e}"
                
        return insights
        
    def get_conversation_themes(self, session_id: str) -> List[str]:
        """Extract major themes from a conversation."""
        try:
            # Get session messages
            messages = self.client.apps.users.sessions.messages.list(
                app_id=self.app.id,
                session_id=session_id
            )
            
            # Extract themes from metadata
            themes = set()
            for message in messages:
                if message.metadata and "themes" in message.metadata:
                    themes.update(message.metadata["themes"])
                    
            return list(themes)
        except Exception as e:
            print(f"Error getting themes: {e}")
            return []
            
    def suggest_next_topic(self, user_id: str, current_topic: str) -> str:
        """Suggest the next topic based on user's interests and history."""
        query = f"""
        Given that we're currently discussing {current_topic}, 
        what related topic would most engage this user based on their interests and past conversations?
        Consider their favorite themes and the connections they've made before.
        """
        
        return self.get_user_insights(user_id, query)
        
    def calibrate_emotional_response(self, user_id: str, user_message: str) -> EmotionalState:
        """Determine appropriate emotional response based on user's state and preferences."""
        query = f"""
        The user just said: "{user_message}"
        Based on their communication style and what engages them emotionally,
        what emotional tone should I adopt in my response?
        Choose from: WONDER, CONTEMPLATION, FERVOR, SERENITY, MELANCHOLY
        """
        
        insight = self.get_user_insights(user_id, query)
        
        # Parse the response to extract emotion
        # Simple keyword matching for now
        emotion_map = {
            "wonder": EmotionalState.WONDER,
            "contemplation": EmotionalState.CONTEMPLATION,
            "fervor": EmotionalState.FERVOR,
            "serenity": EmotionalState.SERENITY,
            "melancholy": EmotionalState.MELANCHOLY
        }
        
        insight_lower = insight.lower()
        for keyword, emotion in emotion_map.items():
            if keyword in insight_lower:
                return emotion
                
        return EmotionalState.CONTEMPLATION  # Default
        
    def get_personalized_question(self, user_id: str, topic: str) -> str:
        """Generate a personalized Socratic question."""
        query = f"""
        Generate a thought-provoking question about {topic} that would resonate with this specific user.
        Consider their interests, the depth they enjoy, and their preferred style of inquiry.
        The question should guide them to discover insights themselves.
        """
        
        return self.get_user_insights(user_id, query)
        
    def save_breakthrough_moment(
        self, 
        user_id: str, 
        session_id: str,
        insight: str,
        context: Dict[str, Any]
    ):
        """Record a significant moment of understanding."""
        user = self.get_or_create_user(user_id)
        
        # Add to user metadata
        metadata = user.metadata or {}
        breakthroughs = metadata.get("breakthroughs", [])
        
        breakthroughs.append({
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "insight": insight,
            "context": context
        })
        
        metadata["breakthroughs"] = breakthroughs
        
        # Update user metadata
        self.client.apps.users.update(
            app_id=self.app.id,
            user_id=user.id,
            metadata=metadata
        )
        
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of the user's journey."""
        summary_query = """
        Provide a comprehensive summary of this user's journey with The Brothers Karamazov:
        - What themes resonate most with them?
        - What's their intellectual and emotional engagement style?  
        - What insights have they discovered?
        - What questions drive their curiosity?
        Format as a psychological and intellectual profile.
        """
        
        profile = self.get_user_insights(user_id, summary_query)
        
        # Get additional structured data
        user = self.get_or_create_user(user_id)
        
        return {
            "profile": profile,
            "metadata": user.metadata,
            "created_at": user.metadata.get("created_at") if user.metadata else None,
            "breakthroughs": user.metadata.get("breakthroughs", []) if user.metadata else []
        }