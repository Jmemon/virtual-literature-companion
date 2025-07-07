"""
Honcho client for conversation memory management.

This module integrates with Honcho to provide:
- Conversation history storage and retrieval
- User preference tracking
- Context-aware memory for the persona agent
- Session management
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import httpx
from loguru import logger

from ..config import settings
from ..models.schemas import (
    ConversationTurn,
    ConversationSession,
    EmotionalResponse
)


class HonchoMemoryClient:
    """
    Client for interacting with Honcho memory service.
    
    Honcho provides persistent conversation memory that enables:
    - Long-term memory across sessions
    - User-specific context and preferences
    - Semantic search over conversation history
    - Structured metadata storage
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.honcho_api_key
        self.base_url = base_url or settings.honcho_base_url
        self.app_id = settings.honcho_app_id
        
        if not self.api_key:
            logger.warning("Honcho API key not provided. Memory features will be limited.")
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def initialize_app(self) -> bool:
        """
        Initialize or verify the Honcho application.
        
        Returns:
            bool: True if successful
        """
        try:
            # Check if app exists
            response = await self.client.get(f"/apps/{self.app_id}")
            
            if response.status_code == 404:
                # Create app
                create_response = await self.client.post(
                    "/apps",
                    json={
                        "name": self.app_id,
                        "metadata": {
                            "type": "literature_companion",
                            "version": "1.0",
                            "created_at": datetime.utcnow().isoformat()
                        }
                    }
                )
                create_response.raise_for_status()
                logger.info(f"Created Honcho app: {self.app_id}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Honcho app: {e}")
            return False
            
    async def create_or_get_user(self, user_id: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create or retrieve a user in Honcho.
        
        Args:
            user_id: Unique user identifier
            metadata: Optional user metadata
            
        Returns:
            User object from Honcho
        """
        try:
            # Try to get existing user
            response = await self.client.get(
                f"/apps/{self.app_id}/users/{user_id}"
            )
            
            if response.status_code == 200:
                return response.json()
                
            # Create new user
            create_response = await self.client.post(
                f"/apps/{self.app_id}/users",
                json={
                    "name": user_id,
                    "metadata": metadata or {
                        "created_at": datetime.utcnow().isoformat(),
                        "preferences": {},
                        "reading_history": []
                    }
                }
            )
            create_response.raise_for_status()
            return create_response.json()
            
        except Exception as e:
            logger.error(f"Failed to create/get user {user_id}: {e}")
            return {"name": user_id, "metadata": {}}
            
    async def create_session(self, user_id: str, book_id: str, 
                           metadata: Optional[Dict] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            book_id: Book being discussed
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        try:
            response = await self.client.post(
                f"/apps/{self.app_id}/users/{user_id}/sessions",
                json={
                    "location_id": book_id,
                    "metadata": {
                        "book_id": book_id,
                        "started_at": datetime.utcnow().isoformat(),
                        **(metadata or {})
                    }
                }
            )
            response.raise_for_status()
            session = response.json()
            return session["uuid"]
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            # Return a local session ID as fallback
            return f"local_session_{datetime.utcnow().timestamp()}"
            
    async def add_message(self, session_id: str, user_id: str, 
                         role: str, content: str, 
                         metadata: Optional[Dict] = None) -> bool:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Current session ID
            user_id: User identifier
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata (emotions, citations, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            response = await self.client.post(
                f"/apps/{self.app_id}/users/{user_id}/sessions/{session_id}/messages",
                json={
                    "is_user": role == "user",
                    "content": content,
                    "metadata": metadata or {}
                }
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return False
            
    async def add_conversation_turn(self, session_id: str, user_id: str,
                                  turn: ConversationTurn) -> bool:
        """
        Add a complete conversation turn (user message + assistant response).
        
        Args:
            session_id: Current session ID
            user_id: User identifier
            turn: ConversationTurn object
            
        Returns:
            bool: Success status
        """
        # Add user message
        user_success = await self.add_message(
            session_id, user_id, "user", turn.user_message,
            metadata={"timestamp": turn.timestamp.isoformat()}
        )
        
        # Add assistant response with emotional metadata
        assistant_metadata = {
            "timestamp": turn.assistant_response.timestamp.isoformat(),
            "emotions": [e.value for e in turn.assistant_response.emotions],
            "emotion_intensities": turn.assistant_response.emotion_intensities,
            "citations": [c.dict() for c in turn.assistant_response.citations]
        }
        
        assistant_success = await self.add_message(
            session_id, user_id, "assistant", 
            turn.assistant_response.text,
            metadata=assistant_metadata
        )
        
        return user_success and assistant_success
        
    async def get_conversation_history(self, user_id: str, session_id: Optional[str] = None,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a user.
        
        Args:
            user_id: User identifier
            session_id: Optional specific session
            limit: Maximum messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            if session_id:
                url = f"/apps/{self.app_id}/users/{user_id}/sessions/{session_id}/messages"
            else:
                url = f"/apps/{self.app_id}/users/{user_id}/messages"
                
            response = await self.client.get(
                url,
                params={"limit": limit, "reverse": True}
            )
            response.raise_for_status()
            
            messages = response.json().get("items", [])
            
            # Convert to our format
            history = []
            for msg in messages:
                history.append({
                    "role": "user" if msg.get("is_user") else "assistant",
                    "content": msg.get("content", ""),
                    "metadata": msg.get("metadata", {}),
                    "created_at": msg.get("created_at")
                })
                
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
            
    async def get_relevant_context(self, user_id: str, query: str, 
                                 book_id: Optional[str] = None,
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve contextually relevant conversation history.
        
        Uses Honcho's semantic search to find relevant past conversations.
        
        Args:
            user_id: User identifier
            query: Current query/topic
            book_id: Optional book filter
            limit: Maximum results
            
        Returns:
            List of relevant conversation snippets
        """
        try:
            params = {
                "query": query,
                "limit": limit
            }
            
            if book_id:
                params["filter"] = {"book_id": book_id}
                
            response = await self.client.get(
                f"/apps/{self.app_id}/users/{user_id}/messages/search",
                params=params
            )
            response.raise_for_status()
            
            return response.json().get("items", [])
            
        except Exception as e:
            logger.error(f"Failed to search conversation history: {e}")
            return []
            
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user reading preferences and habits.
        
        Args:
            user_id: User identifier
            preferences: Preference updates
            
        Returns:
            bool: Success status
        """
        try:
            # Get current user data
            user_response = await self.client.get(
                f"/apps/{self.app_id}/users/{user_id}"
            )
            user_response.raise_for_status()
            
            user_data = user_response.json()
            current_metadata = user_data.get("metadata", {})
            current_preferences = current_metadata.get("preferences", {})
            
            # Merge preferences
            updated_preferences = {**current_preferences, **preferences}
            current_metadata["preferences"] = updated_preferences
            current_metadata["updated_at"] = datetime.utcnow().isoformat()
            
            # Update user
            update_response = await self.client.patch(
                f"/apps/{self.app_id}/users/{user_id}",
                json={"metadata": current_metadata}
            )
            update_response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
            
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences dictionary
        """
        try:
            response = await self.client.get(
                f"/apps/{self.app_id}/users/{user_id}"
            )
            response.raise_for_status()
            
            user_data = response.json()
            return user_data.get("metadata", {}).get("preferences", {})
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return {}
            
    async def add_reading_progress(self, user_id: str, book_id: str,
                                 chapter: int, progress: float) -> bool:
        """
        Track user's reading progress.
        
        Args:
            user_id: User identifier
            book_id: Book identifier
            chapter: Current chapter
            progress: Reading progress (0-1)
            
        Returns:
            bool: Success status
        """
        try:
            # Get current user data
            user_response = await self.client.get(
                f"/apps/{self.app_id}/users/{user_id}"
            )
            user_response.raise_for_status()
            
            user_data = user_response.json()
            metadata = user_data.get("metadata", {})
            reading_history = metadata.get("reading_history", [])
            
            # Update or add progress entry
            progress_entry = {
                "book_id": book_id,
                "chapter": chapter,
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Find existing entry
            existing_index = next(
                (i for i, entry in enumerate(reading_history) 
                 if entry.get("book_id") == book_id),
                None
            )
            
            if existing_index is not None:
                reading_history[existing_index] = progress_entry
            else:
                reading_history.append(progress_entry)
                
            metadata["reading_history"] = reading_history
            
            # Update user
            update_response = await self.client.patch(
                f"/apps/{self.app_id}/users/{user_id}",
                json={"metadata": metadata}
            )
            update_response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update reading progress: {e}")
            return False
            
    async def get_emotional_profile(self, user_id: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze user's emotional engagement patterns.
        
        Args:
            user_id: User identifier
            book_id: Optional book filter
            
        Returns:
            Emotional profile analysis
        """
        try:
            # Get conversation history
            history = await self.get_conversation_history(user_id, limit=50)
            
            emotion_counts = {}
            emotion_intensities = {}
            total_interactions = 0
            
            for msg in history:
                if msg["role"] == "assistant":
                    metadata = msg.get("metadata", {})
                    
                    # Filter by book if specified
                    if book_id and metadata.get("book_id") != book_id:
                        continue
                        
                    emotions = metadata.get("emotions", [])
                    intensities = metadata.get("emotion_intensities", {})
                    
                    total_interactions += 1
                    
                    for emotion in emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        
                        if emotion in intensities:
                            if emotion not in emotion_intensities:
                                emotion_intensities[emotion] = []
                            emotion_intensities[emotion].append(intensities[emotion])
                            
            # Calculate averages
            emotion_profile = {
                "total_interactions": total_interactions,
                "emotion_frequencies": {
                    emotion: count / total_interactions 
                    for emotion, count in emotion_counts.items()
                } if total_interactions > 0 else {},
                "average_intensities": {
                    emotion: sum(values) / len(values)
                    for emotion, values in emotion_intensities.items()
                    if values
                },
                "dominant_emotions": sorted(
                    emotion_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3] if emotion_counts else []
            }
            
            return emotion_profile
            
        except Exception as e:
            logger.error(f"Failed to get emotional profile: {e}")
            return {
                "total_interactions": 0,
                "emotion_frequencies": {},
                "average_intensities": {},
                "dominant_emotions": []
            }


# Convenience functions for synchronous usage

def create_honcho_client() -> HonchoMemoryClient:
    """Create a Honcho client instance."""
    return HonchoMemoryClient()


async def get_or_create_session(client: HonchoMemoryClient, user_id: str, 
                              book_id: str) -> str:
    """Get existing session or create new one."""
    # For now, always create new session
    # In production, might want to resume previous sessions
    return await client.create_session(user_id, book_id)