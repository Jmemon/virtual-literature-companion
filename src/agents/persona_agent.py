"""
Persona agent for emotional literary discussions.

This agent:
- Engages in deep, emotionally-aware discussions about literature
- Maintains character consistency
- Uses Honcho for conversation memory
- Generates responses with emotion tags
- Cites specific text passages
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from ..config import settings, get_prompt_template
from ..models.schemas import (
    EmotionTag,
    EmotionalResponse,
    Citation,
    TextChunk,
    ConversationTurn,
    PersonaState
)
from ..models.state import AgentState
from ..memory.honcho_client import HonchoMemoryClient


class PersonaAgent:
    """
    The emotional literature companion agent.
    
    This agent embodies a passionate, empathetic literature enthusiast
    who engages in deep discussions about books while expressing
    authentic emotions and maintaining conversation memory.
    """
    
    def __init__(self, 
                 character_name: str = "Athena",
                 model_name: Optional[str] = None,
                 honcho_client: Optional[HonchoMemoryClient] = None):
        self.character_name = character_name
        self.model_name = model_name or settings.llm_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_tokens=settings.max_tokens
        )
        self.honcho_client = honcho_client or HonchoMemoryClient()
        self.emotion_pattern = re.compile(r'\[emotion:\s*(\w+)\]')
        self.citation_pattern = re.compile(r'\[cite:\s*"([^"]+)"\s*(?:p\.?\s*(\d+))?\]')
        
    async def respond(self, state: AgentState) -> AgentState:
        """
        Generate an emotional response to the user's message.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated response
        """
        try:
            # Get user message
            user_message = state.get_latest_user_message()
            if not user_message:
                raise ValueError("No user message found")
                
            logger.info(f"Generating response for: {user_message[:100]}...")
            
            # Retrieve relevant book context
            relevant_chunks = await self._get_relevant_context(state, user_message)
            state.relevant_chunks = relevant_chunks
            
            # Get conversation history from Honcho
            conversation_history = await self._get_conversation_history(state)
            
            # Get user preferences
            user_preferences = await self.honcho_client.get_user_preferences(state.user_id)
            
            # Generate response
            response_text = await self._generate_response(
                user_message,
                relevant_chunks,
                conversation_history,
                user_preferences,
                state
            )
            
            # Parse emotions and citations
            emotions, clean_text = self._extract_emotions(response_text)
            citations = self._extract_citations(response_text, relevant_chunks)
            
            # Calculate emotion intensities based on context
            emotion_intensities = self._calculate_emotion_intensities(
                emotions,
                user_message,
                relevant_chunks
            )
            
            # Create emotional response
            emotional_response = EmotionalResponse(
                text=clean_text,
                emotions=emotions,
                emotion_intensities=emotion_intensities,
                citations=citations,
                voice_modulation=self._get_voice_modulation(emotions, emotion_intensities)
            )
            
            # Update state
            state.generated_response = emotional_response
            state.current_emotions = emotions
            state.update_emotional_state(emotions, emotion_intensities)
            
            # Create conversation turn
            turn = ConversationTurn(
                user_message=user_message,
                assistant_response=emotional_response,
                context_used=relevant_chunks
            )
            
            # Save to Honcho
            await self.honcho_client.add_conversation_turn(
                state.session_id,
                state.user_id,
                turn
            )
            
            # Update reading progress if discussing specific chapter
            if state.current_chapter and state.book_id:
                await self.honcho_client.add_reading_progress(
                    state.user_id,
                    state.book_id,
                    state.current_chapter,
                    0.5  # Midway through chapter
                )
                
            logger.info(f"Generated response with emotions: {[e.value for e in emotions]}")
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.add_error("persona_agent", str(e))
            return state
            
    async def _get_relevant_context(self, state: AgentState, 
                                   query: str) -> List[TextChunk]:
        """
        Retrieve relevant text chunks for the query.
        
        Args:
            state: Current state
            query: User query
            
        Returns:
            List of relevant text chunks
        """
        # For now, simple implementation - in production would use vector search
        relevant = []
        
        # If discussing specific chapter, prioritize those chunks
        if state.current_chapter:
            chapter_chunks = [
                chunk for chunk in state.processed_chunks
                if chunk.chapter_number == state.current_chapter
            ]
            relevant.extend(chapter_chunks[:5])
            
        # Add chunks based on keyword matching
        query_lower = query.lower()
        for chunk in state.processed_chunks:
            if any(keyword in chunk.content.lower() 
                   for keyword in query_lower.split()[:5]):
                relevant.append(chunk)
                if len(relevant) >= 10:
                    break
                    
        return relevant[:5]  # Limit to top 5 chunks
        
    async def _get_conversation_history(self, state: AgentState) -> str:
        """
        Get formatted conversation history from Honcho.
        
        Args:
            state: Current state
            
        Returns:
            Formatted conversation history
        """
        # Get recent messages from Honcho
        history = await self.honcho_client.get_conversation_history(
            state.user_id,
            state.session_id,
            limit=10
        )
        
        # Format for prompt
        formatted_history = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            
            # Include emotional metadata for assistant messages
            if role == "Assistant" and msg.get("metadata", {}).get("emotions"):
                emotions = msg["metadata"]["emotions"]
                formatted_history.append(f"{role} [{', '.join(emotions)}]: {content}")
            else:
                formatted_history.append(f"{role}: {content}")
                
        return "\n".join(formatted_history)
        
    async def _generate_response(self, user_message: str, 
                               relevant_chunks: List[TextChunk],
                               conversation_history: str,
                               user_preferences: Dict[str, Any],
                               state: AgentState) -> str:
        """
        Generate the actual response using the LLM.
        
        Args:
            user_message: User's message
            relevant_chunks: Relevant book context
            conversation_history: Past conversation
            user_preferences: User reading preferences
            state: Current state
            
        Returns:
            Generated response with emotion tags
        """
        # Format book context
        book_context = self._format_book_context(relevant_chunks)
        
        # Get book title
        book_title = state.book_metadata.title if state.book_metadata else "the book"
        
        # Build prompt
        system_prompt = get_prompt_template(
            "persona",
            "system",
            character_name=self.character_name,
            emotion_tags=", ".join(settings.emotion_tags)
        )
        
        discussion_prompt = get_prompt_template(
            "persona",
            "discussion",
            topic=self._extract_topic(user_message),
            book_title=book_title,
            book_context=book_context,
            conversation_history=conversation_history,
            user_message=user_message,
            character_name=self.character_name
        )
        
        # Add user preference context
        if user_preferences:
            discussion_prompt += f"\n\nUser preferences: {user_preferences}"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=discussion_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
        
    def _format_book_context(self, chunks: List[TextChunk]) -> str:
        """
        Format text chunks into readable context.
        
        Args:
            chunks: Text chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No specific context available."
            
        context_parts = []
        for i, chunk in enumerate(chunks[:3]):  # Limit to top 3
            chapter_info = f"Chapter {chunk.chapter_number}" if chunk.chapter_number else "Unknown chapter"
            page_info = f"Page {chunk.page_number}" if chunk.page_number else ""
            
            context_parts.append(
                f"[{chapter_info}, {page_info}]\n{chunk.content[:500]}..."
            )
            
        return "\n\n".join(context_parts)
        
    def _extract_topic(self, message: str) -> str:
        """
        Extract the main topic from user message.
        
        Args:
            message: User message
            
        Returns:
            Extracted topic
        """
        # Simple implementation - could be enhanced with NLP
        # Remove question words and get key phrases
        topic_words = message.lower().replace("?", "").split()
        stop_words = {"what", "how", "why", "when", "where", "who", "is", "are", 
                      "the", "a", "an", "about", "think", "feel"}
        
        topic_words = [w for w in topic_words if w not in stop_words]
        return " ".join(topic_words[:5])  # First 5 meaningful words
        
    def _extract_emotions(self, text: str) -> Tuple[List[EmotionTag], str]:
        """
        Extract emotion tags from response text.
        
        Args:
            text: Response text with emotion tags
            
        Returns:
            Tuple of (emotions list, clean text)
        """
        emotions = []
        
        # Find all emotion tags
        for match in self.emotion_pattern.finditer(text):
            emotion_str = match.group(1).lower()
            try:
                emotion = EmotionTag(emotion_str)
                if emotion not in emotions:
                    emotions.append(emotion)
            except ValueError:
                logger.warning(f"Unknown emotion tag: {emotion_str}")
                
        # Remove emotion tags from text
        clean_text = self.emotion_pattern.sub("", text).strip()
        
        # Default to curious if no emotions found
        if not emotions:
            emotions = [EmotionTag.CURIOUS]
            
        return emotions, clean_text
        
    def _extract_citations(self, text: str, chunks: List[TextChunk]) -> List[Citation]:
        """
        Extract and validate citations from response.
        
        Args:
            text: Response text
            chunks: Available text chunks
            
        Returns:
            List of citations
        """
        citations = []
        
        for match in self.citation_pattern.finditer(text):
            quote = match.group(1)
            page = int(match.group(2)) if match.group(2) else None
            
            # Try to find the quote in chunks
            for chunk in chunks:
                if quote.lower() in chunk.content.lower():
                    citation = Citation(
                        text=quote,
                        chapter=chunk.chapter_number,
                        page=page or chunk.page_number
                    )
                    citations.append(citation)
                    break
            else:
                # Quote not found in chunks, add anyway
                citations.append(Citation(text=quote, page=page))
                
        return citations
        
    def _calculate_emotion_intensities(self, emotions: List[EmotionTag],
                                     user_message: str,
                                     chunks: List[TextChunk]) -> Dict[str, float]:
        """
        Calculate intensity for each emotion based on context.
        
        Args:
            emotions: List of emotions
            user_message: User's message
            chunks: Text context
            
        Returns:
            Dictionary of emotion to intensity (0-1)
        """
        intensities = {}
        
        # Base intensity
        base_intensity = 0.5
        
        for emotion in emotions:
            intensity = base_intensity
            
            # Adjust based on user message sentiment
            if "love" in user_message.lower() or "amazing" in user_message.lower():
                if emotion in [EmotionTag.JOY, EmotionTag.PASSIONATE]:
                    intensity += 0.2
                    
            if "sad" in user_message.lower() or "tragic" in user_message.lower():
                if emotion in [EmotionTag.SADNESS, EmotionTag.MELANCHOLIC]:
                    intensity += 0.2
                    
            # Adjust based on chunk emotional tone
            for chunk in chunks[:2]:  # Top 2 chunks
                if chunk.emotional_tone:
                    if emotion.value in chunk.emotional_tone.lower():
                        intensity += 0.1
                        
            # Ensure within bounds
            intensities[emotion.value] = min(max(intensity, 0.1), 1.0)
            
        return intensities
        
    def _get_voice_modulation(self, emotions: List[EmotionTag],
                            intensities: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate voice modulation parameters based on emotions.
        
        Args:
            emotions: List of emotions
            intensities: Emotion intensities
            
        Returns:
            Voice modulation parameters
        """
        # Start with neutral settings
        modulation = {
            "pitch": 1.0,
            "speed": 1.0,
            "stability": 0.5
        }
        
        # Apply emotion-based modulations weighted by intensity
        total_weight = sum(intensities.get(e.value, 0.5) for e in emotions)
        
        for emotion in emotions:
            weight = intensities.get(emotion.value, 0.5) / total_weight
            emotion_settings = settings.EMOTION_VOICE_MAPPING.get(emotion.value, {})
            
            for param, value in emotion_settings.items():
                if param in modulation:
                    # Weighted average
                    modulation[param] = modulation[param] * (1 - weight) + value * weight
                    
        return modulation


# Node function for LangGraph
async def persona_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node function for the persona agent in LangGraph.
    
    Args:
        state: Graph state dictionary
        
    Returns:
        Updated state dictionary
    """
    # Convert dict state to AgentState
    agent_state = AgentState(**state)
    
    # Create Honcho client
    async with HonchoMemoryClient() as honcho_client:
        # Initialize Honcho app if needed
        await honcho_client.initialize_app()
        
        # Create or get user
        await honcho_client.create_or_get_user(agent_state.user_id)
        
        # Create persona agent
        agent = PersonaAgent(honcho_client=honcho_client)
        
        # Generate response
        updated_state = await agent.respond(agent_state)
        
    # Convert back to dict and update
    state.update(updated_state.dict())
    
    # Set agent output
    if "agent_outputs" not in state:
        state["agent_outputs"] = {}
        
    state["agent_outputs"]["persona"] = {
        "response": updated_state.generated_response,
        "emotions": updated_state.current_emotions,
        "emotional_trajectory": updated_state.get_emotional_trajectory()
    }
    
    return state