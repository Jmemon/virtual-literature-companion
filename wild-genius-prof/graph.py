"""
LangGraph implementation of the Wild Genius Professor agent.
Orchestrates the conversation flow with emotional awareness and pedagogical intent.
"""

from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os

from .state import (
    AgentState, 
    EmotionalState, 
    Message,
    Citation,
    extract_emotions_from_text,
    calculate_emotional_intensity,
    update_conversation_context
)
from .tools import (
    search_citations,
    analyze_user_emotion,
    check_spoiler_safety,
    generate_image_prompt,
    extract_themes,
    format_socratic_question,
    plan_ui_update
)
from .persona import WildGeniusProfessorPersona


class WildGeniusProfessorGraph:
    """Main graph implementation for the Wild Genius Professor."""
    
    def __init__(
        self, 
        llm_provider: Literal["openai", "anthropic"] = "openai",
        model_name: Optional[str] = None,
        honcho_api_key: Optional[str] = None
    ):
        """Initialize the professor graph."""
        # Set up LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name or "gpt-4-turbo-preview",
                temperature=0.8,  # Higher for more emotional variation
                streaming=True
            )
        else:
            self.llm = ChatAnthropic(
                model=model_name or "claude-3-opus-20240229",
                temperature=0.8,
                streaming=True
            )
            
        # Set up persona management
        self.persona = WildGeniusProfessorPersona(api_key=honcho_api_key)
        
        # Set up tools
        self.tools = [
            search_citations,
            analyze_user_emotion,
            check_spoiler_safety,
            generate_image_prompt,
            extract_themes,
            format_socratic_question,
            plan_ui_update
        ]
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        prompts = {}
        prompt_files = ["system", "emotional_tags", "socratic", "citation"]
        
        for filename in prompt_files:
            try:
                with open(f"wild-genius-prof/prompts/{filename}.txt", "r") as f:
                    prompts[filename] = f.read()
            except Exception as e:
                print(f"Error loading prompt {filename}: {e}")
                prompts[filename] = ""
                
        return prompts
        
    def _build_graph(self) -> StateGraph:
        """Build the conversation flow graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("check_progress", self.check_reading_progress)
        graph.add_node("analyze_input", self.analyze_user_input)
        graph.add_node("generate_response", self.generate_professor_response)
        graph.add_node("enhance_response", self.enhance_with_multimedia)
        graph.add_node("save_to_memory", self.save_conversation_memory)
        
        # Add edges
        graph.set_entry_point("check_progress")
        
        graph.add_edge("check_progress", "analyze_input")
        graph.add_edge("analyze_input", "generate_response")
        graph.add_edge("generate_response", "enhance_response")
        graph.add_edge("enhance_response", "save_to_memory")
        
        # Conditional edges
        graph.add_conditional_edges(
            "save_to_memory",
            self.should_continue,
            {
                "continue": "check_progress",
                "end": END
            }
        )
        
        return graph.compile()
        
    def check_reading_progress(self, state: AgentState) -> AgentState:
        """Check if we need to update reading progress."""
        messages = state["messages"]
        
        if not messages:
            # First message - ask about progress
            state["next_action"] = "clarify_progress"
            return state
            
        # Check if user mentioned a new chapter/book
        last_message = messages[-1]
        if last_message.role == "user":
            # Simple pattern matching for chapter mentions
            import re
            book_pattern = r"book\s+(\d+)"
            chapter_pattern = r"chapter\s+(\d+)"
            
            book_match = re.search(book_pattern, last_message.content.lower())
            chapter_match = re.search(chapter_pattern, last_message.content.lower())
            
            if book_match or chapter_match:
                if book_match:
                    state["reading_progress"].current_book = int(book_match.group(1))
                if chapter_match:
                    state["reading_progress"].current_chapter = int(chapter_match.group(1))
                    
        return state
        
    def analyze_user_input(self, state: AgentState) -> AgentState:
        """Analyze user's message for emotional tone and intent."""
        if not state["messages"]:
            return state
            
        last_message = state["messages"][-1]
        if last_message.role != "user":
            return state
            
        # Analyze emotion
        emotion_analysis = analyze_user_emotion.invoke({
            "text": last_message.content,
            "context": str(state["conversation_context"])
        })
        
        # Extract themes
        themes = extract_themes.invoke({"text": last_message.content})
        if themes:
            state["conversation_context"].themes_discussed.extend(themes)
            
        # Get personalized emotional response from Honcho
        suggested_emotion = self.persona.calibrate_emotional_response(
            state["user_id"], 
            last_message.content
        )
        
        # Update state
        state["current_emotion"] = suggested_emotion
        state["emotional_intensity"] = emotion_analysis.get("intensity", 0.5)
        
        return state
        
    def generate_professor_response(self, state: AgentState) -> AgentState:
        """Generate the professor's response with appropriate emotion and pedagogy."""
        # Build conversation history for LLM
        messages = self._build_message_history(state)
        
        # Add system prompt with current emotional state
        system_prompt = self._build_system_prompt(state)
        messages.insert(0, SystemMessage(content=system_prompt))
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Extract emotions from response
        emotions = extract_emotions_from_text(response.content)
        if emotions:
            state["current_emotion"] = emotions[0]  # Primary emotion
            
        # Calculate emotional intensity
        intensity = calculate_emotional_intensity(emotions, response.content)
        state["emotional_intensity"] = intensity
        
        # Check for citations needed
        if "Book" in response.content or "Chapter" in response.content:
            # Search for relevant citations
            citations = search_citations.invoke({
                "query": response.content,
                "reading_progress": state["reading_progress"].dict()
            })
            state["pending_citations"] = citations
            
        # Create message object
        professor_message = Message(
            role="assistant",
            content=response.content,
            emotions=emotions,
            citations=state["pending_citations"]
        )
        
        state["messages"].append(professor_message)
        state["conversation_context"] = update_conversation_context(
            state["conversation_context"],
            professor_message
        )
        
        return state
        
    def enhance_with_multimedia(self, state: AgentState) -> AgentState:
        """Add multimedia enhancements based on emotional state."""
        current_emotion = state["current_emotion"]
        intensity = state["emotional_intensity"]
        
        # Plan UI updates
        ui_update = plan_ui_update.invoke({
            "emotion": current_emotion.value,
            "intensity": intensity,
            "action": "transition"
        })
        state["ui_updates"] = ui_update
        
        # Determine if we should generate an image
        if intensity > 0.7 or len(state["messages"]) % 5 == 0:
            # Generate image for high intensity or every 5 messages
            themes = state["conversation_context"].themes_discussed
            if themes:
                image_prompt = generate_image_prompt.invoke({
                    "theme": themes[-1],  # Most recent theme
                    "emotion": current_emotion.value,
                    "style_notes": "Russian literary realism, Dostoyevsky atmosphere"
                })
                state["image_prompt"] = image_prompt["prompt"]
                
        # Set music directive
        state["music_directive"] = f"{current_emotion.value}:{intensity}"
        
        return state
        
    def save_conversation_memory(self, state: AgentState) -> AgentState:
        """Save conversation to Honcho for long-term memory."""
        if not state["honcho_session_id"]:
            # Create session if needed
            session = self.persona.create_session(state["user_id"])
            state["honcho_session_id"] = session.id
            
        # Save the latest message
        if state["messages"]:
            last_message = state["messages"][-1]
            
            metadata = {
                "themes": state["conversation_context"].themes_discussed[-5:],  # Last 5 themes
                "emotional_intensity": state["emotional_intensity"],
                "citations": [c.dict() for c in state["pending_citations"]]
            }
            
            self.persona.add_message(
                session_id=state["honcho_session_id"],
                role=last_message.role,
                content=last_message.content,
                emotions=last_message.emotions,
                metadata=metadata
            )
            
        # Check for breakthrough moments
        if state["emotional_intensity"] > 0.8 and state["current_emotion"] in [
            EmotionalState.ECSTASY, 
            EmotionalState.WONDER,
            EmotionalState.RAPTURE
        ]:
            # This might be a breakthrough moment
            last_message = state["messages"][-1]
            if last_message.role == "user":
                self.persona.save_breakthrough_moment(
                    user_id=state["user_id"],
                    session_id=state["honcho_session_id"],
                    insight=last_message.content,
                    context={
                        "emotion": state["current_emotion"].value,
                        "themes": state["conversation_context"].themes_discussed[-3:]
                    }
                )
                
        return state
        
    def should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine whether to continue the conversation."""
        if state["should_end"] or state["error"]:
            return "end"
            
        # Check if conversation is getting too long
        if len(state["messages"]) > 100:
            return "end"
            
        return "continue"
        
    def _build_message_history(self, state: AgentState) -> List[Any]:
        """Convert state messages to LangChain messages."""
        lc_messages = []
        
        for msg in state["messages"]:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
                
        return lc_messages
        
    def _build_system_prompt(self, state: AgentState) -> str:
        """Build system prompt with current context."""
        base_prompt = self.prompts["system"]
        emotion_prompt = self.prompts["emotional_tags"]
        
        # Add current state information
        context_additions = f"""
        
CURRENT STATE:
- User is at Book {state["reading_progress"].current_book}, Chapter {state["reading_progress"].current_chapter}
- Current emotional tone: {state["current_emotion"].value}
- Themes being explored: {', '.join(state["conversation_context"].themes_discussed[-3:])}
- User ID: {state["user_id"]}

EMOTIONAL GUIDANCE:
{emotion_prompt}

Remember: Never discuss content beyond their current reading progress!
        """
        
        # Get personalized insights from Honcho
        if state["user_id"] and len(state["messages"]) > 2:
            try:
                learning_style = self.persona.get_user_insights(
                    state["user_id"],
                    "What's the most effective way to engage this user in literary discussion?"
                )
                context_additions += f"\n\nPERSONALIZED APPROACH:\n{learning_style}"
            except:
                pass  # Continue without personalization if Honcho fails
                
        return base_prompt + context_additions
        
    def run(self, user_message: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """Run a single conversation turn."""
        # Create or get state
        state = self._get_or_create_state(user_id, session_id)
        
        # Add user message
        user_msg = Message(
            role="user",
            content=user_message
        )
        state["messages"].append(user_msg)
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Extract response
        response_message = result["messages"][-1] if result["messages"] else None
        
        return {
            "response": response_message.content if response_message else "",
            "emotions": [e.value for e in response_message.emotions] if response_message else [],
            "ui_updates": result.get("ui_updates", {}),
            "image_prompt": result.get("image_prompt"),
            "music_directive": result.get("music_directive"),
            "citations": result.get("pending_citations", [])
        }
        
    def _get_or_create_state(self, user_id: str, session_id: str) -> AgentState:
        """Get existing state or create new one."""
        # In production, this would retrieve from a state store
        # For now, create fresh state
        from .state import create_initial_state
        return create_initial_state(user_id, session_id)