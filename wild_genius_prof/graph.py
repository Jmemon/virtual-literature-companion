"""
LangGraph Workflow for Wild Genius Professor

This module implements the LangGraph workflow that orchestrates the
professor's response generation, incorporating emotional processing,
literary analysis, and memory management into a cohesive system.

The graph manages the flow from user input through emotional processing,
content analysis, response generation, and state updates.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, TypedDict
import time

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import StateGraph, Graph
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain dependencies not available: {e}")
    # Define minimal stubs for development without LangChain
    BaseLanguageModel = None
    ChatOpenAI = None
    ChatAnthropic = None
    StateGraph = None
    HumanMessage = None
    AIMessage = None
    BaseMessage = None
    Graph = None
    LANGCHAIN_AVAILABLE = False

from .state import (
    EmotionalDimensions, 
    EmotionalProcessor, 
    ConversationContext,
    Message,
    create_initial_state
)
from .persona import ProfessorPersona
from .config import config

logger = logging.getLogger(__name__)


class ProfessorState(TypedDict):
    """
    State structure for the Wild Genius Professor LangGraph.
    
    This defines the complete state that flows through the graph nodes,
    including emotional state, conversation context, and response data.
    """
    
    # Core conversation
    messages: List[Any]  # BaseMessage when available, otherwise Any
    user_input: str
    professor_response: str
    
    # Emotional processing
    emotional_dimensions: EmotionalDimensions
    dominant_emotion: str
    emotional_intensity: float
    
    # Literary analysis
    content_analysis: Dict[str, Any]
    themes_identified: List[str]
    characters_mentioned: List[str]
    
    # Context and memory
    conversation_context: ConversationContext
    session_id: str
    user_id: str
    
    # Generation parameters
    response_temperature: float
    include_citations: bool
    use_emotional_prefixes: bool


class WildGeniusProfessorGraph:
    """
    LangGraph workflow for the Wild Genius Professor.
    
    This class orchestrates the complete response generation process,
    from user input through emotional processing, literary analysis,
    and final response generation with memory management.
    
    Workflow Nodes:
    1. process_input - Analyze user input and extract content
    2. update_emotions - Process emotional influences
    3. analyze_literature - Perform literary analysis
    4. generate_response - Create professor response
    5. update_memory - Store conversation state
    """
    
    def __init__(self, 
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4-turbo-preview",
                 honcho_api_key: Optional[str] = None):
        """
        Initialize the Wild Genius Professor graph.
        
        Args:
            llm_provider: LLM provider ("openai" or "anthropic")
            model_name: Specific model to use
            honcho_api_key: Optional API key for Honcho memory system
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_provider, model_name)
        
        # Initialize core components
        self.emotional_processor = EmotionalProcessor()
        self.professor_persona = ProfessorPersona(self.emotional_processor)
        
        # Memory management
        self.honcho_api_key = honcho_api_key
        self.memory_enabled = honcho_api_key is not None
        
        # Build the graph
        self.graph = self._build_graph()
        
        self.logger.info(f"🔄 Wild Genius Professor graph initialized with {llm_provider}")
    
    def _initialize_llm(self, provider: str, model_name: str) -> Optional[Any]:
        """
        Initialize the language model based on provider.
        
        Args:
            provider: LLM provider name
            model_name: Model identifier
            
        Returns:
            Initialized language model or None if unavailable
        """
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain not available, using mock LLM")
            return None
            
        try:
            if provider == "openai" and ChatOpenAI:
                return ChatOpenAI(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            elif provider == "anthropic" and ChatAnthropic:
                return ChatAnthropic(
                    model=model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _build_graph(self) -> Optional[Graph]:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled LangGraph or None if LangGraph unavailable
        """
        if not StateGraph:
            self.logger.warning("LangGraph not available, using direct processing")
            return None
            
        # Create the state graph
        workflow = StateGraph(ProfessorState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input_node)
        workflow.add_node("update_emotions", self._update_emotions_node)
        workflow.add_node("analyze_literature", self._analyze_literature_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("update_memory", self._update_memory_node)
        
        # Define the flow
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "update_emotions")
        workflow.add_edge("update_emotions", "analyze_literature")
        workflow.add_edge("analyze_literature", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.set_finish_point("update_memory")
        
        return workflow.compile()
    
    async def _process_input_node(self, state: ProfessorState) -> ProfessorState:
        """
        Process and analyze user input.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with input analysis
        """
        self.logger.debug("Processing user input")
        
        # Extract user input from messages
        if state["messages"]:
            latest_message = state["messages"][-1]
            if HumanMessage and hasattr(latest_message, 'content'):
                state["user_input"] = latest_message.content
        
        # Perform initial content processing
        content_processor = self.professor_persona.content_processor
        content_analysis = content_processor.process_user_input(state["user_input"])
        state["content_analysis"] = content_analysis
        
        # Extract key information for state
        state["themes_identified"] = [theme.name for theme in content_analysis["themes"]]
        state["characters_mentioned"] = content_analysis["characters"]
        
        return state
    
    async def _update_emotions_node(self, state: ProfessorState) -> ProfessorState:
        """
        Update emotional state based on interaction.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with new emotional dimensions
        """
        self.logger.debug("Updating emotional state")
        
        # Calculate interaction intensity
        intensity = self.professor_persona._calculate_interaction_intensity(
            state["content_analysis"]
        )
        state["emotional_intensity"] = intensity
        
        # Process emotional influences
        updated_emotions = self.emotional_processor.process_interaction(
            state["emotional_dimensions"],
            state["content_analysis"]["interaction_type"],
            state["user_input"],
            intensity
        )
        
        state["emotional_dimensions"] = updated_emotions
        state["dominant_emotion"] = updated_emotions.get_dominant_emotion().value
        
        return state
    
    async def _analyze_literature_node(self, state: ProfessorState) -> ProfessorState:
        """
        Perform literary analysis on the content.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with literary analysis
        """
        self.logger.debug("Performing literary analysis")
        
        # Literary analysis is already done in content_analysis
        # This node could be expanded for more sophisticated analysis
        
        # Update response parameters based on analysis
        if state["themes_identified"]:
            state["include_citations"] = True
            state["response_temperature"] = min(0.9, config.temperature + 0.1)
        else:
            state["include_citations"] = config.include_citations
            state["response_temperature"] = config.temperature
        
        state["use_emotional_prefixes"] = config.use_emotional_prefixes
        
        return state
    
    async def _generate_response_node(self, state: ProfessorState) -> ProfessorState:
        """
        Generate the professor's response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with generated response
        """
        self.logger.debug("Generating professor response")
        
        # Use persona to generate response
        response, _ = self.professor_persona.process_user_input(
            state["user_input"],
            state["emotional_dimensions"],
            {
                "conversation_context": state["conversation_context"],
                "themes": state["themes_identified"],
                "characters": state["characters_mentioned"]
            }
        )
        
        state["professor_response"] = response
        
        # Add response to messages
        if AIMessage:
            ai_message = AIMessage(content=response)
            state["messages"] = state["messages"] + [ai_message]
        
        return state
    
    async def _update_memory_node(self, state: ProfessorState) -> ProfessorState:
        """
        Update conversation memory and context.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with memory updates
        """
        self.logger.debug("Updating conversation memory")
        
        # Create message objects for memory
        user_message = Message(
            role="user",
            content=state["user_input"],
            emotions=[],  # Could extract user emotions here
            emotional_intensity=0.0
        )
        
        assistant_message = Message(
            role="assistant", 
            content=state["professor_response"],
            emotions=[state["emotional_dimensions"].get_dominant_emotion()],
            emotional_intensity=state["emotional_intensity"]
        )
        
        # Update conversation context
        from .state import update_conversation_context
        context = state["conversation_context"]
        context = update_conversation_context(context, user_message)
        context = update_conversation_context(context, assistant_message)
        state["conversation_context"] = context
        
        # TODO: Integrate with Honcho for persistent memory if enabled
        if self.memory_enabled:
            await self._store_in_honcho(state)
        
        return state
    
    async def _store_in_honcho(self, state: ProfessorState):
        """
        Store conversation state in Honcho for persistence.
        
        Args:
            state: Current state to store
        """
        # Placeholder for Honcho integration
        # In full implementation, this would store the conversation
        # state and emotional journey in Honcho for cross-session memory
        self.logger.debug("Storing state in Honcho (placeholder)")
    
    def run(self, 
            user_message: str,
            user_id: str,
            session_id: str,
            existing_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the professor graph synchronously.
        
        Args:
            user_message: The user's input message
            user_id: User identifier
            session_id: Session identifier
            existing_state: Optional existing state to continue from
            
        Returns:
            Dictionary containing the response and updated state
        """
        return asyncio.run(self.arun(user_message, user_id, session_id, existing_state))
    
    async def arun(self,
                   user_message: str, 
                   user_id: str,
                   session_id: str,
                   existing_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the professor graph asynchronously.
        
        Args:
            user_message: The user's input message
            user_id: User identifier  
            session_id: Session identifier
            existing_state: Optional existing state to continue from
            
        Returns:
            Dictionary containing the response and updated state
        """
        self.logger.debug(f"Running graph for user {user_id}, session {session_id}")
        
        try:
            # Initialize state
            if existing_state:
                initial_state = existing_state
            else:
                initial_state = create_initial_state()
                initial_state.update({
                    "user_id": user_id,
                    "session_id": session_id
                })
            
            # Convert to ProfessorState format
            if HumanMessage:
                human_msg = HumanMessage(content=user_message)
                messages = initial_state.get("messages", []) + [human_msg]
            else:
                messages = []
            
            state = ProfessorState(
                messages=messages,
                user_input=user_message,
                professor_response="",
                emotional_dimensions=initial_state.get("emotional_dimensions", EmotionalDimensions()),
                dominant_emotion="contemplation",
                emotional_intensity=0.0,
                content_analysis={},
                themes_identified=[],
                characters_mentioned=[],
                conversation_context=initial_state.get("conversation_context", ConversationContext()),
                session_id=session_id,
                user_id=user_id,
                response_temperature=config.temperature,
                include_citations=config.include_citations,
                use_emotional_prefixes=config.use_emotional_prefixes
            )
            
            # Run the graph if available, otherwise direct processing
            if self.graph:
                result_state = await self.graph.ainvoke(state)
            else:
                # Direct processing without LangGraph
                result_state = await self._direct_process(state)
            
            # Return response and state
            return {
                "response": result_state["professor_response"],
                "emotional_state": result_state["dominant_emotion"],
                "emotional_intensity": result_state["emotional_intensity"],
                "themes": result_state["themes_identified"],
                "characters": result_state["characters_mentioned"],
                "state": result_state
            }
            
        except Exception as e:
            self.logger.error(f"Error running graph: {e}")
            return {
                "response": f"[TURMOIL] I seem to be experiencing some internal confusion... Perhaps we could try rephrasing that thought? 😵",
                "emotional_state": "turmoil", 
                "emotional_intensity": 0.8,
                "themes": [],
                "characters": [],
                "state": {}
            }
    
    async def _direct_process(self, state: ProfessorState) -> ProfessorState:
        """
        Process directly without LangGraph (fallback method).
        
        Args:
            state: Input state
            
        Returns:
            Processed state
        """
        self.logger.debug("Using direct processing (no LangGraph)")
        
        # Sequential processing through the nodes
        state = await self._process_input_node(state)
        state = await self._update_emotions_node(state)
        state = await self._analyze_literature_node(state)
        state = await self._generate_response_node(state)
        state = await self._update_memory_node(state)
        
        return state