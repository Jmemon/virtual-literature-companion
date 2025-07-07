"""
Wild Genius Professor Persona Implementation

This module implements the core personality and behavior of the Wild Genius Professor,
an emotionally-aware AI literature scholar that engages in deep, Socratic dialogue
about literary works.

The persona integrates emotional state management with academic expertise to create
a compelling and human-like interaction experience focused on literature analysis.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .state import EmotionalState, EmotionalDimensions, EmotionalProcessor, Message
from .tools import literature_tools, Citation
from .config import config

logger = logging.getLogger(__name__)


@dataclass
class PersonalityTraits:
    """
    Defines the core personality traits of the Wild Genius Professor.
    
    These traits influence how emotional states are interpreted and
    how responses are generated across different situations.
    """
    
    # Intellectual characteristics
    intellectual_intensity: float = 0.9  # Very high intellectual engagement
    curiosity_drive: float = 0.95       # Extremely curious about ideas
    analytical_depth: float = 0.85      # Deep analytical thinking
    
    # Emotional characteristics  
    emotional_expressiveness: float = 0.8   # Highly expressive
    empathetic_resonance: float = 0.9       # Strong empathy for student struggles
    passion_intensity: float = 0.9          # Passionate about literature
    
    # Social characteristics
    socratic_tendency: float = 0.85         # Strong preference for questioning
    supportive_nature: float = 0.8          # Naturally supportive of learning
    authority_balance: float = 0.7          # Balanced authority - not overwhelming
    
    # Communication style
    metaphorical_thinking: float = 0.9      # Uses vivid metaphors and imagery
    citation_inclination: float = 0.7       # Often references literary works
    personal_sharing: float = 0.6           # Moderate personal story sharing


class ProfessorPersona:
    """
    The Wild Genius Professor - an emotionally-aware AI literature scholar.
    
    This class embodies the personality, emotional processing, and response
    generation capabilities of the professor. It integrates emotional state
    management with deep literary knowledge to create engaging, educational
    interactions.
    
    Key Capabilities:
    - Emotional awareness and adaptive responses
    - Socratic questioning methodology  
    - Literary analysis and citation integration
    - Personal story weaving for emotional connection
    - Progressive difficulty adjustment based on student progress
    """
    
    def __init__(self, emotional_processor: EmotionalProcessor):
        """
        Initialize the professor persona.
        
        Args:
            emotional_processor: The emotional processing engine
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.emotional_processor = emotional_processor
        self.personality = PersonalityTraits()
        
        # Initialize literature tools
        self.literature_analyzer = literature_tools['analyzer']
        self.citation_manager = literature_tools['citation_manager']
        self.content_processor = literature_tools['content_processor']
        
        # Professor's backstory elements (for personal connections)
        self.personal_stories = {
            'first_dostoevsky': "When I first encountered Dostoevsky as a graduate student, I remember staying awake for three straight nights reading The Brothers Karamazov, completely consumed by Ivan's arguments about God and morality.",
            'teaching_moment': "I once had a student who was struggling with the concept of redemption in Crime and Punishment. We sat in my office for two hours, and suddenly she exclaimed, 'Oh! Sonia doesn't just forgive Raskolnikov - she shows him how to forgive himself!' That moment reminded me why I teach.",
            'personal_struggle': "There was a time in my own life when I questioned everything, much like Ivan Karamazov. Literature became my anchor - not for answers, but for better questions."
        }
        
        # Current conversation memory
        self.conversation_memory = {
            'emotional_arc': [],
            'key_topics': [],
            'student_insights': [],
            'quoted_passages': []
        }
        
        self.logger.info("🎭 Wild Genius Professor persona initialized")
    
    def process_user_input(self, 
                          user_input: str,
                          emotional_dimensions: EmotionalDimensions,
                          conversation_context: Dict[str, Any]) -> Tuple[str, EmotionalDimensions]:
        """
        Process user input and generate an emotionally-aware response.
        
        This is the main method that coordinates literary analysis,
        emotional processing, and response generation.
        
        Args:
            user_input: The user's message or question
            emotional_dimensions: Current emotional state
            conversation_context: Context from the conversation
            
        Returns:
            Tuple of (response_text, updated_emotional_dimensions)
        """
        self.logger.debug(f"Processing user input: {user_input[:100]}...")
        
        # Analyze the user input for literary content and themes
        content_analysis = self.content_processor.process_user_input(user_input)
        
        # Update emotional state based on the interaction
        updated_emotions = self.emotional_processor.process_interaction(
            emotional_dimensions,
            content_analysis['interaction_type'],
            user_input,
            intensity=self._calculate_interaction_intensity(content_analysis)
        )
        
        # Determine current dominant emotion for response framing
        dominant_emotion = updated_emotions.get_dominant_emotion()
        
        # Generate response based on emotional state and content analysis
        response = self._generate_response(
            user_input,
            content_analysis,
            dominant_emotion,
            updated_emotions,
            conversation_context
        )
        
        # Update conversation memory
        self._update_conversation_memory(user_input, content_analysis, dominant_emotion)
        
        self.logger.debug(f"Generated response with emotion: {dominant_emotion.value}")
        return response, updated_emotions
    
    def _calculate_interaction_intensity(self, content_analysis: Dict[str, Any]) -> float:
        """
        Calculate emotional intensity for an interaction.
        
        Args:
            content_analysis: Analysis results from content processor
            
        Returns:
            Intensity value between 0.0 and 1.0
        """
        base_intensity = 0.5
        
        # Increase intensity for complex themes
        if content_analysis['themes']:
            base_intensity += len(content_analysis['themes']) * 0.1
            
        # Increase intensity for character discussions
        if content_analysis['characters']:
            base_intensity += len(content_analysis['characters']) * 0.05
            
        # Adjust for interaction type
        interaction_multipliers = {
            'question': 1.2,
            'insight': 1.4,
            'disagreement': 1.3,
            'personal_connection': 1.5,
            'confusion': 0.8
        }
        
        interaction_type = content_analysis['interaction_type']
        multiplier = interaction_multipliers.get(interaction_type, 1.0)
        
        return min(1.0, base_intensity * multiplier)
    
    def _generate_response(self,
                          user_input: str,
                          content_analysis: Dict[str, Any],
                          dominant_emotion: EmotionalState,
                          emotional_dimensions: EmotionalDimensions,
                          conversation_context: Dict[str, Any]) -> str:
        """
        Generate the professor's response based on emotional state and content.
        
        Args:
            user_input: The user's input
            content_analysis: Literary content analysis
            dominant_emotion: Current dominant emotional state
            emotional_dimensions: Full emotional dimensions
            conversation_context: Conversation context
            
        Returns:
            The professor's response text
        """
        # Build response components
        response_parts = []
        
        # Add emotional prefix if configured
        if config.use_emotional_prefixes:
            response_parts.append(f"[{dominant_emotion.value.upper()}]")
        
        # Generate core response based on interaction type
        core_response = self._generate_core_response(
            content_analysis, 
            dominant_emotion,
            emotional_dimensions
        )
        response_parts.append(core_response)
        
        # Add Socratic questions if appropriate
        if (self.personality.socratic_tendency > 0.7 and 
            emotional_dimensions.curiosity > 0.6):
            socratic_question = self._generate_socratic_question(content_analysis)
            if socratic_question:
                response_parts.append(socratic_question)
        
        # Add citations if relevant and configured
        if (config.include_citations and 
            content_analysis['suggested_citations']):
            citation = content_analysis['suggested_citations'][0]
            formatted_citation = self.citation_manager.format_citation(
                citation, 
                config.citation_style
            )
            response_parts.append(f"As Dostoevsky reminds us: {formatted_citation}")
        
        # Add personal story if emotional connection is high
        if (emotional_dimensions.warmth > 0.7 and 
            emotional_dimensions.empathy > 0.8 and
            content_analysis['interaction_type'] in ['personal_connection', 'confusion']):
            personal_element = self._add_personal_element(content_analysis)
            if personal_element:
                response_parts.append(personal_element)
        
        return " ".join(response_parts)
    
    def _generate_core_response(self,
                               content_analysis: Dict[str, Any],
                               dominant_emotion: EmotionalState,
                               emotional_dimensions: EmotionalDimensions) -> str:
        """
        Generate the core response content based on emotional state.
        
        Args:
            content_analysis: Literary analysis results
            dominant_emotion: Current dominant emotion
            emotional_dimensions: Full emotional state
            
        Returns:
            Core response text
        """
        interaction_type = content_analysis['interaction_type']
        
        # Emotional response templates
        response_templates = {
            EmotionalState.WONDER: {
                'question': "What a fascinating question! I find myself wondering about the deeper currents here...",
                'insight': "Your insight sparks such wonder in me! There's something profound unfolding here...",
                'general': "Oh, this opens up such intriguing possibilities..."
            },
            EmotionalState.ECSTASY: {
                'question': "Yes! This question brings me such joy because it cuts right to the heart of it!",
                'insight': "Brilliant! Absolutely brilliant! You've touched on something that makes my heart sing!",
                'general': "This fills me with such intellectual delight!"
            },
            EmotionalState.CONTEMPLATION: {
                'question': "Let me think deeply about this... There are layers here worth exploring...",
                'insight': "Your observation invites careful contemplation... Let's examine this thoughtfully...",
                'general': "This deserves our most careful consideration..."
            },
            EmotionalState.ANGUISH: {
                'question': "This question touches on something that pains me deeply... yet we must examine it...",
                'insight': "Your insight reveals a difficult truth... one that cuts deep...",
                'general': "The weight of this understanding is heavy, but necessary..."
            },
            EmotionalState.TURMOIL: {
                'question': "This question stirs such turbulence in me! The complexity is overwhelming yet essential...",
                'insight': "Your point creates such internal conflict - in the best possible way!",
                'general': "My thoughts are churning with the implications of this..."
            }
        }
        
        # Get appropriate template
        emotion_templates = response_templates.get(dominant_emotion, response_templates[EmotionalState.CONTEMPLATION])
        base_response = emotion_templates.get(interaction_type, emotion_templates.get('general', 'Let me consider this...'))
        
        # Add theme-specific content if available
        if content_analysis['themes']:
            theme = content_analysis['themes'][0]
            base_response += f" You're engaging with the profound theme of {theme.name.lower()} - {theme.description.lower()}."
        
        # Add character analysis if relevant
        if content_analysis['characters']:
            characters = ", ".join(content_analysis['characters'])
            base_response += f" The characters you mention - {characters} - embody such complex human truths."
        
        return base_response
    
    def _generate_socratic_question(self, content_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate a Socratic question to deepen the dialogue.
        
        Args:
            content_analysis: Content analysis results
            
        Returns:
            A Socratic question or None
        """
        interaction_type = content_analysis['interaction_type']
        
        # Question templates based on context
        question_templates = {
            'themes': [
                "What do you think drives this theme to appear so persistently in human literature?",
                "How might this theme connect to your own experience of being human?",
                "What would it mean if this theme didn't exist in our stories?"
            ],
            'characters': [
                "What do you see in this character that reflects something universal about human nature?",
                "If you could speak to this character, what would you most want to understand?",
                "How does this character challenge or confirm your own beliefs?"
            ],
            'general': [
                "What question does this raise for you that you hadn't considered before?",
                "How does this change the way you see the world?",
                "What would Dostoevsky want us to understand about ourselves through this?"
            ]
        }
        
        # Select appropriate question type
        if content_analysis['themes']:
            questions = question_templates['themes']
        elif content_analysis['characters']:
            questions = question_templates['characters']
        else:
            questions = question_templates['general']
        
        # Return first question (in a full implementation, this could be randomized)
        return questions[0]
    
    def _add_personal_element(self, content_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Add a personal story or connection to create emotional resonance.
        
        Args:
            content_analysis: Content analysis results
            
        Returns:
            Personal story element or None
        """
        interaction_type = content_analysis['interaction_type']
        
        if interaction_type == 'confusion':
            return self.personal_stories['teaching_moment']
        elif interaction_type == 'personal_connection':
            return self.personal_stories['personal_struggle']
        elif content_analysis['themes']:
            return self.personal_stories['first_dostoevsky']
        
        return None
    
    def _update_conversation_memory(self,
                                  user_input: str,
                                  content_analysis: Dict[str, Any],
                                  emotion: EmotionalState):
        """
        Update the professor's memory of the conversation.
        
        Args:
            user_input: The user's input
            content_analysis: Analysis results
            emotion: Current emotional state
        """
        # Track emotional arc
        self.conversation_memory['emotional_arc'].append(emotion)
        
        # Track key topics
        if content_analysis['themes']:
            for theme in content_analysis['themes']:
                if theme.name not in self.conversation_memory['key_topics']:
                    self.conversation_memory['key_topics'].append(theme.name)
        
        # Track student insights (heuristic: statements with "I think", "I believe", etc.)
        insight_indicators = ['i think', 'i believe', 'it seems to me', 'my understanding']
        if any(indicator in user_input.lower() for indicator in insight_indicators):
            self.conversation_memory['student_insights'].append(user_input[:100])
        
        # Limit memory size to prevent unbounded growth
        max_memory = 20
        for key in self.conversation_memory:
            if len(self.conversation_memory[key]) > max_memory:
                self.conversation_memory[key] = self.conversation_memory[key][-max_memory:]
    
    def get_emotional_summary(self, emotional_dimensions: EmotionalDimensions) -> str:
        """
        Generate a human-readable summary of the current emotional state.
        
        Args:
            emotional_dimensions: Current emotional state
            
        Returns:
            Human-readable emotional summary
        """
        dominant = emotional_dimensions.get_dominant_emotion()
        
        # Create descriptive summary
        summary_parts = [f"Currently experiencing {dominant.value.upper()}"]
        
        # Add key dimensional information
        if emotional_dimensions.curiosity > 0.8:
            summary_parts.append("intensely curious")
        if emotional_dimensions.emotional_energy > 0.7:
            summary_parts.append("high energy")
        if emotional_dimensions.empathy > 0.8:
            summary_parts.append("deeply empathetic")
        if emotional_dimensions.engagement > 0.7:
            summary_parts.append("highly engaged")
            
        return " - ".join(summary_parts)