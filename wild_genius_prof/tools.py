"""
Literature tools and utilities for Wild Genius Professor.

This module provides specialized tools for literary analysis,
citation management, and content processing that enhance the
professor's ability to engage with literary texts.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a literary citation with context."""
    book: str
    chapter: Optional[int] = None
    page: Optional[int] = None
    quote: Optional[str] = None
    context: Optional[str] = None
    emotional_relevance: Optional[str] = None


@dataclass
class LiteraryTheme:
    """Represents a literary theme with associated content."""
    name: str
    description: str
    key_passages: List[str]
    emotional_associations: List[str]
    related_characters: List[str]


class LiteratureAnalyzer:
    """
    Analyzes literary content for themes, patterns, and emotional triggers.
    
    This tool helps the professor identify relevant literary elements
    and connect them to emotional states and user interests.
    """
    
    def __init__(self):
        """Initialize with knowledge base of literary themes and patterns."""
        
        # Dostoevsky-specific themes and patterns
        self.dostoevsky_themes = {
            'suffering_and_redemption': LiteraryTheme(
                name="Suffering and Redemption",
                description="The transformative power of suffering leading to spiritual awakening",
                key_passages=[
                    "Pain and suffering are always inevitable for a large intelligence and a deep heart",
                    "Can a man of perception respect himself at all?"
                ],
                emotional_associations=["anguish", "contemplation", "wonder"],
                related_characters=["Raskolnikov", "Dmitri", "Ivan", "Alyosha"]
            ),
            'faith_vs_reason': LiteraryTheme(
                name="Faith vs. Reason", 
                description="The eternal struggle between rational thought and spiritual faith",
                key_passages=[
                    "If God does not exist, everything is permitted",
                    "I want to suffer so that I may love"
                ],
                emotional_associations=["turmoil", "contemplation", "anguish"],
                related_characters=["Ivan", "Alyosha", "Zosima"]
            ),
            'brotherhood_and_isolation': LiteraryTheme(
                name="Brotherhood and Isolation",
                description="The human need for connection versus the reality of spiritual isolation",
                key_passages=[
                    "We are all responsible for all",
                    "Hell is the suffering of being unable to love"
                ],
                emotional_associations=["melancholy", "warmth", "empathy"],
                related_characters=["Alyosha", "Dmitri", "Grushenka"]
            )
        }
        
        # Character emotional profiles
        self.character_emotions = {
            'Raskolnikov': ['anguish', 'turmoil', 'contempt', 'isolation'],
            'Dmitri': ['passion', 'turmoil', 'joy', 'despair'],
            'Ivan': ['intellectual_torment', 'doubt', 'anguish', 'pride'],
            'Alyosha': ['serenity', 'compassion', 'wonder', 'faith'],
            'Zosima': ['wisdom', 'serenity', 'love', 'acceptance']
        }
    
    def analyze_content_themes(self, content: str) -> List[LiteraryTheme]:
        """
        Analyze content for literary themes.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of relevant LiteraryTheme objects
        """
        relevant_themes = []
        content_lower = content.lower()
        
        for theme_key, theme in self.dostoevsky_themes.items():
            # Check for theme-related keywords
            theme_keywords = theme.description.lower().split() + [theme.name.lower()]
            
            if any(keyword in content_lower for keyword in theme_keywords):
                relevant_themes.append(theme)
                
        return relevant_themes
    
    def extract_character_references(self, content: str) -> List[str]:
        """
        Extract character names mentioned in content.
        
        Args:
            content: Text to analyze
            
        Returns:
            List of character names found
        """
        characters = []
        
        for character in self.character_emotions.keys():
            if character.lower() in content.lower():
                characters.append(character)
                
        return characters
    
    def suggest_emotional_response(self, themes: List[LiteraryTheme], 
                                 characters: List[str]) -> Dict[str, float]:
        """
        Suggest emotional influences based on literary analysis.
        
        Args:
            themes: Relevant literary themes
            characters: Referenced characters
            
        Returns:
            Dictionary of suggested emotional influences
        """
        influences = {}
        
        # Aggregate influences from themes
        for theme in themes:
            for emotion in theme.emotional_associations:
                influences[emotion] = influences.get(emotion, 0) + 0.3
                
        # Add character-based influences
        for character in characters:
            char_emotions = self.character_emotions.get(character, [])
            for emotion in char_emotions:
                influences[emotion] = influences.get(emotion, 0) + 0.2
                
        # Normalize influences to [0, 1] range
        if influences:
            max_influence = max(influences.values())
            if max_influence > 1.0:
                influences = {k: v/max_influence for k, v in influences.items()}
                
        return influences


class CitationManager:
    """
    Manages literary citations and references.
    
    Provides tools for creating, formatting, and retrieving
    contextually relevant citations from literary works.
    """
    
    def __init__(self):
        """Initialize with citation database."""
        
        # Sample citations from The Brothers Karamazov
        self.citations_db = [
            Citation(
                book="The Brothers Karamazov",
                chapter=1,
                quote="Above all, don't lie to yourself.",
                context="Zosima's advice about self-honesty",
                emotional_relevance="contemplation_truth"
            ),
            Citation(
                book="The Brothers Karamazov", 
                chapter=5,
                quote="If God does not exist, everything is permitted.",
                context="Ivan's philosophical argument",
                emotional_relevance="turmoil_doubt"
            ),
            Citation(
                book="The Brothers Karamazov",
                chapter=11,
                quote="Love all God's creation, both the whole and every grain of sand.",
                context="Zosima's teaching on universal love",
                emotional_relevance="wonder_love"
            )
        ]
    
    def find_relevant_citations(self, emotional_state: str, 
                              theme: Optional[str] = None) -> List[Citation]:
        """
        Find citations relevant to current emotional state and theme.
        
        Args:
            emotional_state: Current emotional state
            theme: Optional thematic focus
            
        Returns:
            List of relevant Citation objects
        """
        relevant = []
        
        for citation in self.citations_db:
            if citation.emotional_relevance and emotional_state in citation.emotional_relevance:
                relevant.append(citation)
            elif theme and citation.context and theme.lower() in citation.context.lower():
                relevant.append(citation)
                
        return relevant
    
    def format_citation(self, citation: Citation, style: str = "academic") -> str:
        """
        Format a citation in the specified style.
        
        Args:
            citation: Citation to format
            style: Citation style ("academic", "conversational", "minimal")
            
        Returns:
            Formatted citation string
        """
        if style == "academic":
            return f'"{citation.quote}" (Dostoevsky, {citation.book}, Chapter {citation.chapter})'
        elif style == "conversational":
            return f'As Dostoevsky writes, "{citation.quote}"'
        else:  # minimal
            return f'"{citation.quote}"'


class ContentProcessor:
    """
    Processes and enriches content with literary analysis.
    
    This tool helps the professor understand user input in
    literary context and generate appropriate responses.
    """
    
    def __init__(self):
        """Initialize content processor with analyzers."""
        self.analyzer = LiteratureAnalyzer()
        self.citation_manager = CitationManager()
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of user input.
        
        Args:
            user_input: The user's message or question
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'themes': self.analyzer.analyze_content_themes(user_input),
            'characters': self.analyzer.extract_character_references(user_input),
            'emotional_triggers': {},
            'suggested_citations': [],
            'interaction_type': self._classify_interaction(user_input)
        }
        
        # Generate emotional suggestions
        analysis['emotional_triggers'] = self.analyzer.suggest_emotional_response(
            analysis['themes'], 
            analysis['characters']
        )
        
        # Find relevant citations
        if analysis['themes']:
            theme_name = analysis['themes'][0].name
            analysis['suggested_citations'] = self.citation_manager.find_relevant_citations(
                'contemplation',  # Default emotional state for citation search
                theme_name
            )
        
        return analysis
    
    def _classify_interaction(self, text: str) -> str:
        """
        Classify the type of interaction based on text content.
        
        Args:
            text: Input text to classify
            
        Returns:
            Interaction type string
        """
        text_lower = text.lower()
        
        if '?' in text:
            return 'question'
        elif any(word in text_lower for word in ['i think', 'i believe', 'it seems']):
            return 'insight'
        elif any(word in text_lower for word in ['confused', 'don\'t understand', 'unclear']):
            return 'confusion'
        elif any(word in text_lower for word in ['disagree', 'however', 'but', 'wrong']):
            return 'disagreement'
        elif any(word in text_lower for word in ['quote', 'says', 'writes', 'page']):
            return 'citation'
        elif any(word in text_lower for word in ['feel', 'reminds me', 'my experience']):
            return 'personal_connection'
        else:
            return 'general_discussion'


# Global tools instance
literature_tools = {
    'analyzer': LiteratureAnalyzer(),
    'citation_manager': CitationManager(), 
    'content_processor': ContentProcessor()
}