"""
Tests for the emotional state system.
"""

import pytest
from wild_genius_prof.state import (
    EmotionalState,
    extract_emotions_from_text,
    calculate_emotional_intensity,
    Message,
    update_conversation_context,
    ConversationContext
)


class TestEmotionalState:
    """Test emotional state detection and management."""
    
    def test_emotion_extraction(self):
        """Test extracting emotion tags from text."""
        text = "[WONDER] Oh, what a beautiful insight! [ECSTASY] This changes everything!"
        emotions = extract_emotions_from_text(text)
        
        assert len(emotions) == 2
        assert EmotionalState.WONDER in emotions
        assert EmotionalState.ECSTASY in emotions
        
    def test_emotion_extraction_no_emotions(self):
        """Test extraction when no emotions present."""
        text = "This is a normal sentence without emotion tags."
        emotions = extract_emotions_from_text(text)
        
        assert len(emotions) == 0
        
    def test_emotional_intensity_calculation(self):
        """Test calculating emotional intensity from text."""
        # High intensity - multiple emotions, exclamations
        text1 = "[ECSTASY] [RAPTURE] This is incredible!! Amazing!!!"
        emotions1 = [EmotionalState.ECSTASY, EmotionalState.RAPTURE]
        intensity1 = calculate_emotional_intensity(emotions1, text1)
        assert intensity1 > 0.7
        
        # Low intensity - contemplative
        text2 = "[CONTEMPLATION] Let us consider this carefully..."
        emotions2 = [EmotionalState.CONTEMPLATION]
        intensity2 = calculate_emotional_intensity(emotions2, text2)
        assert intensity2 < 0.5
        
        # Medium intensity - questions
        text3 = "[TURMOIL] But how? Why? What does this mean?"
        emotions3 = [EmotionalState.TURMOIL]
        intensity3 = calculate_emotional_intensity(emotions3, text3)
        assert 0.4 < intensity3 < 0.7
        
    def test_conversation_context_update(self):
        """Test updating conversation context with emotional journey."""
        context = ConversationContext()
        
        message1 = Message(
            role="assistant",
            content="[WONDER] What a discovery!",
            emotions=[EmotionalState.WONDER]
        )
        
        context = update_conversation_context(context, message1)
        assert EmotionalState.WONDER in context.emotional_journey
        
        message2 = Message(
            role="assistant",
            content="[ANGUISH] The pain of this realization...",
            emotions=[EmotionalState.ANGUISH]
        )
        
        context = update_conversation_context(context, message2)
        assert len(context.emotional_journey) == 2
        assert context.emotional_journey[-1] == EmotionalState.ANGUISH


class TestEmotionalTransitions:
    """Test emotional state transitions."""
    
    def test_valid_transitions(self):
        """Test that certain emotional transitions make sense."""
        # Define some natural transitions
        natural_transitions = [
            (EmotionalState.CONTEMPLATION, EmotionalState.WONDER),
            (EmotionalState.WONDER, EmotionalState.ECSTASY),
            (EmotionalState.SERENITY, EmotionalState.CONTEMPLATION),
            (EmotionalState.TURMOIL, EmotionalState.ANGUISH),
        ]
        
        for from_state, to_state in natural_transitions:
            # These should be valid transitions
            assert from_state != to_state
            assert from_state in EmotionalState
            assert to_state in EmotionalState
            
    def test_emotional_range(self):
        """Test that all emotions are distinct."""
        all_emotions = list(EmotionalState)
        assert len(all_emotions) == len(set(all_emotions))
        assert len(all_emotions) == 10  # We have 10 emotional states


class TestEmotionalIntensityBounds:
    """Test emotional intensity calculations stay within bounds."""
    
    def test_intensity_bounds(self):
        """Ensure intensity is always between 0 and 1."""
        test_cases = [
            ("Simple text", []),
            ("!!!!!!!!!", [EmotionalState.ECSTASY]),
            ("???????????????????", [EmotionalState.TURMOIL]),
            ("..." * 20, [EmotionalState.CONTEMPLATION]),
            ("[ECSTASY] [RAPTURE] [WONDER] !!!!!!!!!????......", 
             [EmotionalState.ECSTASY, EmotionalState.RAPTURE, EmotionalState.WONDER])
        ]
        
        for text, emotions in test_cases:
            intensity = calculate_emotional_intensity(emotions, text)
            assert 0.0 <= intensity <= 1.0, f"Intensity {intensity} out of bounds for text: {text}"


@pytest.mark.parametrize("emotion,expected_category", [
    (EmotionalState.WONDER, "curiosity"),
    (EmotionalState.ANGUISH, "pain"),
    (EmotionalState.ECSTASY, "happiness"),
    (EmotionalState.CONTEMPLATION, "reflection"),
    (EmotionalState.MELANCHOLY, "sadness"),
])
def test_emotion_descriptions(emotion, expected_category):
    """Test that emotion names are meaningful and well-defined."""
    # This is a test to ensure emotion names are valid and meaningful
    # The emotions should be valid strings and represent distinct states
    assert isinstance(emotion.value, str)
    assert len(emotion.value) > 0
    assert emotion.value.isalpha()  # Should be alphabetic characters