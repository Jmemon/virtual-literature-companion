"""
Unit tests for data models.

This module tests:
- Schema validation
- Model methods
- Data transformations
"""

import pytest
from datetime import datetime
from src.models.schemas import (
    EmotionTag,
    EmotionalResponse,
    Citation,
    TextChunk,
    UIAnimation,
    EmotiveUIState,
    BookMetadata,
    TableOfContents
)
from src.models.state import AgentState


class TestEmotionalResponse:
    """Test the EmotionalResponse model."""
    
    def test_create_basic_response(self):
        """Test creating a basic emotional response."""
        response = EmotionalResponse(
            text="I find this chapter deeply moving.",
            emotions=[EmotionTag.MELANCHOLIC, EmotionTag.CONTEMPLATIVE]
        )
        
        assert response.text == "I find this chapter deeply moving."
        assert len(response.emotions) == 2
        assert EmotionTag.MELANCHOLIC in response.emotions
        
    def test_emotion_intensities_validation(self):
        """Test that emotion intensities are properly validated."""
        response = EmotionalResponse(
            text="What a joyful passage!",
            emotions=[EmotionTag.JOY],
            emotion_intensities={"joy": 0.8}
        )
        
        assert response.emotion_intensities["joy"] == 0.8
        
        # Test invalid intensity
        with pytest.raises(ValueError):
            EmotionalResponse(
                text="Test",
                emotions=[EmotionTag.JOY],
                emotion_intensities={"joy": 1.5}  # > 1.0
            )
            
    def test_default_intensity(self):
        """Test that emotions get default intensity if not specified."""
        response = EmotionalResponse(
            text="Interesting observation.",
            emotions=[EmotionTag.CURIOUS, EmotionTag.ANALYTICAL]
        )
        
        # Should have default intensities
        assert response.emotion_intensities["curious"] == 0.5
        assert response.emotion_intensities["analytical"] == 0.5


class TestCitation:
    """Test the Citation model."""
    
    def test_create_citation_with_page(self):
        """Test creating a citation with page reference."""
        citation = Citation(
            text="It was the best of times, it was the worst of times",
            chapter=1,
            page=1
        )
        
        assert citation.text == "It was the best of times, it was the worst of times"
        assert citation.chapter == 1
        assert citation.page == 1
        
    def test_citation_without_page(self):
        """Test creating a citation without specific page."""
        citation = Citation(
            text="Call me Ishmael",
            chapter=1
        )
        
        assert citation.page is None


class TestTextChunk:
    """Test the TextChunk model."""
    
    def test_create_chunk(self):
        """Test creating a text chunk."""
        chunk = TextChunk(
            book_id="book123",
            chapter_number=5,
            page_number=42,
            content="This is a sample text chunk from the book.",
            start_char=1000,
            end_char=1043
        )
        
        assert chunk.book_id == "book123"
        assert chunk.chapter_number == 5
        assert chunk.page_number == 42
        assert len(chunk.content) == 43  # end_char - start_char
        
    def test_chunk_validation(self):
        """Test chunk content validation."""
        # Empty content should fail
        with pytest.raises(ValueError):
            TextChunk(
                book_id="book123",
                content="",  # Empty
                start_char=0,
                end_char=0
            )
            
        # Whitespace-only content should fail
        with pytest.raises(ValueError):
            TextChunk(
                book_id="book123",
                content="   \n\t  ",  # Only whitespace
                start_char=0,
                end_char=7
            )


class TestBookMetadata:
    """Test the BookMetadata model."""
    
    def test_create_book_metadata(self):
        """Test creating book metadata."""
        metadata = BookMetadata(
            title="Pride and Prejudice",
            author="Jane Austen",
            isbn="9780141439518",
            publication_year=1813,
            page_count=432,
            file_path="/uploads/pride_and_prejudice.pdf"
        )
        
        assert metadata.title == "Pride and Prejudice"
        assert metadata.author == "Jane Austen"
        assert metadata.book_id  # Should have auto-generated ID
        
    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        metadata = BookMetadata(
            title="Unknown Book",
            file_path="/uploads/unknown.pdf"
        )
        
        assert metadata.author is None
        assert metadata.isbn is None
        assert metadata.publication_year is None


class TestTableOfContents:
    """Test the TableOfContents model."""
    
    def test_create_toc(self):
        """Test creating a table of contents."""
        chapters = [
            TableOfContents.Chapter(
                chapter_number=1,
                title="The Beginning",
                start_page=1,
                end_page=20
            ),
            TableOfContents.Chapter(
                chapter_number=2,
                title="The Middle",
                start_page=21,
                end_page=50
            )
        ]
        
        toc = TableOfContents(
            chapters=chapters,
            total_chapters=2,
            has_prologue=False,
            has_epilogue=False
        )
        
        assert len(toc.chapters) == 2
        assert toc.total_chapters == 2
        assert toc.chapters[0].title == "The Beginning"


class TestAgentState:
    """Test the AgentState model."""
    
    def test_create_agent_state(self):
        """Test creating an agent state."""
        state = AgentState(user_id="user123")
        
        assert state.user_id == "user123"
        assert state.session_id  # Should have auto-generated session ID
        assert isinstance(state.messages, list)
        assert len(state.messages) == 0
        
    def test_add_message(self):
        """Test adding messages to state."""
        state = AgentState(user_id="user123")
        
        state.add_message("user", "Hello, let's discuss Chapter 1")
        state.add_message("assistant", "I'd love to discuss Chapter 1!")
        
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"
        assert state.messages[1]["content"] == "I'd love to discuss Chapter 1!"
        
    def test_get_latest_user_message(self):
        """Test retrieving the latest user message."""
        state = AgentState(user_id="user123")
        
        state.add_message("user", "First message")
        state.add_message("assistant", "Response")
        state.add_message("user", "Second message")
        
        latest = state.get_latest_user_message()
        assert latest == "Second message"
        
    def test_emotional_state_tracking(self):
        """Test emotional state tracking."""
        state = AgentState(user_id="user123")
        
        emotions = [EmotionTag.JOY, EmotionTag.CURIOUS]
        intensities = {"joy": 0.8, "curious": 0.6}
        
        state.update_emotional_state(emotions, intensities)
        
        assert state.current_emotions == emotions
        assert len(state.emotion_history) == 1
        assert state.emotion_history[0]["emotions"] == ["joy", "curious"]
        
    def test_conversation_context(self):
        """Test getting conversation context."""
        state = AgentState(user_id="user123")
        
        # Add some messages
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            state.add_message(role, f"Message {i}")
            
        # Get last 2 exchanges (4 messages)
        context = state.get_conversation_context(max_turns=2)
        
        assert "Message 2" not in context  # Should not include older messages
        assert "Message 3" in context
        assert "Message 5" in context


class TestUIAnimation:
    """Test the UIAnimation model."""
    
    def test_create_animation(self):
        """Test creating a UI animation."""
        animation = UIAnimation(
            element_id="emotion-shape",
            animation_type="morph",
            duration=2.5,
            parameters={
                "easing": "easeInOut",
                "loop": True
            }
        )
        
        assert animation.element_id == "emotion-shape"
        assert animation.duration == 2.5
        assert animation.parameters["loop"] is True
        assert animation.trigger_time == 0  # Default
        
    def test_animation_with_delay(self):
        """Test animation with trigger delay."""
        animation = UIAnimation(
            element_id="particles",
            animation_type="burst",
            duration=1.0,
            trigger_time=0.5,
            parameters={"count": 50}
        )
        
        assert animation.trigger_time == 0.5


class TestEmotiveUIState:
    """Test the EmotiveUIState model."""
    
    def test_create_ui_state(self):
        """Test creating an emotive UI state."""
        animations = [
            UIAnimation(
                element_id="background",
                animation_type="gradient",
                duration=3.0,
                parameters={}
            )
        ]
        
        ui_state = EmotiveUIState(
            primary_emotion=EmotionTag.CONTEMPLATIVE,
            emotion_blend={
                EmotionTag.CONTEMPLATIVE: 0.7,
                EmotionTag.MELANCHOLIC: 0.3
            },
            animations=animations,
            color_palette={
                "primary": "#708090",
                "secondary": "#778899"
            },
            particle_effects={
                "enabled": True,
                "count": 30
            },
            morphing_shape={
                "type": "circle",
                "vertices": 12
            }
        )
        
        assert ui_state.primary_emotion == EmotionTag.CONTEMPLATIVE
        assert len(ui_state.animations) == 1
        assert ui_state.emotion_blend[EmotionTag.CONTEMPLATIVE] == 0.7


if __name__ == "__main__":
    pytest.main([__file__])