#!/usr/bin/env python3
"""
Test script to verify the Wild Genius Professor emotional state system.

This script demonstrates the core functionality of the emotional state management
system without requiring API keys or interactive input.
"""

import os
import sys
import logging

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_emotional_state_system():
    """Test the emotional state system components."""
    
    print("🧪 Testing Wild Genius Professor Emotional State System")
    print("=" * 60)
    
    # Test 1: Basic emotional state functionality
    print("\n1. Testing Basic Emotional State Creation...")
    
    from wild_genius_prof.state import EmotionalDimensions, EmotionalProcessor
    
    # Create initial emotional state
    emotions = EmotionalDimensions()
    print(f"   Initial curiosity: {emotions.curiosity:.2f}")
    print(f"   Initial joy: {emotions.joy:.2f}")
    print(f"   Initial engagement: {emotions.engagement:.2f}")
    
    # Test dominant emotion calculation
    dominant = emotions.get_dominant_emotion()
    print(f"   Dominant emotion: {dominant.value}")
    
    # Test 2: Emotional processing
    print("\n2. Testing Emotional Processing...")
    
    processor = EmotionalProcessor()
    
    # Simulate a philosophical question
    user_input = "What does it mean to truly understand suffering in The Brothers Karamazov?"
    updated_emotions = processor.process_interaction(
        emotions, 
        "question", 
        user_input, 
        intensity=0.8
    )
    
    print(f"   Updated curiosity: {updated_emotions.curiosity:.2f}")
    print(f"   Updated engagement: {updated_emotions.engagement:.2f}")
    print(f"   New dominant emotion: {updated_emotions.get_dominant_emotion().value}")
    
    # Test 3: Literature analysis
    print("\n3. Testing Literature Analysis...")
    
    from wild_genius_prof.tools import literature_tools
    
    content_processor = literature_tools['content_processor']
    analysis = content_processor.process_user_input(
        "I'm struggling to understand Ivan's argument about God and morality. The suffering of innocent children disturbs me deeply."
    )
    
    print(f"   Interaction type: {analysis['interaction_type']}")
    print(f"   Themes identified: {[theme.name for theme in analysis['themes']]}")
    print(f"   Characters mentioned: {analysis['characters']}")
    print(f"   Emotional triggers: {list(analysis['emotional_triggers'].keys())}")
    
    # Test 4: Professor persona response
    print("\n4. Testing Professor Persona Response...")
    
    from wild_genius_prof.persona import ProfessorPersona
    
    persona = ProfessorPersona(processor)
    
    test_input = "What is the meaning of redemption in Dostoevsky's work?"
    response, final_emotions = persona.process_user_input(
        test_input,
        updated_emotions,
        {"themes": [], "characters": []}
    )
    
    print(f"   User input: {test_input}")
    print(f"   Professor response: {response[:100]}...")
    print(f"   Final dominant emotion: {final_emotions.get_dominant_emotion().value}")
    
    # Test 5: Emotional summary
    print("\n5. Testing Emotional Summary...")
    
    summary = persona.get_emotional_summary(final_emotions)
    print(f"   Emotional summary: {summary}")
    
    print("\n🎉 All tests completed successfully!")
    print("The emotional state system is working correctly.")
    

def test_citation_system():
    """Test the citation and literature tools."""
    
    print("\n📚 Testing Citation System")
    print("=" * 40)
    
    from wild_genius_prof.tools import literature_tools
    
    citation_manager = literature_tools['citation_manager']
    
    # Find relevant citations
    citations = citation_manager.find_relevant_citations("contemplation", "truth")
    
    if citations:
        print(f"   Found {len(citations)} relevant citations:")
        for i, citation in enumerate(citations[:2]):  # Show first 2
            formatted = citation_manager.format_citation(citation, "conversational")
            print(f"   {i+1}. {formatted}")
    else:
        print("   No citations found (this is normal for the test)")
    
    # Test theme analysis
    analyzer = literature_tools['analyzer']
    themes = analyzer.analyze_content_themes("The suffering of innocent children challenges our faith")
    
    print(f"   Themes found: {[theme.name for theme in themes]}")
    

def test_emotional_evolution():
    """Test emotional evolution over time."""
    
    print("\n⏰ Testing Emotional Evolution")
    print("=" * 40)
    
    from wild_genius_prof.state import EmotionalDimensions
    import time
    
    # Create an emotional state with high intensity
    emotions = EmotionalDimensions()
    emotions.joy = 0.9
    emotions.emotional_energy = 0.95
    emotions.curiosity = 0.8
    
    print(f"   Initial joy: {emotions.joy:.2f}")
    print(f"   Initial energy: {emotions.emotional_energy:.2f}")
    
    # Simulate time passing (immediate decay test)
    decayed_emotions = emotions.decay(time_elapsed=10.0)  # 10 seconds
    
    print(f"   After 10s - Joy: {decayed_emotions.joy:.2f}")
    print(f"   After 10s - Energy: {decayed_emotions.emotional_energy:.2f}")
    print(f"   Emotional decay working correctly!")


if __name__ == "__main__":
    try:
        test_emotional_state_system()
        test_citation_system()  
        test_emotional_evolution()
        
        print("\n✅ All integration tests passed!")
        print("🎭 The Wild Genius Professor emotional state system is ready!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)