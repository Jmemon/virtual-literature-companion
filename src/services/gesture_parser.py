"""
Gesture parsing service for the Literary Companion.

This module handles:
- Extraction of gesture tags from AI responses
- Parsing gesture parameters
- Validation of gesture syntax
- Removal of gesture tags for clean display
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

from ..models.state import Gesture, GestureType

logger = logging.getLogger(__name__)


class GestureParser:
    """
    Parses and extracts gesture annotations from text.
    
    Gestures are embedded in text using the format:
    [GESTURE:TYPE] or [GESTURE:TYPE:param1,param2,...]
    """
    
    # Pattern to match gesture tags
    GESTURE_PATTERN = re.compile(
        r'\[GESTURE:([A-Z_]+)(?::([^\]]+))?\]'
    )
    
    # Default durations for different gesture types (in seconds)
    GESTURE_DURATIONS = {
        GestureType.LEAN_IN: 2.0,
        GestureType.PULL_BACK: 2.5,
        GestureType.TREMBLE: 1.5,
        GestureType.ILLUMINATE: 3.0,
        GestureType.FRAGMENT: 2.0,
        GestureType.WHISPER: 3.5,
        GestureType.GRIP: 2.0,
        GestureType.SHATTER: 3.0,
        GestureType.BREATHE: 4.0,
        GestureType.REACH: 2.5,
        GestureType.DANCE: 3.0,
    }
    
    def extract_gestures(self, text: str) -> List[Gesture]:
        """
        Extract all gestures from the given text.
        
        Args:
            text: Text containing gesture annotations
            
        Returns:
            List of parsed Gesture objects
        """
        gestures = []
        
        for match in self.GESTURE_PATTERN.finditer(text):
            gesture_type_str = match.group(1)
            parameters_str = match.group(2)
            
            try:
                # Validate gesture type
                gesture_type = GestureType(gesture_type_str)
                
                # Parse parameters if present
                parameters = []
                if parameters_str:
                    parameters = [p.strip() for p in parameters_str.split(',')]
                
                # Create gesture with appropriate duration
                gesture = Gesture(
                    type=gesture_type,
                    parameters=parameters,
                    timestamp=datetime.now().timestamp(),
                    duration=self.GESTURE_DURATIONS.get(gesture_type, 2.0)
                )
                
                gestures.append(gesture)
                
            except ValueError as e:
                logger.warning(f"Invalid gesture type: {gesture_type_str}")
                continue
        
        return gestures
    
    def remove_gesture_tags(self, text: str) -> str:
        """
        Remove all gesture tags from text for clean display.
        
        Args:
            text: Text containing gesture annotations
            
        Returns:
            Text with gesture tags removed
        """
        return self.GESTURE_PATTERN.sub('', text).strip()
    
    def validate_gesture_syntax(self, gesture_str: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a gesture string syntax.
        
        Args:
            gesture_str: Gesture string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        match = self.GESTURE_PATTERN.match(gesture_str)
        
        if not match:
            return False, "Invalid gesture format. Use [GESTURE:TYPE] or [GESTURE:TYPE:params]"
        
        gesture_type_str = match.group(1)
        
        try:
            GestureType(gesture_type_str)
        except ValueError:
            valid_types = ', '.join([g.value for g in GestureType])
            return False, f"Invalid gesture type. Valid types: {valid_types}"
        
        return True, None
    
    def parse_single_gesture(self, gesture_str: str) -> Optional[Gesture]:
        """
        Parse a single gesture string.
        
        Args:
            gesture_str: Single gesture string
            
        Returns:
            Parsed Gesture object or None if invalid
        """
        gestures = self.extract_gestures(gesture_str)
        return gestures[0] if gestures else None
    
    def get_gesture_info(self, gesture_type: GestureType) -> Dict[str, Any]:
        """
        Get information about a specific gesture type.
        
        Args:
            gesture_type: The gesture type to get info for
            
        Returns:
            Dictionary with gesture information
        """
        return {
            'type': gesture_type.value,
            'duration': self.GESTURE_DURATIONS.get(gesture_type, 2.0),
            'description': self._get_gesture_description(gesture_type)
        }
    
    def _get_gesture_description(self, gesture_type: GestureType) -> str:
        """Get human-readable description of a gesture."""
        descriptions = {
            GestureType.LEAN_IN: "Interface moves closer, creating intimacy",
            GestureType.PULL_BACK: "Creates contemplative space after heavy revelations",
            GestureType.TREMBLE: "Individual letters vibrate with overwhelming emotion",
            GestureType.ILLUMINATE: "Key words pulse with golden significance",
            GestureType.FRAGMENT: "Words break apart and reform to show shattered understanding",
            GestureType.WHISPER: "Text fades to near-transparency, requiring focused attention",
            GestureType.GRIP: "Borders thicken, container contracts to seize attention",
            GestureType.SHATTER: "The entire interface cracks like breaking glass",
            GestureType.BREATHE: "Everything expands/contracts in meditative rhythm",
            GestureType.REACH: "Words create bridges to connect ideas visually",
            GestureType.DANCE: "Elements spiral in synchronized joy"
        }
        
        return descriptions.get(gesture_type, "Unknown gesture")
    
    def combine_overlapping_gestures(self, gestures: List[Gesture]) -> List[Dict[str, Any]]:
        """
        Analyze gestures for overlapping timing and choreography.
        
        Args:
            gestures: List of gestures to analyze
            
        Returns:
            List of gesture timing information for UI choreography
        """
        if not gestures:
            return []
        
        # Sort by timestamp
        sorted_gestures = sorted(gestures, key=lambda g: g.timestamp)
        
        timeline = []
        for i, gesture in enumerate(sorted_gestures):
            gesture_info = {
                'gesture': gesture,
                'start_time': gesture.timestamp,
                'end_time': gesture.timestamp + gesture.duration,
                'overlaps_with': []
            }
            
            # Check for overlaps with other gestures
            for j, other in enumerate(sorted_gestures):
                if i != j:
                    # Check if gestures overlap in time
                    if (gesture.timestamp <= other.timestamp <= gesture_info['end_time'] or
                        other.timestamp <= gesture.timestamp <= other.timestamp + other.duration):
                        gesture_info['overlaps_with'].append(j)
            
            timeline.append(gesture_info)
        
        return timeline