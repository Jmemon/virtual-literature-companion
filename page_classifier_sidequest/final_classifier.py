import re
from typing import List, Dict
import statistics

def classify_page(text: str, page_num: int, total_pages: int, mean_words: float, std_words: float) -> str:
    text = text.strip()
    word_count = len(text.split())
    
    def has_dialog_patterns(text: str) -> bool:
        patterns = [
            r'"[^"]+(?:said|asked|replied|murmured)"',  # Quote with speech verb
            r'"[^"]{10,}"',  # General quoted speech
            r'(?m)^[A-Z][A-Z\s]+:',  # Character names in plays
            r'(?m)^FTLN \d+',  # Folger play line markers
            r'(?m)^[A-Z][A-Z\s]+(?:\s+sings|\s+enters|\s+exits)',  # Stage directions
            r'[A-Z][a-z]+:\s*"',  # Character name followed by quote
            r'\b(?:said|asked|replied)\b.*"'  # Speech verbs with quotes
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def has_narrative_patterns(text: str) -> bool:
        patterns = [
            r'\b(?:he|she|they)\s+(?:was|were|had|felt|saw|heard|thought)\b',
            r'(?<=[.!?])\s+[A-Z][a-z]+\s+[a-z]+ed\b',  # Sentence + character action
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:looked|turned|walked|smiled)',
            r'[.!?][^.!?"]{10,}[.!?]'  # Longer narrative sentences
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def is_valid_content(text: str, word_count: int) -> bool:
        if word_count < 15:
            return False
        
        # Check for excessive punctuation or numbers
        punct_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        num_ratio = len(re.findall(r'\d', text)) / len(text) if text else 0
        
        if punct_ratio > 0.3 or num_ratio > 0.2:
            return False
            
        return True

    # Main classification logic
    if not is_valid_content(text, word_count):
        return "throwaway"

    # Strong indicators of content
    if (has_dialog_patterns(text) or has_narrative_patterns(text)) and word_count > 15:
        return "content"

    # Check for page numbers only
    if re.match(r'^\s*\d+\s*$', text):
        return "throwaway"

    # Default for text that looks like content
    if word_count > 20 and not re.search(r'(?i)(copyright|all rights reserved|contents)', text):
        return "content"

    return "throwaway"