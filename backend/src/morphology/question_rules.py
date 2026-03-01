"""
Question-Type Aware Refinement Rules

Different question types (who, where, when, what, etc.) expect different
answer types and have different morphological patterns.

This module provides rules for context-aware answer boundary refinement.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class QuestionTypeRule:
    """Rules for a specific question type."""
    question_words: List[str]      # Telugu question words
    expected_type: str             # Expected answer category
    always_remove: List[str]       # Suffixes to always trim
    usually_remove: List[str]      # Suffixes to usually trim
    preserve: List[str]            # Patterns to preserve
    description: str               # Human-readable description


# =============================================================================
# QUESTION TYPE RULES
# =============================================================================

QUESTION_TYPE_RULES: Dict[str, QuestionTypeRule] = {
    
    # WHO questions - expect person/entity names
    'who': QuestionTypeRule(
        question_words=['ఎవరు', 'ఎవర్', 'ఎవరి', 'ఎవరికి', 'ఎవరితో'],
        expected_type='PERSON',
        always_remove=['గారు', 'వారు', 'గా', 'తో', 'కు', 'కి', 'ని', 'చే'],
        usually_remove=['యొక్క', 'గారి', 'వారి'],
        preserve=['రావు', 'రెడ్డి', 'నాయుడు', 'శర్మ', 'వర్మ'],  # Common name suffixes
        description='Person names - remove honorifics and case markers'
    ),
    
    # WHERE questions - expect location names
    'where': QuestionTypeRule(
        question_words=['ఎక్కడ', 'ఎక్కడి', 'ఎక్కడికి', 'ఎక్కడినుండి', 'ఎక్కడ్'],
        expected_type='LOCATION',
        always_remove=['లో', 'లోని', 'లోకి', 'లోంచి', 'పై', 'పైన', 'మీద', 'కింద', 
                       'వద్ద', 'దగ్గర', 'నుండి', 'నుంచి', 'వైపు'],
        usually_remove=['కు', 'కి', 'లోపల'],
        preserve=['పురం', 'నగర్', 'బాద్', 'పేట', 'గూడెం'],  # Common place name suffixes
        description='Location names - remove locative markers'
    ),
    
    # WHEN questions - expect time/date
    'when': QuestionTypeRule(
        question_words=['ఎప్పుడు', 'ఎప్పటి', 'ఎప్పటికి', 'ఎప్పటినుండి'],
        expected_type='TIME',
        always_remove=['లో', 'న', 'నాడు', 'కు', 'నుండి', 'వరకు', 'తరువాత', 'ముందు'],
        usually_remove=['లోని'],
        preserve=['సంవత్సరం', 'నెల', 'తేది', 'శతాబ్దం'],  # Time-related words
        description='Time expressions - remove temporal markers'
    ),
    
    # HOW MUCH/MANY questions - expect quantities
    'how_much': QuestionTypeRule(
        question_words=['ఎంత', 'ఎన్ని', 'ఎంతమంది', 'ఎంతకు'],
        expected_type='QUANTITY',
        always_remove=['కు', 'వరకు', 'కి', 'గా'],
        usually_remove=['మంది', 'లు'],
        preserve=['కోట్లు', 'లక్షలు', 'వేలు', 'శాతం', 'కి.మీ', 'మీటర్లు'],
        description='Quantities - preserve units'
    ),
    
    # WHAT questions - general entities
    'what': QuestionTypeRule(
        question_words=['ఏమిటి', 'ఏమి', 'ఏ', 'ఏవి', 'ఏంటి', 'ఏమిటో'],
        expected_type='ENTITY',
        always_remove=['అయిన', 'మైన', 'అనే', 'అని'],
        usually_remove=['గా', 'యొక్క', 'లోని'],
        preserve=[],
        description='General entities - remove participial markers'
    ),
    
    # WHY questions - expect reasons
    'why': QuestionTypeRule(
        question_words=['ఎందుకు', 'ఎందుకని', 'ఎందువల్ల', 'ఎందుచేత'],
        expected_type='REASON',
        always_remove=['వల్ల', 'చేత', 'కారణంగా'],
        usually_remove=['కోసం', 'కు'],
        preserve=['వల్ల', 'కారణం'],  # Sometimes valid in reason answers
        description='Reasons - may include causal markers'
    ),
    
    # HOW questions - expect manner/method
    'how': QuestionTypeRule(
        question_words=['ఎలా', 'ఏ విధంగా', 'ఎట్లా'],
        expected_type='MANNER',
        always_remove=['గా', 'లా'],
        usually_remove=['ద్వారా', 'చేత'],
        preserve=['విధంగా', 'రీతిలో'],
        description='Manner/method - remove adverbial endings'
    ),
    
    # WHICH questions - specific selection
    'which': QuestionTypeRule(
        question_words=['ఏ', 'ఏది', 'ఏవి', 'ఏదైనా'],
        expected_type='SELECTION',
        always_remove=['అయిన', 'మైన'],
        usually_remove=['లో', 'లోని'],
        preserve=[],
        description='Selection questions - context-dependent'
    ),
}


def detect_question_type(question: str) -> str:
    """
    Detect the type of question based on Telugu question words.
    
    Args:
        question: The Telugu question string
        
    Returns:
        Question type key ('who', 'where', 'when', etc.) or 'unknown'
    """
    question_lower = question.strip()
    
    for qtype, rule in QUESTION_TYPE_RULES.items():
        for qword in rule.question_words:
            if qword in question_lower:
                return qtype
    
    return 'unknown'


def get_rule_for_question(question: str) -> QuestionTypeRule:
    """
    Get the refinement rule for a given question.
    
    Args:
        question: The Telugu question string
        
    Returns:
        QuestionTypeRule or default rule for unknown questions
    """
    qtype = detect_question_type(question)
    
    if qtype in QUESTION_TYPE_RULES:
        return QUESTION_TYPE_RULES[qtype]
    
    # Default rule for unknown question types
    return QuestionTypeRule(
        question_words=[],
        expected_type='UNKNOWN',
        always_remove=['బడిన', 'అయిన', 'చేసిన'],  # Most common over-extraction
        usually_remove=['లో', 'లోని', 'తో', 'కు'],
        preserve=[],
        description='Default rules for unknown question type'
    )


def get_suffixes_to_remove(question: str, aggressive: bool = False) -> Set[str]:
    """
    Get the set of suffixes that should be removed based on question type.
    
    Args:
        question: The Telugu question string
        aggressive: If True, include 'usually_remove' suffixes too
        
    Returns:
        Set of suffix strings to remove
    """
    rule = get_rule_for_question(question)
    
    suffixes = set(rule.always_remove)
    if aggressive:
        suffixes.update(rule.usually_remove)
    
    # Remove preserved patterns from the removal set
    suffixes -= set(rule.preserve)
    
    return suffixes


def should_preserve_ending(answer: str, question: str) -> bool:
    """
    Check if the answer ending should be preserved based on question type.
    
    Args:
        answer: The predicted answer
        question: The question
        
    Returns:
        True if ending should be preserved
    """
    rule = get_rule_for_question(question)
    
    for preserve_pattern in rule.preserve:
        if answer.endswith(preserve_pattern):
            return True
    
    return False
