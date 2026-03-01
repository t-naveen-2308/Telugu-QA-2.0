# -*- coding: utf-8 -*-
"""
Telugu Coreference Resolution for QA Post-processing

Resolves demonstrative pronouns and anaphoric references to their antecedents.
Handles cases like "ఈ నగరంలో" → "విజయవాడలో"
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ResolvedReference:
    """Result of coreference resolution."""
    original: str
    resolved: str
    antecedent: str
    demonstrative: str
    noun_class: str
    confidence: float


# Telugu demonstrative pronouns
DEMONSTRATIVES = {
    # Proximal (this)
    'ఈ': 'proximal',
    'ఇది': 'proximal',
    'ఇక్కడ': 'proximal_place',
    'ఇప్పుడు': 'proximal_time',
    
    # Distal (that)
    'ఆ': 'distal',
    'అది': 'distal',
    'అక్కడ': 'distal_place',
    'అప్పుడు': 'distal_time',
    
    # Other anaphoric
    'దీని': 'proximal_genitive',
    'దానిలో': 'distal_locative',
    'వాటిలో': 'distal_plural_locative',
}

# Noun classes with their Telugu forms and entity patterns
NOUN_CLASSES = {
    'city': {
        'nouns': ['నగరం', 'నగరంలో', 'పట్టణం', 'పట్టణంలో', 'ఊరు', 'ఊరిలో', 'నగరంలోని'],
        'entity_suffixes': ['పురం', 'నగర్', 'బాద్', 'పేట', 'వాడ', 'గూడెం', 'పల్లి', 'పట్నం'],
        'entity_pattern': r'[ఁ-ౡా-ో్]+(?:పురం|నగర్|బాద్|పేట|వాడ|గూడెం|పల్లి|పట్నం)|హైదరాబాద్|విజయవాడ|విశాఖపట్నం|తిరుపతి|వరంగల్|గుంటూరు|నెల్లూరు|కాకినాడ|రాజమండ్రి',
    },
    'state': {
        'nouns': ['రాష్ట్రం', 'రాష్ట్రంలో', 'ప్రదేశ్', 'ప్రాంతం', 'రాష్ట్రంలోని', 'ప్రాంతంలో'],
        'entity_suffixes': ['ప్రదేశ్', 'రాష్ట్రం'],
        'entity_pattern': r'తెలంగాణ|ఆంధ్రప్రదేశ్|కర్ణాటక|తమిళనాడు|కేరళ|మహారాష్ట్ర|గుజరాత్|రాజస్థాన్|[ఁ-ౡా-ో్]+(?:ప్రదేశ్|రాష్ట్రం)',
    },
    'country': {
        'nouns': ['దేశం', 'దేశంలో', 'రాజ్యం', 'దేశంలోని'],
        'entity_suffixes': [],
        'entity_pattern': r'భారతదేశం|భారత్|అమెరికా|చైనా|జపాన్|రష్యా|బ్రిటన్|జర్మనీ|ఫ్రాన్స్',
    },
    'river': {
        'nouns': ['నది', 'నదిలో', 'నదీ'],
        'entity_suffixes': ['నది', 'గంగ', 'యమున'],
        'entity_pattern': r'[ఁ-ౡా-ో్]+(?:నది|గంగ)|కృష్ణా|గోదావరి|కావేరి|నర్మద|తుంగభద్ర|పెన్నా',
    },
    'temple': {
        'nouns': ['ఆలయం', 'ఆలయంలో', 'గుడి', 'దేవాలయం', 'మందిరం', 'క్షేత్రం'],
        'entity_suffixes': ['ఆలయం', 'గుడి', 'మందిరం', 'క్షేత్రం'],
        'entity_pattern': r'[ఁ-ౡా-ో్]+(?:ఆలయం|గుడి|మందిరం|క్షేత్రం)',
    },
    'person': {
        'nouns': ['వ్యక్తి', 'నాయకుడు', 'రాజు', 'మహారాజు', 'ప్రధాని', 'ముఖ్యమంత్రి'],
        'entity_suffixes': ['రావు', 'రెడ్డి', 'నాయుడు', 'శర్మ', 'వర్మ', 'గుప్త'],
        'entity_pattern': r'[ఁ-ౡా-ో్]+(?:రావు|రెడ్డి|నాయుడు|శర్మ|వర్మ|గుప్త|గారు)',
    },
    'organization': {
        'nouns': ['సంస్థ', 'కంపెనీ', 'విశ్వవిద్యాలయం', 'కళాశాల', 'పాఠశాల'],
        'entity_suffixes': ['విశ్వవిద్యాలయం', 'యూనివర్సిటీ'],
        'entity_pattern': r'[ఁ-ౡా-ో్]+(?:విశ్వవిద్యాలయం|యూనివర్సిటీ|కంపెనీ|సంస్థ)',
    },
    'year': {
        'nouns': ['సంవత్సరం', 'సంవత్సరంలో', 'ఏడాది'],
        'entity_suffixes': [],
        'entity_pattern': r'\d{4}',
    },
}


class CoreferenceResolver:
    """
    Resolves coreference in Telugu QA answers.
    
    Handles demonstrative pronouns (ఈ, ఆ) + noun patterns and 
    resolves them to actual entities from context.
    """
    
    def __init__(self):
        self.demonstratives = DEMONSTRATIVES
        self.noun_classes = NOUN_CLASSES
        
        # Build combined patterns
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for detection."""
        # Pattern for demonstrative + noun
        demo_pattern = '|'.join(re.escape(d) for d in self.demonstratives.keys())
        
        all_nouns = []
        for nc in self.noun_classes.values():
            all_nouns.extend(nc['nouns'])
        noun_pattern = '|'.join(re.escape(n) for n in all_nouns)
        
        # Match "ఈ నగరంలో" or "ఈ నగరం"
        self.demo_noun_pattern = re.compile(
            f'({demo_pattern})\\s*({noun_pattern})',
            re.UNICODE
        )
    
    def detect_demonstrative_reference(self, answer: str) -> Optional[Tuple[str, str, str]]:
        """
        Detect if answer contains a demonstrative reference.
        
        Args:
            answer: The answer text
            
        Returns:
            Tuple of (demonstrative, noun, noun_class) or None
        """
        match = self.demo_noun_pattern.search(answer)
        if match:
            demo = match.group(1)
            noun = match.group(2)
            
            # Find noun class
            for class_name, class_info in self.noun_classes.items():
                if noun in class_info['nouns']:
                    return (demo, noun, class_name)
        
        return None
    
    def find_antecedent(
        self,
        noun_class: str,
        context: str,
        answer_start: int = -1
    ) -> Optional[str]:
        """
        Find the antecedent entity in context for given noun class.
        
        Args:
            noun_class: The class of noun (city, state, etc.)
            context: The full context
            answer_start: Position of answer in context (search before this)
            
        Returns:
            The antecedent entity or None
        """
        if noun_class not in self.noun_classes:
            return None
        
        class_info = self.noun_classes[noun_class]
        pattern = class_info.get('entity_pattern')
        
        if not pattern:
            return None
        
        # Search for entities in context
        search_text = context[:answer_start] if answer_start > 0 else context
        
        matches = list(re.finditer(pattern, search_text, re.UNICODE))
        
        if matches:
            # Return the most recent match (closest to the reference)
            return matches[-1].group(0)
        
        # Fallback: look for capitalized/named entities before common nouns
        # For Telugu, look for words ending in entity suffixes
        for suffix in class_info.get('entity_suffixes', []):
            suffix_pattern = rf'([ఁ-ౡా-ో్]+{re.escape(suffix)})'
            suffix_matches = list(re.finditer(suffix_pattern, search_text, re.UNICODE))
            if suffix_matches:
                return suffix_matches[-1].group(1)
        
        return None
    
    def resolve(
        self,
        answer: str,
        context: str,
        question: str = ""
    ) -> Optional[ResolvedReference]:
        """
        Attempt to resolve coreference in answer.
        
        Args:
            answer: The extracted answer
            context: The source context
            question: The question (for additional hints)
            
        Returns:
            ResolvedReference if resolution found, None otherwise
        """
        # Detect demonstrative reference
        detection = self.detect_demonstrative_reference(answer)
        if not detection:
            return None
        
        demo, noun, noun_class = detection
        
        # Find antecedent
        antecedent = self.find_antecedent(noun_class, context)
        if not antecedent:
            return None
        
        # Build resolved answer
        # Replace "ఈ నగరంలో" with "విజయవాడలో" etc.
        demo_noun = f"{demo} {noun}"
        if demo_noun.replace(' ', '') in answer.replace(' ', ''):
            demo_noun_normalized = demo_noun
        else:
            demo_noun_normalized = f"{demo}{noun}"  # Try without space
        
        # Transfer suffix from noun to antecedent
        # e.g., "ఈ నగరంలో" → antecedent + "లో"
        suffix = ""
        if noun.endswith('లో'):
            suffix = 'లో'
            base_antecedent = antecedent.rstrip('లో')
        elif noun.endswith('లోని'):
            suffix = 'లోని'
            base_antecedent = antecedent.rstrip('లోని')
        else:
            base_antecedent = antecedent
        
        resolved_entity = base_antecedent + suffix if suffix else antecedent
        
        # Replace in answer
        resolved_answer = answer
        # Try both with and without spaces
        for pattern in [f"{demo} {noun}", f"{demo}{noun}", demo_noun]:
            if pattern in resolved_answer:
                resolved_answer = resolved_answer.replace(pattern, resolved_entity, 1)
                break
        
        # Calculate confidence based on resolution quality
        confidence = 0.7
        if antecedent in context:
            confidence = 0.85
        if noun_class in ['city', 'state', 'country']:
            confidence += 0.05  # Higher confidence for place names
        
        return ResolvedReference(
            original=answer,
            resolved=resolved_answer,
            antecedent=antecedent,
            demonstrative=demo,
            noun_class=noun_class,
            confidence=min(1.0, confidence)
        )
    
    def get_alternative_answers(
        self,
        answer: str,
        context: str,
        question: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Generate alternative answer candidates by resolving references.
        
        Args:
            answer: Original answer
            context: Source context
            question: The question
            
        Returns:
            List of (alternative_answer, confidence_boost) tuples
        """
        alternatives = []
        
        resolution = self.resolve(answer, context, question)
        if resolution:
            alternatives.append((resolution.resolved, 0.1))
            alternatives.append((resolution.antecedent, 0.15))
        
        return alternatives


def resolve_coreference(answer: str, context: str, question: str = "") -> Tuple[str, List[str]]:
    """
    Convenience function to resolve coreference in an answer.
    
    Args:
        answer: The extracted answer
        context: The source context
        question: The question
        
    Returns:
        Tuple of (resolved_answer, [removed_references])
    """
    resolver = CoreferenceResolver()
    result = resolver.resolve(answer, context, question)
    
    if result:
        return result.resolved, [f"{result.demonstrative} {result.noun_class}"]
    
    return answer, []


# Export
__all__ = [
    'CoreferenceResolver',
    'ResolvedReference',
    'resolve_coreference',
    'DEMONSTRATIVES',
    'NOUN_CLASSES',
]
