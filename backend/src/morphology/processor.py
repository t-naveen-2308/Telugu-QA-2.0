"""
Telugu Morphology Processor

Main class for morphology-aware answer refinement in Telugu QA.
Combines multiple techniques:
1. Rule-based suffix trimming with priority ordering
2. Question-type aware refinement
3. Compound word normalization
4. Confidence re-scoring
5. Context-aware span adjustment
6. Character-level boundary detection
"""

from typing import Dict, List, Optional, Tuple, Set
import re
import unicodedata
from dataclasses import dataclass

from .suffix_patterns import (
    SuffixPattern, 
    get_all_suffixes, 
    get_suffixes_by_priority,
    PARTICIPIAL_ENDINGS,
    POSTPOSITIONS,
    CASE_MARKERS,
    HONORIFIC_SUFFIXES
)
from .question_rules import (
    detect_question_type, 
    get_rule_for_question,
    get_suffixes_to_remove,
    should_preserve_ending
)
from .compound_normalizer import CompoundNormalizer
from .coreference import CoreferenceResolver


@dataclass
class RefinementResult:
    """Result of morphology-aware answer refinement."""
    original_answer: str
    refined_answer: str
    removed_suffixes: List[str]
    confidence_adjustment: float
    question_type: str
    refinement_applied: bool
    notes: List[str]


class TeluguMorphologyProcessor:
    """
    Comprehensive morphology-aware answer refinement for Telugu QA.
    
    Applies multiple techniques to improve answer boundary detection
    and semantic completeness.
    """
    
    # Telugu character range for validation
    TELUGU_RANGE = range(0x0C00, 0x0C7F + 1)
    
    # Minimum remaining characters after trimming
    MIN_ANSWER_LENGTH = 2
    
    # Characters that indicate word boundaries
    WORD_BOUNDARY_CHARS = set(' \t\n।॥,.;:!?()[]{}"\'-')
    
    # Exceptions - words that should not be trimmed
    EXCEPTIONS: Set[str] = {
        'లో',      # When standalone (meaning "in")
        'తో',      # When standalone
        'గా',      # When standalone
        'కు',      # When standalone
        'ని',      # When standalone
    }
    
    def __init__(self, aggressive_mode: bool = False):
        """
        Initialize the morphology processor.
        
        The processor is designed to be model-independent - it works the same
        way regardless of which QA model generated the raw answer.
        
        Args:
            aggressive_mode: More aggressive suffix removal (removes priority 3 suffixes too)
        """
        self.aggressive_mode = aggressive_mode
        
        # Load all suffixes up to priority 3 (or 4 in aggressive mode)
        max_priority = 4 if aggressive_mode else 3
        self.suffixes = get_suffixes_by_priority(max_priority)
        
        # Initialize compound normalizer
        self.compound_normalizer = CompoundNormalizer()
        
        # Initialize coreference resolver
        self.coreference_resolver = CoreferenceResolver()
    
    def refine_answer(
        self,
        answer: str,
        question: str,
        context: str,
        confidence: float = 1.0
    ) -> RefinementResult:
        """
        Apply morphology-aware refinement to an extracted answer.
        
        Args:
            answer: The predicted answer span
            question: The question (used for question-type rules)
            context: The source context (for validation)
            confidence: Original confidence score
            
        Returns:
            RefinementResult with refined answer and metadata
        """
        if not answer or not answer.strip():
            return RefinementResult(
                original_answer=answer,
                refined_answer=answer,
                removed_suffixes=[],
                confidence_adjustment=0.0,
                question_type='unknown',
                refinement_applied=False,
                notes=['Empty answer']
            )
        
        # Normalize Unicode
        answer = unicodedata.normalize('NFC', answer.strip())
        context = unicodedata.normalize('NFC', context) if context else ""
        question = unicodedata.normalize('NFC', question) if question else ""
        
        notes = []
        removed_suffixes = []
        refinement_applied = False
        
        # Detect question type - this drives which suffixes to remove
        question_type = detect_question_type(question)
        
        # Initialize refined answer
        refined = answer
        coref_applied = False
        coref_confidence = 0.0
        
        # Step 1: Apply coreference resolution (before suffix trimming)
        if self.coreference_resolver:
            coref_result = self.coreference_resolver.resolve(answer, context, question)
            if coref_result:
                refined = coref_result.resolved
                refinement_applied = True
                coref_applied = True
                coref_confidence = coref_result.confidence
                notes.append(f"Resolved '{coref_result.demonstrative} {coref_result.noun_class}' to '{coref_result.antecedent}'")
        
        # Step 2: Apply suffix trimming based on question type
        refined, removed = self._apply_suffix_trimming(
            refined, question, question_type
        )
        if removed:
            removed_suffixes.extend(removed)
            refinement_applied = True
            notes.append(f"Trimmed suffixes: {', '.join(removed)}")
        
        # Step 3: Apply compound normalization
        self.compound_normalizer.context = context
        compound_refined = self.compound_normalizer.normalize(refined, context)
        if compound_refined != refined:
            refined = compound_refined
            refinement_applied = True
            notes.append("Applied compound normalization")
        
        # Step 4: Context validation (try to find exact match)
        answer_in_context = False
        if context:
            context_validated = self._validate_in_context(refined, context)
            if context_validated:
                answer_in_context = True
                if context_validated != refined:
                    refined = context_validated
                    notes.append("Adjusted to match context form")
        
        # Step 5: Confidence adjustment based on refinements
        confidence_adjustment = self._calculate_confidence_adjustment(
            answer, refined, removed_suffixes, coref_applied, coref_confidence,
            answer_in_context
        )
        if confidence_adjustment != 0.0:
            notes.append(f"Confidence adjustment: {confidence_adjustment:+.2f}")
        
        return RefinementResult(
            original_answer=answer,
            refined_answer=refined,
            removed_suffixes=removed_suffixes,
            confidence_adjustment=confidence_adjustment,
            question_type=question_type,
            refinement_applied=refinement_applied,
            notes=notes
        )
    
    def _apply_suffix_trimming(
        self,
        answer: str,
        question: str,
        question_type: str
    ) -> Tuple[str, List[str]]:
        """
        Apply rule-based suffix trimming.
        
        Args:
            answer: Answer to trim
            question: Original question
            question_type: Detected question type
            
        Returns:
            Tuple of (trimmed_answer, list_of_removed_suffixes)
        """
        removed = []
        current = answer
        
        # Get question-specific suffixes to remove (always use question-aware rules)
        question_suffixes = get_suffixes_to_remove(question, self.aggressive_mode)
        
        # Check for preserved endings first (e.g., name suffixes like రావు, రెడ్డి)
        if should_preserve_ending(answer, question):
            return answer, []
        
        # Sort suffixes by length (longest first) for greedy matching
        # This ensures "గారు" matches before "ారు"
        sorted_suffixes = sorted(self.suffixes, key=lambda x: -len(x.telugu))
        
        # Apply suffix trimming iteratively (handle multiple suffixes like "గారిని")
        max_iterations = 5  # Prevent infinite loops
        for _ in range(max_iterations):
            trimmed = False
            best_match = None
            best_suffix_len = 0
            
            # Find the longest matching suffix
            for suffix in sorted_suffixes:
                # Skip if answer is too short
                if len(current) <= suffix.min_remaining + len(suffix.telugu):
                    continue
                
                # Check if answer ends with this suffix
                if current.endswith(suffix.telugu):
                    # Prefer longer suffixes
                    if len(suffix.telugu) > best_suffix_len:
                        # Suffix should be removed if:
                        # 1. It's in the question-specific removal set, OR
                        # 2. It's a high-priority suffix (priority 1-2), OR
                        # 3. Aggressive mode is on and it's priority 3
                        should_remove = (
                            suffix.telugu in question_suffixes or
                            suffix.priority <= 2 or
                            (self.aggressive_mode and suffix.priority <= 3)
                        )
                        if should_remove:
                            best_match = suffix
                            best_suffix_len = len(suffix.telugu)
            
            # Apply the best (longest) match found
            if best_match:
                potential = current[:-len(best_match.telugu)]
                
                # Validate remaining is valid Telugu
                if self._is_valid_remainder(potential):
                    current = potential
                    removed.append(best_match.telugu)
                    trimmed = True
            
            if not trimmed:
                break
        
        return current.strip(), removed
    
    def _is_valid_remainder(self, text: str) -> bool:
        """
        Check if the remaining text after trimming is valid.
        
        Args:
            text: Remaining text
            
        Returns:
            True if valid, False otherwise
        """
        if not text or len(text) < self.MIN_ANSWER_LENGTH:
            return False
        
        text = text.strip()
        
        # Check if text is in exceptions (should not be stripped further)
        if text in self.EXCEPTIONS:
            return False
        
        # Check if ends with a valid Telugu character
        if text:
            last_char = text[-1]
            if ord(last_char) in self.TELUGU_RANGE:
                return True
            # Also allow digits and common punctuation
            if last_char.isdigit():
                return True
        
        return len(text) >= self.MIN_ANSWER_LENGTH
    
    def _validate_in_context(self, answer: str, context: str) -> Optional[str]:
        """
        Find the best matching form of the answer in the context.
        
        Args:
            answer: Refined answer
            context: Source context
            
        Returns:
            Context-matched form or None
        """
        if not context:
            return None
        
        # Direct match
        if answer in context:
            return answer
        
        # Try without trailing/leading spaces
        answer_stripped = answer.strip()
        if answer_stripped in context:
            return answer_stripped
        
        # Try finding the answer as part of a word boundary match
        # This helps with tokenization artifacts
        pattern = re.escape(answer_stripped)
        matches = list(re.finditer(pattern, context))
        if matches:
            return answer_stripped
        
        # Try case variations (for mixed script)
        # Telugu doesn't have case, but this helps with transliterated content
        
        return answer_stripped
    
    def _calculate_confidence_adjustment(
        self,
        original: str,
        refined: str,
        removed_suffixes: List[str],
        coref_applied: bool = False,
        coref_confidence: float = 0.0,
        answer_in_context: bool = False
    ) -> float:
        """
        Calculate confidence adjustment based on refinement.
        
        Boost confidence for good refinements, penalize dubious ones.
        
        Args:
            original: Original answer
            refined: Refined answer
            removed_suffixes: List of removed suffixes
            coref_applied: Whether coreference resolution was applied
            coref_confidence: Confidence from coreference resolution (0-1)
            answer_in_context: Whether the answer was validated in context
            
        Returns:
            Confidence adjustment (-1.0 to +1.0)
        """
        adjustment = 0.0
        
        # Boost for successful coreference resolution
        if coref_applied:
            # Scale by coreference confidence (typically 0.7-0.95)
            adjustment += 0.08 * coref_confidence
        
        # Process suffix removals
        if removed_suffixes:
            # Boost for removing participial endings (high confidence refinement)
            participial = [p.telugu for p in PARTICIPIAL_ENDINGS]
            for suffix in removed_suffixes:
                if suffix in participial:
                    adjustment += 0.05
            
            # Moderate boost for postpositions
            postpos = [p.telugu for p in POSTPOSITIONS]
            for suffix in removed_suffixes:
                if suffix in postpos:
                    adjustment += 0.03
            
            # Moderate boost for honorific suffixes (common in person names)
            honorific = [p.telugu for p in HONORIFIC_SUFFIXES]
            for suffix in removed_suffixes:
                if suffix in honorific:
                    adjustment += 0.04
            
            # Small boost for case markers
            case_markers = [p.telugu for p in CASE_MARKERS]
            for suffix in removed_suffixes:
                if suffix in case_markers:
                    adjustment += 0.02
            
            # Slight penalty for aggressive trimming (too much removed)
            removal_ratio = 1 - (len(refined) / len(original)) if original else 0
            if removal_ratio > 0.5:  # Removed more than 50%
                adjustment -= 0.1
        
        # Cap adjustment
        return max(-0.2, min(0.15, adjustment))
    
    def batch_refine(
        self,
        predictions: List[Dict],
        questions: List[str],
        contexts: List[str]
    ) -> List[RefinementResult]:
        """
        Refine a batch of predictions.
        
        Args:
            predictions: List of prediction dicts with 'answer' and 'score'
            questions: List of questions
            contexts: List of contexts
            
        Returns:
            List of RefinementResults
        """
        results = []
        
        for pred, question, context in zip(predictions, questions, contexts):
            answer = pred.get('answer', '')
            confidence = pred.get('score', 1.0)
            
            result = self.refine_answer(answer, question, context, confidence)
            results.append(result)
        
        return results
    
    def analyze_morphology_errors(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: List[str]
    ) -> Dict:
        """
        Analyze morphology-related errors in predictions.
        
        Useful for understanding where refinement can help.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            questions: List of questions
            
        Returns:
            Analysis dictionary with error statistics
        """
        analysis = {
            'total': len(predictions),
            'over_extraction': 0,
            'under_extraction': 0,
            'suffix_errors': {},
            'question_type_errors': {},
            'compound_errors': 0,
            'examples': {
                'over_extraction': [],
                'under_extraction': [],
                'compound': []
            }
        }
        
        for pred, gold, question in zip(predictions, ground_truths, questions):
            qtype = detect_question_type(question)
            
            # Check for over-extraction (pred contains gold + extra)
            if gold in pred and len(pred) > len(gold):
                analysis['over_extraction'] += 1
                extra = pred.replace(gold, '', 1).strip()
                
                # Categorize the extra characters
                for suffix in self.suffixes:
                    if extra.endswith(suffix.telugu) or extra.startswith(suffix.telugu):
                        analysis['suffix_errors'][suffix.telugu] = \
                            analysis['suffix_errors'].get(suffix.telugu, 0) + 1
                
                # Track by question type
                analysis['question_type_errors'][qtype] = \
                    analysis['question_type_errors'].get(qtype, 0) + 1
                
                if len(analysis['examples']['over_extraction']) < 10:
                    analysis['examples']['over_extraction'].append({
                        'pred': pred,
                        'gold': gold,
                        'extra': extra,
                        'qtype': qtype
                    })
            
            # Check for under-extraction (gold contains pred + extra)
            elif pred in gold and len(gold) > len(pred):
                analysis['under_extraction'] += 1
                if len(analysis['examples']['under_extraction']) < 10:
                    analysis['examples']['under_extraction'].append({
                        'pred': pred,
                        'gold': gold,
                        'qtype': qtype
                    })
            
            # Check for compound word mismatches
            if pred.replace(' ', '') == gold or gold.replace(' ', '') == pred:
                analysis['compound_errors'] += 1
                if len(analysis['examples']['compound']) < 10:
                    analysis['examples']['compound'].append({
                        'pred': pred,
                        'gold': gold
                    })
        
        # Calculate rates
        total = analysis['total']
        analysis['over_extraction_rate'] = analysis['over_extraction'] / total if total > 0 else 0
        analysis['under_extraction_rate'] = analysis['under_extraction'] / total if total > 0 else 0
        analysis['compound_error_rate'] = analysis['compound_errors'] / total if total > 0 else 0
        
        return analysis


# Convenience function for simple use
def refine_telugu_answer(
    answer: str,
    question: str = "",
    context: str = "",
    confidence: float = 1.0
) -> str:
    """
    Quick function to refine a Telugu answer.
    
    Args:
        answer: The predicted answer
        question: The question (optional, improves refinement)
        context: The source context (optional, for validation)
        confidence: Original confidence score
        
    Returns:
        Refined answer string
    """
    processor = TeluguMorphologyProcessor()
    result = processor.refine_answer(answer, question, context, confidence)
    return result.refined_answer
