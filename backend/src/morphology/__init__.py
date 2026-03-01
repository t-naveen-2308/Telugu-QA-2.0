"""
Telugu Morphology Module

Provides morphology-aware answer refinement for Telugu QA systems.
"""

from .processor import TeluguMorphologyProcessor
from .suffix_patterns import (
    POSTPOSITIONS,
    CASE_MARKERS,
    VERB_SUFFIXES,
    PARTICIPIAL_ENDINGS,
    ADVERBIAL_ENDINGS,
    PLURAL_MARKERS,
    HONORIFIC_SUFFIXES
)
from .question_rules import QUESTION_TYPE_RULES
from .compound_normalizer import CompoundNormalizer
from .coreference import CoreferenceResolver, resolve_coreference

__all__ = [
    'TeluguMorphologyProcessor',
    'CompoundNormalizer',
    'CoreferenceResolver',
    'resolve_coreference',
    'POSTPOSITIONS',
    'CASE_MARKERS',
    'VERB_SUFFIXES',
    'PARTICIPIAL_ENDINGS',
    'ADVERBIAL_ENDINGS',
    'PLURAL_MARKERS',
    'HONORIFIC_SUFFIXES',
    'QUESTION_TYPE_RULES'
]
