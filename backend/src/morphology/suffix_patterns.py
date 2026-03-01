"""
Telugu Suffix Patterns for Morphology-Aware Answer Refinement

Comprehensive collection of Telugu morphological suffixes organized by category.
Each suffix includes the Telugu form, transliteration, meaning, and priority.

Priority levels (lower = higher priority for trimming):
  1 = Always trim (participial, rarely valid in answers)
  2 = Usually trim (postpositions, case markers)
  3 = Sometimes trim (depends on question type)
  4 = Rarely trim (may be part of valid answer)
"""

from typing import Dict, List, Tuple, NamedTuple


class SuffixPattern(NamedTuple):
    """Represents a Telugu suffix pattern."""
    telugu: str           # Telugu suffix
    transliteration: str  # ITRANS/ISO transliteration
    meaning: str          # English meaning
    priority: int         # Trimming priority (1=highest)
    min_remaining: int    # Minimum chars that must remain after trim


# =============================================================================
# POSTPOSITIONS (విభక్తి ప్రత్యయాలు)
# Indicate grammatical relationships - usually should be trimmed
# =============================================================================

POSTPOSITIONS: List[SuffixPattern] = [
    # Compound postpositions (longer ones first for greedy matching)
    SuffixPattern('నుండి', 'nuṇḍi', 'from', 2, 2),
    SuffixPattern('నుంచి', 'nuṃci', 'from (variant)', 2, 2),
    SuffixPattern('కోసం', 'kōsaṃ', 'for the purpose of', 2, 2),
    SuffixPattern('గురించి', 'guriñci', 'about/regarding', 2, 2),
    SuffixPattern('వరకు', 'varaku', 'until/up to', 2, 2),
    SuffixPattern('ద్వారా', 'dvārā', 'through/via', 2, 2),
    SuffixPattern('ప్రకారం', 'prakāraṃ', 'according to', 2, 2),
    SuffixPattern('వల్ల', 'valla', 'because of', 2, 2),
    SuffixPattern('పట్ల', 'paṭla', 'towards', 2, 2),
    SuffixPattern('వద్ద', 'vadda', 'near/at', 2, 2),
    SuffixPattern('దగ్గర', 'daggara', 'near', 2, 2),
    SuffixPattern('బదులు', 'badulu', 'instead of', 2, 2),
    SuffixPattern('మధ్య', 'madhya', 'between', 2, 2),
    SuffixPattern('తరువాత', 'taruvāta', 'after', 2, 2),
    SuffixPattern('ముందు', 'muṃdu', 'before', 2, 2),
    
    # Locative postpositions
    SuffixPattern('లోని', 'lōni', 'that which is in', 2, 2),
    SuffixPattern('లోకి', 'lōki', 'into', 2, 2),
    SuffixPattern('లోంచి', 'lōṃci', 'from within', 2, 2),
    SuffixPattern('లో', 'lō', 'in/inside', 2, 2),
    SuffixPattern('పైన', 'paina', 'on top of', 2, 2),
    SuffixPattern('పై', 'pai', 'on', 2, 2),
    SuffixPattern('మీద', 'mīda', 'on/above', 2, 2),
    SuffixPattern('కింద', 'kiṃda', 'under', 2, 2),
    SuffixPattern('క్రింద', 'krinda', 'below', 2, 2),
    
    # Directional
    SuffixPattern('వైపు', 'vaipu', 'towards', 2, 2),
    SuffixPattern('వేపు', 'vēpu', 'direction of', 2, 2),
]

# =============================================================================
# CASE MARKERS (విభక్తులు)
# Shorter suffixes that mark grammatical case
# =============================================================================

CASE_MARKERS: List[SuffixPattern] = [
    # Instrumental case
    SuffixPattern('తో', 'tō', 'with/by', 2, 2),
    SuffixPattern('చేత', 'cēta', 'by (agent)', 2, 2),
    
    # Dative case - priority 2 since these rarely belong in extracted answers
    SuffixPattern('కు', 'ku', 'to/for', 2, 2),
    SuffixPattern('కి', 'ki', 'to/for (variant)', 2, 2),
    
    # Accusative case - priority 2 since these rarely belong in extracted answers
    SuffixPattern('ని', 'ni', 'object marker', 2, 2),
    SuffixPattern('ను', 'nu', 'object marker (variant)', 2, 2),
    
    # Genitive case
    SuffixPattern('యొక్క', 'yokka', 'of/belonging to', 2, 2),
    
    # Sociative
    SuffixPattern('తోపాటు', 'tōpāṭu', 'along with', 2, 2),
]

# =============================================================================
# VERB SUFFIXES (క్రియా ప్రత్యయాలు)
# Verbal endings - often indicate over-extraction
# =============================================================================

VERB_SUFFIXES: List[SuffixPattern] = [
    # Past tense markers
    SuffixPattern('ఆడు', 'āḍu', 'he did (past)', 2, 3),
    SuffixPattern('ఆరు', 'āru', 'they did (past)', 2, 3),
    SuffixPattern('ింది', 'iṃdi', 'it happened (past)', 2, 3),
    SuffixPattern('ాడు', 'āḍu', 'he did (simplified)', 2, 3),
    SuffixPattern('ారు', 'āru', 'they did (simplified)', 2, 3),
    
    # Present tense
    SuffixPattern('తాడు', 'tāḍu', 'he does', 2, 3),
    SuffixPattern('తారు', 'tāru', 'they do', 2, 3),
    SuffixPattern('తుంది', 'tuṃdi', 'it does', 2, 3),
    SuffixPattern('తున్నాడు', 'tunnāḍu', 'he is doing', 2, 3),
    SuffixPattern('తున్నారు', 'tunnāru', 'they are doing', 2, 3),
    
    # Future tense
    SuffixPattern('తాను', 'tānu', 'I will', 2, 3),
    SuffixPattern('తాము', 'tāmu', 'we will', 2, 3),
    
    # Gerunds/Verbal nouns
    SuffixPattern('డం', 'ḍaṃ', 'gerund (-ing)', 3, 3),
    SuffixPattern('టం', 'ṭaṃ', 'gerund (variant)', 3, 3),
    SuffixPattern('డానికి', 'ḍāniki', 'for doing', 2, 3),
]

# =============================================================================
# PARTICIPIAL ENDINGS (క్రియాజ ప్రత్యయాలు)
# Very common cause of over-extraction in QA
# =============================================================================

PARTICIPIAL_ENDINGS: List[SuffixPattern] = [
    # Past participles (very common in over-extractions)
    SuffixPattern('బడిన', 'baḍina', 'that was (passive)', 1, 3),
    SuffixPattern('చేయబడిన', 'cēyabaḍina', 'that was done', 1, 4),
    SuffixPattern('అయిన', 'ayina', 'that is/was', 1, 3),
    SuffixPattern('చేసిన', 'cēsina', 'having done', 1, 3),
    SuffixPattern('ైన', 'aina', 'that is (contracted)', 1, 2),
    SuffixPattern('ిన', 'ina', 'past participle marker', 2, 2),
    
    # Present/continuous participles
    SuffixPattern('తున్న', 'tunna', 'that is doing', 1, 3),
    SuffixPattern('తూన్న', 'tūnna', 'that is doing (variant)', 1, 3),
    SuffixPattern('ే', 'ē', 'present participle', 3, 2),
    
    # Potential participles
    SuffixPattern('గల', 'gala', 'able to/capable', 2, 2),
    SuffixPattern('వలసిన', 'valasina', 'that should be', 1, 3),
    SuffixPattern('దగ్గ', 'dagga', 'worthy of', 2, 3),
]

# =============================================================================
# ADVERBIAL ENDINGS
# Convert nouns/adjectives to adverbs
# =============================================================================

ADVERBIAL_ENDINGS: List[SuffixPattern] = [
    SuffixPattern('గా', 'gā', 'as/in the manner of', 2, 2),
    SuffixPattern('గు', 'gu', 'adverbial variant', 3, 2),
    SuffixPattern('మైన', 'maina', 'that is (adjectival)', 2, 3),
    SuffixPattern('అని', 'ani', 'that (quotative)', 3, 2),
    SuffixPattern('అనే', 'anē', 'called/named', 3, 2),
]

# =============================================================================
# PLURAL MARKERS
# Usually safe in answers, but sometimes over-extend
# =============================================================================

PLURAL_MARKERS: List[SuffixPattern] = [
    SuffixPattern('లు', 'lu', 'plural', 4, 2),
    SuffixPattern('ల', 'la', 'genitive plural', 4, 2),
    SuffixPattern('లను', 'lanu', 'plural accusative', 3, 2),
    SuffixPattern('లకు', 'laku', 'plural dative', 3, 2),
    SuffixPattern('లలో', 'lalō', 'plural locative', 3, 2),
    SuffixPattern('లతో', 'latō', 'plural instrumental', 3, 2),
]

# =============================================================================
# HONORIFIC SUFFIXES
# Should be trimmed for person names
# =============================================================================

HONORIFIC_SUFFIXES: List[SuffixPattern] = [
    SuffixPattern('గారు', 'gāru', 'respectful suffix', 3, 2),
    SuffixPattern('వారు', 'vāru', 'honorific plural', 3, 2),
    SuffixPattern('గారి', 'gāri', 'possessive honorific', 3, 2),
    SuffixPattern('వారి', 'vāri', 'possessive honorific plural', 3, 2),
    SuffixPattern('మహారాజ్', 'mahārāj', 'king title', 4, 3),
    SuffixPattern('రావు', 'rāvu', 'title suffix', 4, 3),
    SuffixPattern('రెడ్డి', 'reḍḍi', 'caste title', 4, 3),
    SuffixPattern('నాయుడు', 'nāyuḍu', 'caste title', 4, 3),
]

# =============================================================================
# SPECIAL ENDINGS (Context-dependent)
# =============================================================================

SPECIAL_ENDINGS: List[SuffixPattern] = [
    # Emphatic particles
    SuffixPattern('ే', 'ē', 'emphatic/only', 4, 2),
    SuffixPattern('యే', 'yē', 'emphatic', 4, 2),
    
    # Quotative
    SuffixPattern('అని', 'ani', 'quotative', 3, 2),
    SuffixPattern('అంటే', 'aṃṭē', 'meaning/if', 3, 2),
    
    # Negative
    SuffixPattern('కాదు', 'kādu', 'is not', 3, 3),
    SuffixPattern('లేదు', 'lēdu', 'is not/does not exist', 3, 3),
]


# =============================================================================
# AGGREGATED SUFFIX LIST (Ordered by priority and length)
# =============================================================================

def get_all_suffixes() -> List[SuffixPattern]:
    """
    Get all suffixes sorted by priority (ascending) and length (descending).
    This ensures longer, higher-priority suffixes are checked first.
    """
    all_suffixes = (
        PARTICIPIAL_ENDINGS +
        POSTPOSITIONS +
        CASE_MARKERS +
        VERB_SUFFIXES +
        ADVERBIAL_ENDINGS +
        HONORIFIC_SUFFIXES +
        PLURAL_MARKERS +
        SPECIAL_ENDINGS
    )
    
    # Sort by priority (lower first), then by length (longer first for greedy match)
    return sorted(all_suffixes, key=lambda x: (x.priority, -len(x.telugu)))


def get_suffixes_by_priority(max_priority: int = 3) -> List[SuffixPattern]:
    """Get suffixes up to a certain priority level."""
    return [s for s in get_all_suffixes() if s.priority <= max_priority]


# Precomputed sorted list
ALL_SUFFIXES_SORTED = get_all_suffixes()
