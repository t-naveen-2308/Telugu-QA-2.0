"""
Telugu Compound Word Normalizer

Handles spacing inconsistencies in Telugu compound words.
Many Telugu proper nouns and compound words can be written with or without spaces.
This module normalizes predictions to match ground truth format.
"""

from typing import Dict, List, Set, Optional, Tuple
import re
import unicodedata


class CompoundNormalizer:
    """
    Normalizes Telugu compound words for consistent matching.
    
    Handles cases like:
      - "నాగార్జున సాగర్" → "నాగార్జునసాగర్"
      - "హైదర్ ఆబాద్" → "హైదరాబాద్"
      - "ఆంధ్ర ప్రదేశ్" → "ఆంధ్రప్రదేశ్"
    """
    
    # Common Telugu compound words with their normalized forms
    COMPOUND_MAPPINGS: Dict[str, str] = {
        # Place names
        'నాగార్జున సాగర్': 'నాగార్జునసాగర్',
        'హైదర్ ఆబాద్': 'హైదరాబాద్',
        'హైదరా బాద్': 'హైదరాబాద్',
        'విజయ వాడ': 'విజయవాడ',
        'విశాఖ పట్నం': 'విశాఖపట్నం',
        'విశాఖ పట్టణం': 'విశాఖపట్టణం',
        'రాజ మహేంద్రవరం': 'రాజమహేంద్రవరం',
        'రాజ మండ్రి': 'రాజమండ్రి',
        'తిరు పతి': 'తిరుపతి',
        'అమరా వతి': 'అమరావతి',
        'కర్నూలు': 'కర్నూలు',  # Already correct but ensure consistency
        'నెల్లూరు': 'నెల్లూరు',
        'గుంటూరు': 'గుంటూరు',
        
        # State names
        'ఆంధ్ర ప్రదేశ్': 'ఆంధ్రప్రదేశ్',
        'తెలం గాణ': 'తెలంగాణ',
        'తెలంగా ణ': 'తెలంగాణ',
        'మహా రాష్ట్ర': 'మహారాష్ట్ర',
        'కర్ణా టక': 'కర్ణాటక',
        'తమిళ నాడు': 'తమిళనాడు',
        'ఉత్తర ప్రదేశ్': 'ఉత్తరప్రదేశ్',
        'పశ్చిమ బెంగాల్': 'పశ్చిమబెంగాల్',
        'మధ్య ప్రదేశ్': 'మధ్యప్రదేశ్',
        
        # Common compound nouns
        'దక్కను పీఠభూమి': 'దక్కనుపీఠభూమి',
        'సంస్కృత భాష': 'సంస్కృతభాష',
        'తెలుగు భాష': 'తెలుగుభాష',
        'భారత దేశం': 'భారతదేశం',
        'భారత దేశము': 'భారతదేశము',
        'ప్రపంచ యుద్ధం': 'ప్రపంచయుద్ధం',
        'స్వాతంత్ర్య సంగ్రామం': 'స్వాతంత్ర్యసంగ్రామం',
        'రాజ కీయం': 'రాజకీయం',
        'వైద్య శాల': 'వైద్యశాల',
        'పాఠ శాల': 'పాఠశాల',
        'విశ్వ విద్యాలయం': 'విశ్వవిద్యాలయం',
        'ఉస్మానియా విశ్వవిద్యాలయం': 'ఉస్మానియా విశ్వవిద్యాలయం',  # Keep space
        
        # Historical terms
        'మంగోల్ సామ్రాజ్యం': 'మంగోల్‌సామ్రాజ్యం',
        'బ్రిటిష్ పాలన': 'బ్రిటిష్‌పాలన',
        'ముఘల్ సామ్రాజ్యం': 'ముఘల్‌సామ్రాజ్యం',
        'విజయ నగర సామ్రాజ్యం': 'విజయనగరసామ్రాజ్యం',
        
        # Numbers with units (preserve space)
        # These should NOT be merged
    }
    
    # Reverse mapping for splitting
    SPLIT_MAPPINGS: Dict[str, str] = {v: k for k, v in COMPOUND_MAPPINGS.items()}
    
    # Common suffixes that indicate compound words
    COMPOUND_SUFFIXES = [
        'పురం', 'పురి', 'నగర్', 'నగరం', 'బాద్', 'పట్నం', 'పట్టణం', 
        'పేట', 'గూడెం', 'వారి', 'వరం', 'సాగర్', 'ప్రదేశ్',
        'రాష్ట్రం', 'రాజ్యం', 'సామ్రాజ్యం', 'యుద్ధం', 'శాల',
        'విద్యాలయం', 'భాష', 'దేశం', 'దేశము'
    ]
    
    # Common prefixes in compound words
    COMPOUND_PREFIXES = [
        'మహా', 'పెద్ద', 'చిన్న', 'తెలుగు', 'సంస్కృత', 'ప్రాచీన',
        'నూతన', 'ప్రపంచ', 'రాష్ట్ర', 'జాతీయ', 'అంతర్జాతీయ'
    ]
    
    def __init__(self, context: Optional[str] = None):
        """
        Initialize the compound normalizer.
        
        Args:
            context: Optional context to extract compound patterns from
        """
        self.context = context
        self._context_compounds: Set[str] = set()
        
        if context:
            self._extract_context_compounds()
    
    def _extract_context_compounds(self):
        """Extract potential compound words from context."""
        if not self.context:
            return
        
        # Find words in context that match compound patterns
        words = self.context.split()
        
        for i in range(len(words) - 1):
            combined = words[i] + words[i + 1]
            
            # Check if combining makes a known compound
            for suffix in self.COMPOUND_SUFFIXES:
                if combined.endswith(suffix):
                    self._context_compounds.add((words[i], words[i + 1], combined))
                    break
    
    def normalize(self, text: str, context: Optional[str] = None) -> str:
        """
        Normalize compound words in the text.
        
        Tries both joining and splitting to match the format in context.
        
        Args:
            text: Text to normalize
            context: Optional context for format matching
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # Use instance context if none provided
        ctx = context or self.context
        
        # First, apply known mappings
        normalized = text
        
        # Check if text matches a known compound (with space)
        if text in self.COMPOUND_MAPPINGS:
            candidate = self.COMPOUND_MAPPINGS[text]
            # Prefer the form that exists in context
            if ctx and candidate in ctx:
                return candidate
            elif ctx and text in ctx:
                return text
            # Default to joined form
            return candidate
        
        # Check if text is a joined compound that should be split
        if text in self.SPLIT_MAPPINGS:
            candidate = self.SPLIT_MAPPINGS[text]
            if ctx and candidate in ctx:
                return candidate
            # Keep joined if that's what's in context
            if ctx and text in ctx:
                return text
            # Default to joined
            return text
        
        # Try to match with context format
        if ctx:
            normalized = self._match_context_format(text, ctx)
        
        return normalized
    
    def _match_context_format(self, text: str, context: str) -> str:
        """
        Adjust text format to match how it appears in context.
        
        Args:
            text: Answer text
            context: Context passage
            
        Returns:
            Text formatted to match context
        """
        # If text exists exactly in context, keep it
        if text in context:
            return text
        
        # Try joining (remove space)
        joined = text.replace(' ', '')
        if joined in context:
            return joined
        
        # Try with zero-width joiner for compound words
        parts = text.split()
        if len(parts) == 2:
            zwj_joined = parts[0] + '\u200c' + parts[1]  # ZWNJ
            if zwj_joined in context:
                return zwj_joined
        
        # Try common joining patterns
        for space_form, joined_form in self.COMPOUND_MAPPINGS.items():
            if text.startswith(space_form):
                potential = text.replace(space_form, joined_form, 1)
                if potential in context:
                    return potential
        
        # Return original if no match found
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Apply Unicode normalization (NFC) for consistent comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            NFC-normalized text
        """
        return unicodedata.normalize('NFC', text)
    
    def remove_zero_width_chars(self, text: str) -> str:
        """
        Remove zero-width Unicode characters that may affect matching.
        
        Args:
            text: Text to clean
            
        Returns:
            Text without zero-width characters
        """
        # Remove ZWNJ, ZWJ, and other zero-width chars
        zero_width = '\u200b\u200c\u200d\ufeff'
        return ''.join(c for c in text if c not in zero_width)
    
    def get_all_forms(self, text: str) -> List[str]:
        """
        Get all possible forms of a compound word (joined/split).
        
        Useful for fuzzy matching during evaluation.
        
        Args:
            text: Text to expand
            
        Returns:
            List of possible forms
        """
        forms = [text]
        
        # Add joined form
        joined = text.replace(' ', '')
        if joined != text:
            forms.append(joined)
        
        # Add known mappings
        if text in self.COMPOUND_MAPPINGS:
            forms.append(self.COMPOUND_MAPPINGS[text])
        if text in self.SPLIT_MAPPINGS:
            forms.append(self.SPLIT_MAPPINGS[text])
        
        # Unicode normalized forms
        forms.append(self.normalize_unicode(text))
        forms.append(self.remove_zero_width_chars(text))
        
        return list(set(forms))


def compute_compound_similarity(pred: str, gold: str) -> float:
    """
    Compute similarity considering compound word variations.
    
    Returns 1.0 if any form matches, otherwise 0.0.
    
    Args:
        pred: Predicted answer
        gold: Ground truth answer
        
    Returns:
        Similarity score (0.0 or 1.0)
    """
    normalizer = CompoundNormalizer()
    
    pred_forms = normalizer.get_all_forms(pred)
    gold_forms = normalizer.get_all_forms(gold)
    
    # Check if any forms match
    for pf in pred_forms:
        for gf in gold_forms:
            if pf == gf:
                return 1.0
    
    return 0.0
