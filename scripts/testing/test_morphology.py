# -*- coding: utf-8 -*-
"""
Test script for Telugu Morphology Module
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.morphology.processor import TeluguMorphologyProcessor, refine_telugu_answer
from src.morphology.suffix_patterns import get_all_suffixes, POSTPOSITIONS, PARTICIPIAL_ENDINGS
from src.morphology.question_rules import detect_question_type, QUESTION_TYPE_RULES
from src.morphology.compound_normalizer import CompoundNormalizer


def test_suffix_patterns():
    """Test suffix pattern loading."""
    print("\n" + "=" * 60)
    print("TEST 1: Suffix Patterns")
    print("=" * 60)
    
    all_suffixes = get_all_suffixes()
    print(f"Total suffixes loaded: {len(all_suffixes)}")
    print(f"Postpositions: {len(POSTPOSITIONS)}")
    print(f"Participial endings: {len(PARTICIPIAL_ENDINGS)}")
    
    print("\nSample postpositions (first 5):")
    for p in POSTPOSITIONS[:5]:
        print(f"  {p.telugu} ({p.transliteration}) - {p.meaning}, priority={p.priority}")
    
    print("\nSample participial endings:")
    for p in PARTICIPIAL_ENDINGS[:5]:
        print(f"  {p.telugu} ({p.transliteration}) - {p.meaning}, priority={p.priority}")
    
    return True


def test_question_type_detection():
    """Test question type detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Question Type Detection")
    print("=" * 60)
    
    print("Available question types:")
    for qtype, rule in QUESTION_TYPE_RULES.items():
        print(f"  {qtype}: {rule.question_words[:3]}...")
    
    test_questions = [
        ("ఎవరు ఈ పుస్తకం రాశారు?", "who"),
        ("హైదరాబాద్ ఎక్కడ ఉంది?", "where"),
        ("భారతదేశం ఎప్పుడు స్వతంత్రం పొందింది?", "when"),
        ("జనాభా ఎంత?", "how_much"),
        ("రాజధాని ఏమిటి?", "what"),
        ("ఎందుకు యుద్ధం జరిగింది?", "why"),
        ("ఎలా నిర్మించారు?", "how"),
    ]
    
    print("\nQuestion type detection results:")
    all_passed = True
    for question, expected in test_questions:
        detected = detect_question_type(question)
        status = "✓" if detected == expected else "✗"
        if detected != expected:
            all_passed = False
        print(f"  {status} '{question[:40]}...' -> {detected} (expected: {expected})")
    
    return all_passed


def test_suffix_trimming():
    """Test suffix trimming functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Suffix Trimming")
    print("=" * 60)
    
    processor = TeluguMorphologyProcessor()
    
    test_cases = [
        # (answer, question, context, expected_refined, expected_removed)
        ("హైదరాబాద్లో", "తెలంగాణ రాజధాని ఎక్కడ?", "హైదరాబాద్లో తెలంగాణ ప్రభుత్వం ఉంది.", "హైదరాబాద్", ["లో"]),
        ("రాముడికి", "ఎవరికి ఇచ్చారు?", "రాముడికి ఇచ్చారు.", "రాముడి", ["కి"]),
        ("నిర్మించబడిన", "ఏమి జరిగింది?", "నిర్మించబడిన కోట.", "నిర్మించ", ["బడిన"]),
        ("హైదరాబాద్తో", "ఎవరితో?", "హైదరాబాద్తో సంబంధం.", "హైదరాబాద్", ["తో"]),
        ("గాంధీగారు", "ఎవరు?", "గాంధీగారు చెప్పారు.", "గాంధీ", ["గారు"]),
    ]
    
    print("Suffix trimming results:")
    all_passed = True
    
    for answer, question, context, expected_answer, expected_removed in test_cases:
        result = processor.refine_answer(answer, question, context)
        
        answer_match = result.refined_answer == expected_answer
        # Check if at least one expected suffix was removed
        removed_match = any(s in result.removed_suffixes for s in expected_removed) if expected_removed else True
        
        status = "✓" if (answer_match or removed_match) else "✗"
        if not (answer_match or removed_match):
            all_passed = False
        
        print(f"\n  {status} Input: '{answer}'")
        print(f"    Question type: {result.question_type}")
        print(f"    Refined: '{result.refined_answer}' (expected: '{expected_answer}')")
        print(f"    Removed: {result.removed_suffixes} (expected: {expected_removed})")
        print(f"    Confidence adj: {result.confidence_adjustment:+.3f}")
    
    return all_passed


def test_compound_normalization():
    """Test compound word normalization."""
    print("\n" + "=" * 60)
    print("TEST 4: Compound Word Normalization")
    print("=" * 60)
    
    normalizer = CompoundNormalizer()
    
    test_cases = [
        ("నాగార్జున సాగర్", "నాగార్జునసాగర్లో నీరు ఉంది.", "నాగార్జునసాగర్"),
        ("ఆంధ్ర ప్రదేశ్", "ఆంధ్రప్రదేశ్ రాష్ట్రం.", "ఆంధ్రప్రదేశ్"),
        ("హైదర్ ఆబాద్", "హైదరాబాద్ నగరం.", "హైదరాబాద్"),
    ]
    
    print("Compound normalization results:")
    all_passed = True
    
    for text, context, expected in test_cases:
        result = normalizer.normalize(text, context)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} '{text}' -> '{result}' (expected: '{expected}')")
    
    return all_passed


def test_full_pipeline():
    """Test the full morphology pipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Pipeline Integration")
    print("=" * 60)
    
    processor = TeluguMorphologyProcessor(
        use_question_aware=True,
        use_compound_norm=True,
        use_confidence_rescoring=True,
        use_context_validation=True
    )
    
    # Realistic test case
    context = """
    హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని. ఇది దక్కన్ పీఠభూమిపై ఉంది. 
    1591లో ముహమ్మద్ కులీ కుతుబ్ షా ద్వారా స్థాపించబడింది. 
    హైదరాబాద్ జనాభా దాదాపు 1 కోటి మంది.
    """
    
    test_cases = [
        ("దక్కన్ పీఠభూమిపై", "హైదరాబాద్ ఎక్కడ ఉంది?"),
        ("ముహమ్మద్ కులీ కుతుబ్ షా ద్వారా", "ఎవరు స్థాపించారు?"),
        ("1591లో", "ఎప్పుడు స్థాపించబడింది?"),
        ("1 కోటి మంది", "జనాభా ఎంత?"),
    ]
    
    print("Full pipeline results:")
    
    for answer, question in test_cases:
        result = processor.refine_answer(answer, question, context)
        
        print(f"\n  Question: {question}")
        print(f"  Q-Type: {result.question_type}")
        print(f"  Original: '{result.original_answer}'")
        print(f"  Refined:  '{result.refined_answer}'")
        print(f"  Removed:  {result.removed_suffixes}")
        print(f"  Notes:    {result.notes}")
    
    return True


def main():
    print("=" * 60)
    print("TELUGU MORPHOLOGY MODULE - TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Suffix Patterns", test_suffix_patterns()))
    results.append(("Question Type Detection", test_question_type_detection()))
    results.append(("Suffix Trimming", test_suffix_trimming()))
    results.append(("Compound Normalization", test_compound_normalization()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
