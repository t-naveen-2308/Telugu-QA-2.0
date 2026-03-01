 -*- coding: utf-8 -*-
"""Test coreference resolution in Telugu QA."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from src.morphology.coreference import CoreferenceResolver
from src.morphology.processor import TeluguMorphologyProcessor

print("=" * 60)
print("TEST 1: Direct Coreference Resolution")
print("=" * 60)

resolver = CoreferenceResolver()

# Test case: "ఈ నగరంలో" should resolve to "విజయవాడలో"
answer = "ఈ నగరంలో"
context = "విజయవాడ ఆంధ్రప్రదేశ్ లోని ముఖ్యమైన నగరం. ఇది కృష్ణా నది ఒడ్డున ఉంది. కనకదుర్గ ఆలయం ఈ నగరంలో ప్రసిద్ధ పుణ్యక్షేత్రం."
question = "కనకదుర్గ ఆలయం ఎక్కడ ఉంది?"

result = resolver.resolve(answer, context, question)
print(f"Answer: {answer}")
print(f"Resolved: {result.resolved if result else 'N/A'}")
print(f"Antecedent: {result.antecedent if result else 'N/A'}")
print(f"Confidence: {result.confidence if result else 'N/A'}")

print()
print("=" * 60)
print("TEST 2: Full Processor with Coreference")
print("=" * 60)

processor = TeluguMorphologyProcessor(use_coreference=True)

# Test morphology refine with coreference
answer = "ఈ నగరంలో ప్రసిద్ధ పుణ్యక్షేత్రం"
result = processor.refine_answer(answer, question, context)

print(f"Original: {result.original_answer}")
print(f"Refined: {result.refined_answer}")
print(f"Notes: {result.notes}")
print(f"Refinement applied: {result.refinement_applied}")

print()
print("=" * 60)
print("TEST 3: More Coreference Examples")
print("=" * 60)

test_cases = [
    {
        "answer": "ఆ రాష్ట్రంలో",
        "context": "తెలంగాణ దక్షిణ భారతదేశంలో ఉన్న రాష్ట్రం. ఆ రాష్ట్రంలో హైదరాబాద్ రాజధాని.",
        "question": "హైదరాబాద్ ఎక్కడ ఉంది?",
    },
    {
        "answer": "ఈ నదిలో",
        "context": "గోదావరి నది భారతదేశంలో రెండవ పెద్ద నది. ఈ నదిలో చాలా చేపలు ఉంటాయి.",
        "question": "చేపలు ఎక్కడ ఉన్నాయి?",
    },
]

for i, tc in enumerate(test_cases, 1):
    result = resolver.resolve(tc["answer"], tc["context"], tc["question"])
    print(f"{i}. Answer: {tc['answer']}")
    print(f"   Resolved: {result.resolved if result else 'No resolution'}")
    print(f"   Antecedent: {result.antecedent if result else 'N/A'}")
    print()

print("=" * 60)
print("All tests completed!")
print("=" * 60)
