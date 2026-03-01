# -*- coding: utf-8 -*-
"""Test QA Engine with Coreference Resolution."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from src.inference.qa_engine import TeluguQAEngine

print("Loading model...")
engine = TeluguQAEngine(model_key='muril', use_morphology=True)

print()
print("=" * 60)
print("TEST: Kanakadurga Temple with Coreference")  
print("=" * 60)

question = "కనకదుర్గ ఆలయం ఎక్కడ ఉంది?"
context = "విజయవాడ ఆంధ్రప్రదేశ్ లోని ముఖ్యమైన నగరం. ఇది కృష్ణా నది ఒడ్డున ఉంది. కనకదుర్గ ఆలయం ఈ నగరంలో ప్రసిద్ధ పుణ్యక్షేత్రం."

result = engine.answer(question, context)

print(f"Question: {question}")
print(f"Context: {context}")
print("-" * 60)
print(f"Final Answer: {result['answer']}")
print(f"Original (raw model): {result.get('original_answer', 'N/A')}")
print(f"Confidence: {result['score']:.4f}")
print(f"Removed/resolved: {result.get('removed_suffixes', [])}")
print("=" * 60)
