"""
Evaluate Telugu QA Model

Runs EM (Exact Match) and F1 evaluation on the test set.
"""

import sys
import json
import re
import string
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.qa_engine import load_qa_engine
from src.data.tequad_loader import load_tequad_dataset


def normalize_telugu_text(text: str) -> str:
    """Normalize Telugu text for comparison."""
    # Lowercase (for any English mixed in)
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[।,?!.\-:;\'"()।॥]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_telugu_text(prediction) == normalize_telugu_text(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score based on token overlap."""
    pred_tokens = normalize_telugu_text(prediction).split()
    truth_tokens = normalize_telugu_text(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_metrics_for_sample(
    prediction: str,
    ground_truths: List[str]
) -> Tuple[float, float]:
    """
    Compute EM and F1 for a single sample.
    Takes max over all ground truth answers.
    """
    em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
    f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]
    
    return max(em_scores), max(f1_scores)


def evaluate_model(
    model_key: str = "muril",
    split: str = "test",
    max_samples: int = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate the model on the test set.
    
    Args:
        model_key: Which model to evaluate
        split: Which split to evaluate on ("test" or "validation")
        max_samples: Limit number of samples (for quick testing)
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print(f"Evaluating: {model_key.upper()} on {split} set")
    print("="*60)
    
    # Load model
    print("\n[*] Loading model...")
    engine = load_qa_engine(model_key)
    
    # Load dataset
    print("[*] Loading dataset...")
    dataset = load_tequad_dataset(split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"   Evaluating on {len(dataset)} samples")
    
    # Evaluate
    results = []
    em_total = 0
    f1_total = 0
    
    print("\n[*] Running evaluation...")
    for i, sample in enumerate(tqdm(dataset, disable=not verbose)):
        question = sample["question"]
        context = sample["context"]
        ground_truths = sample["answers"]["text"]
        
        # Get prediction
        try:
            result = engine.answer(question, context)
            prediction = result["answer"]
            score = result["score"]
        except Exception as e:
            prediction = ""
            score = 0.0
        
        # Compute metrics
        em, f1 = compute_metrics_for_sample(prediction, ground_truths)
        em_total += em
        f1_total += f1
        
        results.append({
            "id": sample.get("id", i),
            "question": question,
            "prediction": prediction,
            "ground_truths": ground_truths,
            "confidence": score,
            "em": em,
            "f1": f1
        })
    
    # Aggregate metrics
    n = len(results)
    metrics = {
        "model": model_key,
        "split": split,
        "num_samples": n,
        "exact_match": (em_total / n) * 100,
        "f1": (f1_total / n) * 100,
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Model:        {model_key}")
    print(f"  Split:        {split}")
    print(f"  Samples:      {n}")
    print(f"  Exact Match:  {metrics['exact_match']:.2f}%")
    print(f"  F1 Score:     {metrics['f1']:.2f}%")
    print("="*60)
    
    # Show some examples
    if verbose:
        print("\nSample Predictions:")
        for i, r in enumerate(results[:5]):
            status = "[OK]" if r["em"] == 1.0 else "[X]"
            print(f"\n{status} Sample {i+1}:")
            print(f"   Q: {r['question'][:60]}...")
            print(f"   Pred: {r['prediction']}")
            print(f"   Gold: {r['ground_truths'][0]}")
            print(f"   F1: {r['f1']:.2f}")
    
    # Save detailed results - per model
    output_path = project_root / "data" / "processed" / f"evaluation_results_{model_key}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Detailed results saved to: {output_path}")
    
    # Also save to generic path for backward compatibility
    generic_path = project_root / "data" / "processed" / "evaluation_results.json"
    with open(generic_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Telugu QA Model")
    parser.add_argument("--model", default="muril", help="Model key (muril, indicbert)")
    parser.add_argument("--split", default="test", help="Dataset split (test, validation)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick test")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_key=args.model,
        split=args.split,
        max_samples=args.max_samples
    )
