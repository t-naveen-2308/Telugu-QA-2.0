"""
Run evaluation on the Telugu QA model.

Usage:
    python scripts/run_evaluation.py [--samples N]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.qa_engine import load_qa_engine
from src.data.tequad_loader import TeQuADDataset
from src.evaluation.metrics import evaluate_qa_model, save_evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Telugu QA Model")
    parser.add_argument(
        "--samples", 
        type=int, 
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/evaluation_results.json",
        help="Output path for results"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Telugu QA Model Evaluation")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    engine = load_qa_engine("muril")
    
    # Load test data
    print("\n2. Loading test data...")
    dataset = TeQuADDataset()
    test_data = list(dataset.test)
    print(f"   Test samples: {len(test_data)}")
    
    if args.samples:
        print(f"   Limiting to: {args.samples} samples")
    
    # Run evaluation
    print("\n3. Running evaluation...")
    print("-" * 40)
    
    results = evaluate_qa_model(
        engine=engine,
        test_data=test_data,
        max_samples=args.samples,
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  📊 Exact Match (EM): {results['exact_match']:.2f}%")
    print(f"  📊 F1 Score:         {results['f1']:.2f}%")
    print(f"  📊 Samples Evaluated: {results['num_samples']}")
    print()
    
    # Save results
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(results, str(output_path))
    
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
