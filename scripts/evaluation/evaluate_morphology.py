"""
Evaluate model with morphology refinement.

Compares model performance with and without morphology-aware answer refinement.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.qa_engine import load_qa_engine
from src.evaluation.metrics import evaluate_with_morphology_analysis


def load_test_data(data_path: str):
    """Load test data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle SQuAD format
    if 'data' in data:
        samples = []
        for article in data['data']:
            for para in article['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    samples.append({
                        'question': qa['question'],
                        'context': context,
                        'answers': {
                            'text': [a['text'] for a in qa['answers']],
                            'answer_start': [a['answer_start'] for a in qa['answers']]
                        }
                    })
        return samples
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Telugu QA model with morphology refinement'
    )
    parser.add_argument(
        '--model', '-m',
        default='muril',
        choices=['muril', 'indicbert', 'xlmr', 'mbert'],
        help='Model to evaluate'
    )
    parser.add_argument(
        '--test-data', '-d',
        default=str(project_root / 'data' / 'processed' / 'tequad_test_wiki.json'),
        help='Path to test data JSON file'
    )
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Use aggressive morphology trimming'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Telugu QA Evaluation with Morphology Analysis")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Aggressive mode: {args.aggressive}")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test samples")
    
    if args.max_samples:
        print(f"Using first {args.max_samples} samples")
    
    # Load model with morphology enabled
    print(f"\nLoading {args.model} model with morphology refinement...")
    engine = load_qa_engine(
        model_key=args.model,
        use_morphology=True,
        morphology_aggressive=args.aggressive
    )
    
    # Run evaluation
    print("\nRunning evaluation (comparing with/without morphology)...")
    print("-" * 70)
    
    results = evaluate_with_morphology_analysis(
        engine=engine,
        test_data=test_data,
        max_samples=args.max_samples,
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'With Morphology':<20} {'Without Morphology':<20} {'Delta':<10}")
    print("-" * 80)
    
    em_with = results['exact_match']
    em_without = results['comparison']['without_morphology']['exact_match']
    f1_with = results['f1']
    f1_without = results['comparison']['without_morphology']['f1']
    
    em_delta = results['comparison']['improvement']['em_delta']
    f1_delta = results['comparison']['improvement']['f1_delta']
    
    print(f"{'Exact Match':<30} {em_with:>18.2f}% {em_without:>18.2f}% {em_delta:>+9.2f}%")
    print(f"{'F1 Score':<30} {f1_with:>18.2f}% {f1_without:>18.2f}% {f1_delta:>+9.2f}%")
    
    print("\n" + "-" * 70)
    print("MORPHOLOGY REFINEMENT STATISTICS")
    print("-" * 70)
    
    ref_stats = results['comparison']['refinement_stats']
    imp_stats = results['comparison']['improvement']
    
    print(f"Total samples refined: {ref_stats['total_refined']} ({ref_stats['refinement_rate']:.1f}%)")
    print(f"Samples improved (EM): {imp_stats['samples_improved_em']}")
    print(f"Samples hurt (EM):     {imp_stats['samples_hurt_em']}")
    print(f"Samples improved (F1): {imp_stats['samples_improved_f1']}")
    print(f"Samples hurt (F1):     {imp_stats['samples_hurt_f1']}")
    
    # Print morphology error analysis if available
    if results.get('morphology_analysis'):
        analysis = results['morphology_analysis']
        print("\n" + "-" * 70)
        print("MORPHOLOGY ERROR ANALYSIS (Before Refinement)")
        print("-" * 70)
        
        print(f"Over-extractions: {analysis.get('over_extraction', 0)} ({analysis.get('over_extraction_rate', 0)*100:.1f}%)")
        print(f"Under-extractions: {analysis.get('under_extraction', 0)} ({analysis.get('under_extraction_rate', 0)*100:.1f}%)")
        print(f"Compound word errors: {analysis.get('compound_errors', 0)} ({analysis.get('compound_error_rate', 0)*100:.1f}%)")
        
        if analysis.get('suffix_errors'):
            print("\nMost common problematic suffixes:")
            sorted_suffixes = sorted(
                analysis['suffix_errors'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for suffix, count in sorted_suffixes:
                print(f"  {suffix}: {count}")
    
    # Print improvement examples
    if results['examples']['improvements']:
        print("\n" + "-" * 70)
        print("IMPROVEMENT EXAMPLES (First 5)")
        print("-" * 70)
        
        for i, ex in enumerate(results['examples']['improvements'][:5], 1):
            print(f"\n{i}. Question: {ex['question'][:60]}...")
            print(f"   Without morph: {ex['without_morph']}")
            print(f"   With morph:    {ex['with_morph']}")
            print(f"   Gold:          {ex['gold'][0] if ex['gold'] else 'N/A'}")
            if ex.get('removed'):
                print(f"   Removed:       {ex['removed']}")
    
    # Print regression examples if any
    if results['examples']['regressions']:
        print("\n" + "-" * 70)
        print("REGRESSION EXAMPLES (First 5)")
        print("-" * 70)
        
        for i, ex in enumerate(results['examples']['regressions'][:5], 1):
            print(f"\n{i}. Question: {ex['question'][:60]}...")
            print(f"   Without morph: {ex['without_morph']}")
            print(f"   With morph:    {ex['with_morph']}")
            print(f"   Gold:          {ex['gold'][0] if ex['gold'] else 'N/A'}")
            if ex.get('removed'):
                print(f"   Removed:       {ex['removed']}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = str(
            project_root / 'data' / 'processed' / 
            f'evaluation_morphology_{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    # Prepare summary for saving (exclude large result arrays)
    summary = {
        'model': args.model,
        'test_samples': results['num_samples'],
        'timestamp': datetime.now().isoformat(),
        'aggressive_mode': args.aggressive,
        'metrics': {
            'with_morphology': {
                'exact_match': em_with,
                'f1': f1_with
            },
            'without_morphology': {
                'exact_match': em_without,
                'f1': f1_without
            },
            'improvement': {
                'em_delta': em_delta,
                'f1_delta': f1_delta
            }
        },
        'refinement_stats': ref_stats,
        'improvement_breakdown': imp_stats,
        'morphology_analysis': results.get('morphology_analysis'),
        'examples': results['examples']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
