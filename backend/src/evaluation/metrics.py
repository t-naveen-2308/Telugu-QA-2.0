"""
Telugu QA Evaluation Metrics

Computes Exact Match (EM) and F1 scores for QA evaluation.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple
import json
from pathlib import Path


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (keeping Telugu characters)
    # Only remove ASCII punctuation
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^_`{|}~]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score (0 or 1)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    Works for Telugu by splitting on whitespace.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_metrics_for_sample(
    prediction: str,
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Compute EM and F1 for a single sample.
    
    Takes max score across all ground truth answers.
    """
    em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
    f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]
    
    return {
        "exact_match": max(em_scores),
        "f1": max(f1_scores)
    }


def evaluate_qa_model(
    engine,
    test_data: List[Dict],
    max_samples: int = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate QA model on test dataset.
    
    Args:
        engine: TeluguQAEngine instance
        test_data: List of test samples with 'question', 'context', 'answers'
        max_samples: Limit evaluation to N samples (None = all)
        verbose: Print progress
        
    Returns:
        Dictionary with EM, F1, and per-sample results
    """
    if max_samples:
        test_data = test_data[:max_samples]
    
    total_em = 0.0
    total_f1 = 0.0
    results = []
    
    for i, sample in enumerate(test_data):
        question = sample['question']
        context = sample['context']
        ground_truths = sample['answers']['text']
        
        # Get prediction
        try:
            result = engine.answer(question, context)
            prediction = result['answer']
            confidence = result['score']
        except Exception as e:
            prediction = ""
            confidence = 0.0
        
        # Compute metrics
        metrics = compute_metrics_for_sample(prediction, ground_truths)
        
        total_em += metrics['exact_match']
        total_f1 += metrics['f1']
        
        results.append({
            'question': question,
            'prediction': prediction,
            'ground_truths': ground_truths,
            'exact_match': metrics['exact_match'],
            'f1': metrics['f1'],
            'confidence': confidence
        })
        
        if verbose and (i + 1) % 50 == 0:
            print(f"Evaluated {i + 1}/{len(test_data)} samples...")
    
    n = len(test_data)
    avg_em = (total_em / n) * 100
    avg_f1 = (total_f1 / n) * 100
    
    return {
        'exact_match': avg_em,
        'f1': avg_f1,
        'num_samples': n,
        'results': results
    }


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    # Don't save individual results to keep file small
    summary = {
        'exact_match': results['exact_match'],
        'f1': results['f1'],
        'num_samples': results['num_samples'],
        'morphology_enabled': results.get('morphology_enabled', False),
        'morphology_analysis': results.get('morphology_analysis', None)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")


def load_evaluation_results(path: str) -> Dict:
    """Load evaluation results from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_with_morphology_analysis(
    engine,
    test_data: List[Dict],
    max_samples: int = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate QA model with detailed morphology error analysis.
    
    Compares results with and without morphology refinement and
    provides insights into where refinement helps.
    
    Args:
        engine: TeluguQAEngine instance with morphology enabled
        test_data: List of test samples
        max_samples: Limit evaluation to N samples
        verbose: Print progress
        
    Returns:
        Dictionary with metrics, analysis, and comparison
    """
    if max_samples:
        test_data = test_data[:max_samples]
    
    # Collect results with and without morphology
    results_with_morph = []
    results_without_morph = []
    
    predictions_with = []
    predictions_without = []
    ground_truths_list = []
    questions_list = []
    
    for i, sample in enumerate(test_data):
        question = sample['question']
        context = sample['context']
        ground_truths = sample['answers']['text']
        
        # Get prediction WITH morphology refinement
        try:
            result_with = engine.answer(question, context, apply_refinement=True)
            pred_with = result_with['answer']
            conf_with = result_with['score']
            original_pred = result_with.get('original_answer', pred_with)
            refinement_applied = result_with.get('refinement_applied', False)
            removed_suffixes = result_with.get('removed_suffixes', [])
        except Exception as e:
            pred_with = ""
            conf_with = 0.0
            original_pred = ""
            refinement_applied = False
            removed_suffixes = []
        
        # Get prediction WITHOUT morphology refinement
        try:
            result_without = engine.answer(question, context, apply_refinement=False)
            pred_without = result_without['answer']
            conf_without = result_without['score']
        except Exception as e:
            pred_without = ""
            conf_without = 0.0
        
        # Compute metrics
        metrics_with = compute_metrics_for_sample(pred_with, ground_truths)
        metrics_without = compute_metrics_for_sample(pred_without, ground_truths)
        
        results_with_morph.append({
            'question': question,
            'prediction': pred_with,
            'original_prediction': original_pred,
            'ground_truths': ground_truths,
            'exact_match': metrics_with['exact_match'],
            'f1': metrics_with['f1'],
            'confidence': conf_with,
            'refinement_applied': refinement_applied,
            'removed_suffixes': removed_suffixes
        })
        
        results_without_morph.append({
            'question': question,
            'prediction': pred_without,
            'ground_truths': ground_truths,
            'exact_match': metrics_without['exact_match'],
            'f1': metrics_without['f1'],
            'confidence': conf_without
        })
        
        predictions_with.append(pred_with)
        predictions_without.append(pred_without)
        ground_truths_list.append(ground_truths[0] if ground_truths else "")
        questions_list.append(question)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"Evaluated {i + 1}/{len(test_data)} samples...")
    
    n = len(test_data)
    
    # Calculate aggregate metrics
    em_with = sum(r['exact_match'] for r in results_with_morph) / n * 100
    f1_with = sum(r['f1'] for r in results_with_morph) / n * 100
    em_without = sum(r['exact_match'] for r in results_without_morph) / n * 100
    f1_without = sum(r['f1'] for r in results_without_morph) / n * 100
    
    # Count refinement statistics
    refinement_count = sum(1 for r in results_with_morph if r['refinement_applied'])
    
    # Analyze cases where morphology helped or hurt
    improved_em = 0
    hurt_em = 0
    improved_f1 = 0
    hurt_f1 = 0
    
    improvement_examples = []
    regression_examples = []
    
    for r_with, r_without in zip(results_with_morph, results_without_morph):
        if r_with['exact_match'] > r_without['exact_match']:
            improved_em += 1
            if len(improvement_examples) < 10:
                improvement_examples.append({
                    'question': r_with['question'],
                    'with_morph': r_with['prediction'],
                    'without_morph': r_without['prediction'],
                    'gold': r_with['ground_truths'],
                    'removed': r_with.get('removed_suffixes', [])
                })
        elif r_with['exact_match'] < r_without['exact_match']:
            hurt_em += 1
            if len(regression_examples) < 10:
                regression_examples.append({
                    'question': r_with['question'],
                    'with_morph': r_with['prediction'],
                    'without_morph': r_without['prediction'],
                    'gold': r_with['ground_truths'],
                    'removed': r_with.get('removed_suffixes', [])
                })
        
        if r_with['f1'] > r_without['f1']:
            improved_f1 += 1
        elif r_with['f1'] < r_without['f1']:
            hurt_f1 += 1
    
    # Build morphology error analysis using engine's analyzer
    morph_analysis = None
    if hasattr(engine, 'analyze_morphology_errors'):
        morph_analysis = engine.analyze_morphology_errors(
            predictions_without, ground_truths_list, questions_list
        )
    
    return {
        # Metrics with morphology
        'exact_match': em_with,
        'f1': f1_with,
        'num_samples': n,
        'morphology_enabled': True,
        
        # Comparison
        'comparison': {
            'without_morphology': {
                'exact_match': em_without,
                'f1': f1_without
            },
            'improvement': {
                'em_delta': em_with - em_without,
                'f1_delta': f1_with - f1_without,
                'samples_improved_em': improved_em,
                'samples_hurt_em': hurt_em,
                'samples_improved_f1': improved_f1,
                'samples_hurt_f1': hurt_f1,
            },
            'refinement_stats': {
                'total_refined': refinement_count,
                'refinement_rate': refinement_count / n * 100 if n > 0 else 0
            }
        },
        
        # Examples
        'examples': {
            'improvements': improvement_examples,
            'regressions': regression_examples
        },
        
        # Morphology analysis
        'morphology_analysis': morph_analysis,
        
        # Detailed results
        'results': results_with_morph,
        'results_without_morph': results_without_morph
    }


if __name__ == "__main__":
    print("Telugu QA Evaluation Metrics")
    print("=" * 40)
    
    # Test the metrics
    pred = "హైదరాబాద్"
    gold = "హైదరాబాద్"
    
    print(f"Prediction: {pred}")
    print(f"Ground Truth: {gold}")
    print(f"EM: {compute_exact_match(pred, gold)}")
    print(f"F1: {compute_f1(pred, gold)}")
    
    # Test with suffix
    pred_with_suffix = "హైదరాబాద్లో"
    print(f"\nPrediction with suffix: {pred_with_suffix}")
    print(f"Ground Truth: {gold}")
    print(f"EM: {compute_exact_match(pred_with_suffix, gold)}")
    print(f"F1: {compute_f1(pred_with_suffix, gold)}")
