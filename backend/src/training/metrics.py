"""
QA Evaluation Metrics

Implements Exact Match (EM) and F1 Score for Question Answering.
Handles Telugu-specific text normalization.
"""

import collections
import re
import string
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    - Handle Telugu-specific normalization
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (both English and common Telugu punctuation)
    text = re.sub(r'[।॥,.\?!;:\'"(){}[\]]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def get_tokens(text: str) -> List[str]:
    """Split text into tokens."""
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match score.
    
    Returns 1.0 if normalized prediction equals normalized ground truth, else 0.0.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    # Count common tokens
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_metrics_for_example(
    prediction: str,
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Compute metrics for a single example with multiple ground truths.
    
    Returns the maximum EM and F1 across all ground truths.
    """
    exact_match = max(compute_exact_match(prediction, gt) for gt in ground_truths)
    f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    
    return {"exact_match": exact_match, "f1": f1}


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    tokenizer,
    n_best_size: int = 20,
    max_answer_length: int = 30
) -> Dict[str, str]:
    """
    Post-process model predictions to extract answer text.
    
    Args:
        examples: Original examples with context
        features: Tokenized features with offset mapping
        raw_predictions: Tuple of (start_logits, end_logits)
        tokenizer: Tokenizer for decoding
        n_best_size: Number of best predictions to consider
        max_answer_length: Maximum answer length in tokens
        
    Returns:
        Dictionary mapping example_id to predicted answer text
    """
    start_logits, end_logits = raw_predictions
    
    # Build mapping from example to features
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    predictions = {}
    
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        
        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Track null score (CLS position)
            cls_score = start_logit[0] + end_logit[0]
            if min_null_score is None or cls_score < min_null_score:
                min_null_score = cls_score
            
            # Get n_best start and end indices
            start_indexes = np.argsort(start_logit)[-n_best_size:][::-1].tolist()
            end_indexes = np.argsort(end_logit)[-n_best_size:][::-1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid indices
                    if start_index >= len(offset_mapping):
                        continue
                    if end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None:
                        continue
                    if offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    
                    valid_answers.append({
                        "score": start_logit[start_index] + end_logit[end_index],
                        "text": context[start_char:end_char]
                    })
        
        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            predictions[example["id"]] = best_answer["text"]
        else:
            predictions[example["id"]] = ""
    
    return predictions


def compute_qa_metrics(
    predictions: Dict[str, str],
    references: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Compute QA metrics over all examples.
    
    Args:
        predictions: Dict mapping example_id to predicted answer
        references: Dict mapping example_id to list of ground truth answers
        
    Returns:
        Dictionary with "exact_match" and "f1" (both as percentages)
    """
    total_em = 0.0
    total_f1 = 0.0
    count = 0
    
    for example_id, prediction in predictions.items():
        if example_id not in references:
            continue
        
        ground_truths = references[example_id]
        metrics = compute_metrics_for_example(prediction, ground_truths)
        
        total_em += metrics["exact_match"]
        total_f1 += metrics["f1"]
        count += 1
    
    return {
        "exact_match": 100.0 * total_em / count if count > 0 else 0.0,
        "f1": 100.0 * total_f1 / count if count > 0 else 0.0
    }


# Quick test
if __name__ == "__main__":
    # Test with Telugu text
    pred = "హైదరాబాద్"
    gold = ["హైదరాబాద్", "హైదరాబాద్ నగరం"]
    
    print(f"Prediction: {pred}")
    print(f"Ground truths: {gold}")
    
    for gt in gold:
        em = compute_exact_match(pred, gt)
        f1 = compute_f1(pred, gt)
        print(f"  vs '{gt}': EM={em}, F1={f1:.4f}")
    
    metrics = compute_metrics_for_example(pred, gold)
    print(f"\nBest: EM={metrics['exact_match']}, F1={metrics['f1']:.4f}")
