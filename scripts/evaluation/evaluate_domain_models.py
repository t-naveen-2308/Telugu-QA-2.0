"""
Evaluate base vs domain-adapted models on domain test sets.
Compares mBERT base vs mBERT-Domain (LoRA) and MuRIL base vs MuRIL-Domain (LoRA).

Usage:
    python scripts/evaluate_domain_models.py
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    import re
    text = text.strip()
    text = text.lower()
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]\\^_`{|}~]', ' ', text)
    text = ' '.join(text.split())
    return text


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def load_domain_test(domain: str) -> list:
    """Load domain test set and flatten SQuAD format to list of (context, question, answers)."""
    test_path = project_root / "data" / "domain" / domain / "test" / f"{domain}_test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answer_texts = [a["text"].strip() for a in qa["answers"]]
                samples.append({
                    "context": context,
                    "question": qa["question"],
                    "answers": answer_texts,
                    "id": qa.get("id", ""),
                })
    return samples


def evaluate_engine_on_samples(engine, samples: list) -> dict:
    """Run evaluation and return EM, F1."""
    total_em = 0.0
    total_f1 = 0.0
    
    for sample in samples:
        try:
            result = engine.answer(sample["question"], sample["context"])
            pred = result["answer"]
        except Exception:
            pred = ""
        
        # Best score across all gold answers
        em = max(compute_em(pred, gt) for gt in sample["answers"])
        f1 = max(compute_f1(pred, gt) for gt in sample["answers"])
        total_em += em
        total_f1 += f1
    
    n = len(samples)
    return {
        "exact_match": (total_em / n) * 100 if n > 0 else 0,
        "f1": (total_f1 / n) * 100 if n > 0 else 0,
        "num_samples": n
    }


def main():
    from src.inference.qa_engine import TeluguQAEngine
    
    domains = ["government", "literature", "news"]
    
    # Load test data for all domains
    print("=" * 70)
    print("DOMAIN MODEL EVALUATION: Base vs LoRA-adapted")
    print("=" * 70)
    
    domain_data = {}
    total_samples = 0
    for domain in domains:
        samples = load_domain_test(domain)
        domain_data[domain] = samples
        total_samples += len(samples)
        print(f"  {domain}: {len(samples)} test samples")
    print(f"  Total: {total_samples} samples")
    
    # Models to evaluate
    model_pairs = [
        ("mbert", "mbert-domain"),
        ("muril", "muril-domain"),
    ]
    
    all_results = {}
    
    for base_key, domain_key in model_pairs:
        print(f"\n{'='*70}")
        print(f"  Evaluating: {base_key} vs {domain_key}")
        print(f"{'='*70}")
        
        # Load base model
        print(f"\n  Loading {base_key} (base)...")
        t0 = time.time()
        try:
            base_engine = TeluguQAEngine(
                model_key=base_key,
                use_morphology=False
            )
            base_load_time = time.time() - t0
            print(f"  Loaded in {base_load_time:.1f}s")
        except Exception as e:
            print(f"  ERROR loading {base_key}: {e}")
            continue
        
        # Evaluate base on each domain
        base_results = {}
        for domain in domains:
            print(f"  Evaluating {base_key} on {domain}...")
            r = evaluate_engine_on_samples(base_engine, domain_data[domain])
            base_results[domain] = r
            print(f"    EM: {r['exact_match']:.1f}%  F1: {r['f1']:.1f}%  ({r['num_samples']} samples)")
        
        # Free memory
        del base_engine
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Load domain model
        print(f"\n  Loading {domain_key} (LoRA)...")
        t0 = time.time()
        try:
            domain_engine = TeluguQAEngine(
                model_key=domain_key,
                use_morphology=False
            )
            domain_load_time = time.time() - t0
            print(f"  Loaded in {domain_load_time:.1f}s")
        except Exception as e:
            print(f"  ERROR loading {domain_key}: {e}")
            continue
        
        # Evaluate domain model on each domain
        domain_results = {}
        for domain in domains:
            print(f"  Evaluating {domain_key} on {domain}...")
            r = evaluate_engine_on_samples(domain_engine, domain_data[domain])
            domain_results[domain] = r
            print(f"    EM: {r['exact_match']:.1f}%  F1: {r['f1']:.1f}%  ({r['num_samples']} samples)")
        
        # Free memory
        del domain_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Compute averages
        avg_base_em = sum(base_results[d]["exact_match"] for d in domains) / len(domains)
        avg_base_f1 = sum(base_results[d]["f1"] for d in domains) / len(domains)
        avg_domain_em = sum(domain_results[d]["exact_match"] for d in domains) / len(domains)
        avg_domain_f1 = sum(domain_results[d]["f1"] for d in domains) / len(domains)
        
        all_results[base_key] = {
            "base": base_results,
            "domain": domain_results,
            "avg_base": {"exact_match": avg_base_em, "f1": avg_base_f1},
            "avg_domain": {"exact_match": avg_domain_em, "f1": avg_domain_f1},
        }
        
        # Print comparison table
        print(f"\n  {'─'*60}")
        print(f"  {base_key.upper()} comparison: Base vs Domain (LoRA)")
        print(f"  {'─'*60}")
        print(f"  {'Domain':<12} {'Base EM':>8} {'Domain EM':>10} {'Δ EM':>8} {'Base F1':>8} {'Domain F1':>10} {'Δ F1':>8}")
        print(f"  {'─'*60}")
        for domain in domains:
            b = base_results[domain]
            d = domain_results[domain]
            d_em = d["exact_match"] - b["exact_match"]
            d_f1 = d["f1"] - b["f1"]
            print(f"  {domain:<12} {b['exact_match']:>7.1f}% {d['exact_match']:>9.1f}% {d_em:>+7.1f}% {b['f1']:>7.1f}% {d['f1']:>9.1f}% {d_f1:>+7.1f}%")
        print(f"  {'─'*60}")
        d_em_avg = avg_domain_em - avg_base_em
        d_f1_avg = avg_domain_f1 - avg_base_f1
        print(f"  {'Average':<12} {avg_base_em:>7.1f}% {avg_domain_em:>9.1f}% {d_em_avg:>+7.1f}% {avg_base_f1:>7.1f}% {avg_domain_f1:>9.1f}% {d_f1_avg:>+7.1f}%")
    
    # Save results
    output_path = project_root / "data" / "domain" / "domain_evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
