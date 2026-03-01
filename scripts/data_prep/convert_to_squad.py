"""
SQuAD Format Reference and TeQuAD Converter

This script:
1. Shows the standard SQuAD JSON format
2. Converts TeQuAD text files to SQuAD JSON format
3. Saves as HuggingFace-compatible dataset
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


# ============================================================
# STANDARD SQuAD FORMAT (for reference)
# ============================================================
"""
SQuAD JSON Format:
{
    "version": "1.1",
    "data": [
        {
            "title": "Article Title",
            "paragraphs": [
                {
                    "context": "The paragraph text...",
                    "qas": [
                        {
                            "id": "unique_id",
                            "question": "What is...?",
                            "answers": [
                                {
                                    "text": "answer text",
                                    "answer_start": 123  # CHARACTER offset!
                                }
                            ],
                            "is_impossible": false  # SQuAD 2.0 only
                        }
                    ]
                }
            ]
        }
    ]
}

Key points:
- answer_start is CHARACTER offset (not word offset!)
- answers[0].text must EXACTLY match context[answer_start:answer_start+len(text)]
- id must be unique across entire dataset
"""


def find_answer_in_context(context: str, answer: str) -> Optional[int]:
    """
    Find the character position of answer in context.
    Returns the start index or None if not found.
    
    Tries multiple matching strategies:
    1. Exact match
    2. Normalized match (whitespace)
    3. Fuzzy match (substrings)
    """
    # Strategy 1: Exact match
    idx = context.find(answer)
    if idx != -1:
        return idx
    
    # Strategy 2: Normalize whitespace and try again
    def normalize(s):
        return ' '.join(s.split())
    
    norm_answer = normalize(answer)
    norm_context = normalize(context)
    
    idx = norm_context.find(norm_answer)
    if idx != -1:
        # Map back to original context position
        # This is approximate but usually works
        char_count = 0
        norm_char_count = 0
        for i, c in enumerate(context):
            if norm_char_count == idx:
                return i
            if not c.isspace() or (i > 0 and not context[i-1].isspace()):
                norm_char_count += 1
        return None
    
    # Strategy 3: Try finding answer without trailing punctuation
    answer_stripped = answer.rstrip('.,;:!?')
    if answer_stripped != answer:
        idx = context.find(answer_stripped)
        if idx != -1:
            return idx
    
    return None


def convert_tequad_to_squad(
    data_dir: Path,
    output_path: Path,
    prefix: str = "real_",
    dataset_name: str = "train"
) -> Dict:
    """
    Convert TeQuAD text files to SQuAD JSON format.
    
    Args:
        data_dir: Directory containing TeQuAD text files
        output_path: Path to save the output JSON
        prefix: File prefix (e.g., "real_" for train)
        dataset_name: Name for the dataset split
        
    Returns:
        SQuAD-format dictionary
    """
    # Try to find files with various naming patterns
    def find_file(patterns: List[str]) -> Optional[Path]:
        for pattern in patterns:
            path = data_dir / pattern
            if path.exists():
                return path
        return None
    
    # Context file patterns
    con_file = find_file([
        f"{prefix}con_tel.txt",
        "real_con_tel.txt",
        "con_tel.txt"
    ])
    
    # Question file patterns
    que_file = find_file([
        f"{prefix}que_tel.txt",
        "que_tel.txt"
    ])
    
    # Answer file patterns
    ans_file = find_file([
        f"{prefix}ans_tel.txt",
        "corrected_ans_tel.txt",
        "ans_tel.txt"
    ])
    
    if not all([con_file, que_file, ans_file]):
        missing = []
        if not con_file: missing.append("context")
        if not que_file: missing.append("question")
        if not ans_file: missing.append("answer")
        print(f"  ❌ Missing files: {missing}")
        print(f"  Available files: {[f.name for f in data_dir.glob('*.txt')]}")
        return {"version": "1.1", "data": []}
    
    print(f"Loading files from {data_dir}...")
    print(f"  Context: {con_file.name}")
    print(f"  Question: {que_file.name}")
    print(f"  Answer: {ans_file.name}")
    
    with open(con_file, 'r', encoding='utf-8') as f:
        contexts = [line.strip() for line in f.readlines()]
    
    with open(que_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f.readlines()]
    
    with open(ans_file, 'r', encoding='utf-8') as f:
        answers = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(contexts)} samples")
    
    # Convert to SQuAD format
    squad_data = {
        "version": "1.1",
        "data": []
    }
    
    # Group by unique contexts (since same context may have multiple questions)
    context_to_qas = {}
    
    found_count = 0
    not_found_count = 0
    
    for i, (context, question, answer) in enumerate(zip(contexts, questions, answers)):
        # Find answer position in context
        answer_start = find_answer_in_context(context, answer)
        
        if answer_start is None:
            not_found_count += 1
            # Skip samples where answer isn't found in context
            continue
        
        found_count += 1
        
        # Verify the extraction
        extracted = context[answer_start:answer_start + len(answer)]
        if extracted != answer:
            # Try to adjust
            for offset in [-1, 1, -2, 2]:
                test_start = answer_start + offset
                if test_start >= 0 and test_start + len(answer) <= len(context):
                    if context[test_start:test_start + len(answer)] == answer:
                        answer_start = test_start
                        break
        
        qa_entry = {
            "id": f"tequad_{dataset_name}_{i}",
            "question": question,
            "answers": [
                {
                    "text": answer,
                    "answer_start": answer_start
                }
            ]
        }
        
        # Group by context
        if context not in context_to_qas:
            context_to_qas[context] = []
        context_to_qas[context].append(qa_entry)
    
    print(f"Answer positions found: {found_count}/{found_count + not_found_count}")
    print(f"Answer positions not found: {not_found_count}")
    
    # Build final structure
    for idx, (context, qas) in enumerate(context_to_qas.items()):
        article = {
            "title": f"TeQuAD_{dataset_name}_{idx}",
            "paragraphs": [
                {
                    "context": context,
                    "qas": qas
                }
            ]
        }
        squad_data["data"].append(article)
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    print(f"Total articles: {len(squad_data['data'])}")
    print(f"Total QA pairs: {sum(len(a['paragraphs'][0]['qas']) for a in squad_data['data'])}")
    
    return squad_data


def verify_squad_format(data: Dict) -> None:
    """Verify the converted data is valid SQuAD format."""
    print("\n" + "="*60)
    print("🔍 Verifying SQuAD Format")
    print("="*60)
    
    errors = []
    verified = 0
    
    for article in data["data"][:10]:  # Check first 10 articles
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                for ans in qa["answers"]:
                    text = ans["text"]
                    start = ans["answer_start"]
                    
                    extracted = context[start:start + len(text)]
                    
                    if extracted == text:
                        verified += 1
                    else:
                        errors.append({
                            "id": qa["id"],
                            "expected": text,
                            "extracted": extracted,
                            "start": start
                        })
    
    print(f"Verified: {verified}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\nSample errors:")
        for err in errors[:3]:
            print(f"  ID: {err['id']}")
            print(f"    Expected: '{err['expected']}'")
            print(f"    Extracted: '{err['extracted']}'")


def main():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Convert Train set (82,605 samples - translated from English SQuAD)
    train_dir = raw_dir / "Train"
    if train_dir.exists():
        print("\n" + "="*60)
        print("📦 Converting TRAIN set (Translated)")
        print("="*60)
        train_data = convert_tequad_to_squad(
            train_dir,
            processed_dir / "tequad_train.json",
            prefix="real_",
            dataset_name="train"
        )
        verify_squad_format(train_data)
    
    # Convert Test - Translated & Corrected (1,000 samples - for validation)
    test_translated_dir = raw_dir / "Test" / "Translated & Corrected"
    if test_translated_dir.exists():
        print("\n" + "="*60)
        print("📦 Converting VALIDATION set (Translated & Corrected)")
        print("="*60)
        val_data = convert_tequad_to_squad(
            test_translated_dir,
            processed_dir / "tequad_validation.json",
            prefix="",  # No prefix for these files
            dataset_name="validation"
        )
        verify_squad_format(val_data)
    
    # Convert Test - Wiki Data (947 samples - native Telugu Wikipedia content)
    test_wiki_dir = raw_dir / "Test" / "Wiki Data"
    if test_wiki_dir.exists():
        print("\n" + "="*60)
        print("📦 Converting TEST set (Native Telugu Wiki)")
        print("="*60)
        test_data = convert_tequad_to_squad(
            test_wiki_dir,
            processed_dir / "tequad_test_wiki.json",
            prefix="",  # No prefix for these files
            dataset_name="test_wiki"
        )
        verify_squad_format(test_data)
    
    print("\n" + "="*60)
    print("✅ Conversion Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {processed_dir / 'tequad_train.json'} (82K+ training samples)")
    print(f"  - {processed_dir / 'tequad_validation.json'} (1K validation samples)")
    print(f"  - {processed_dir / 'tequad_test_wiki.json'} (947 native Telugu test samples)")
    print(f"\nThese files are now in standard SQuAD format!")
    print("You can load them with HuggingFace datasets library.")


if __name__ == "__main__":
    main()
