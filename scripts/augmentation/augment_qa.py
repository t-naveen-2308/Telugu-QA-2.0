"""
Data augmentation techniques for Telugu QA pairs.

Techniques:
1. Question paraphrasing (rule-based + templates)
2. Context truncation / sentence reordering  
3. Answer span variation
4. Back-translation (requires translation API)
5. Synonym replacement

Usage:
    python scripts/augmentation/augment_qa.py --input data/domain/government/qa_pairs/*.json --output augmented.json
    python scripts/augmentation/augment_qa.py --all --multiplier 3
"""

import os
import sys
import json
import argparse
import random
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import copy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Telugu question word alternatives for paraphrasing
QUESTION_PARAPHRASES = {
    # What variations
    "ఏమిటి": ["ఏంటి", "ఏమి", "ఏది"],
    "ఏంటి": ["ఏమిటి", "ఏమి", "ఏది"],
    
    # Who variations
    "ఎవరు": ["ఎవ్వరు", "ఎవరెవరు"],
    
    # Where variations
    "ఎక్కడ": ["ఎక్కడెక్కడ", "ఏ ప్రదేశంలో", "ఏ చోట"],
    
    # When variations
    "ఎప్పుడు": ["ఎప్పటికి", "ఏ సమయంలో", "ఏ తేదీన"],
    
    # How much variations
    "ఎంత": ["ఎంతమేరకు", "ఎంత మొత్తం"],
    
    # Why variations
    "ఎందుకు": ["ఏ కారణం వల్ల", "ఎందుచేత", "ఏ వల్ల"],
    
    # How variations
    "ఎలా": ["ఏ విధంగా", "ఏ రీతిలో", "ఏ పద్ధతిలో"]
}

# Question structure templates
QUESTION_TEMPLATES = [
    # Government domain
    {"original": "ఈ పథకం ద్వారా ఎంత సహాయం అందుతుంది?",
     "variations": [
         "ఈ పథకంలో లబ్ధిదారులకు ఎంత ధన సహాయం లభిస్తుంది?",
         "పథకం ప్రయోజనం ఎంత?",
         "లబ్ధిదారులకు అందే సహాయం ఎంత?"
     ]},
    {"original": "ఈ పథకం ఎవరికి వర్తిస్తుంది?",
     "variations": [
         "ఈ పథకానికి ఎవరు అర్హులు?",
         "పథకం అర్హత ప్రమాణాలు ఏమిటి?",
         "ఎవరు ఈ పథకం కోసం దరఖాస్తు చేసుకోవచ్చు?"
     ]},
    {"original": "ఈ ఉత్తర్వులు ఏ శాఖకు సంబంధించినవి?",
     "variations": [
         "ఈ GO ఏ డిపార్ట్​మెంట్ జారీ చేసింది?",
         "ఈ నిర్ణయం ఏ శాఖ తీసుకుంది?",
         "సంబంధిత శాఖ ఏది?"
     ]},
    
    # Literature domain
    {"original": "ఈ పద్యం/కవిత రచయిత ఎవరు?",
     "variations": [
         "ఈ రచన ఎవరు రాసారు?",
         "రచయిత పేరు ఏమిటి?",
         "ఈ కవితను ఎవరు వ్రాసారు?"
     ]},
    {"original": "ఈ రచన యొక్క భావం ఏమిటి?",
     "variations": [
         "ఈ కవిత అర్థం ఏమిటి?",
         "పద్యం యొక్క తాత్పర్యం ఏమిటి?",
         "ఈ రచన ద్వారా రచయిత ఏం చెప్పదలిచారు?"
     ]}
]


def paraphrase_question(question: str) -> List[str]:
    """Generate paraphrased versions of a question."""
    paraphrases = [question]  # Include original
    
    # Try direct question word replacement
    for original, alternatives in QUESTION_PARAPHRASES.items():
        if original in question:
            for alt in alternatives:
                new_q = question.replace(original, alt, 1)
                if new_q != question:
                    paraphrases.append(new_q)
    
    # Try template-based paraphrasing
    for template in QUESTION_TEMPLATES:
        if question == template["original"]:
            paraphrases.extend(template["variations"])
        elif similar_questions(question, template["original"]):
            # Apply similar variations with modifications
            for var in template["variations"][:2]:
                paraphrases.append(var)
    
    return list(set(paraphrases))


def similar_questions(q1: str, q2: str) -> bool:
    """Check if two questions are similar (share key words)."""
    words1 = set(q1.split())
    words2 = set(q2.split())
    common = words1 & words2
    return len(common) >= 2


def truncate_context(context: str, answer_start: int, answer_text: str) -> str:
    """Truncate context while keeping the answer."""
    # Split into sentences
    sentences = re.split(r'([.।\n]+)', context)
    
    # Find which sentence contains the answer
    current_pos = 0
    answer_sentence_idx = -1
    
    for idx, sent in enumerate(sentences):
        if current_pos <= answer_start < current_pos + len(sent):
            answer_sentence_idx = idx
            break
        current_pos += len(sent)
    
    if answer_sentence_idx == -1:
        return context
    
    # Include 1-2 sentences before and after the answer sentence
    start_idx = max(0, answer_sentence_idx - 2)
    end_idx = min(len(sentences), answer_sentence_idx + 3)
    
    truncated = ''.join(sentences[start_idx:end_idx])
    
    # Recalculate answer_start for truncated context
    return truncated.strip()


def shuffle_sentences(context: str, answer_text: str) -> Optional[str]:
    """Shuffle non-answer sentences to create variation."""
    # Split into sentences
    sentences = re.split(r'([.।\n]+)', context)
    
    # Pair sentences with their delimiters
    paired = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            paired.append(sentences[i] + sentences[i + 1])
        else:
            paired.append(sentences[i])
    
    # Find sentences without the answer
    answer_idx = -1
    for idx, sent in enumerate(paired):
        if answer_text in sent:
            answer_idx = idx
            break
    
    if answer_idx == -1 or len(paired) < 3:
        return None
    
    # Shuffle non-answer sentences
    other_sentences = [s for i, s in enumerate(paired) if i != answer_idx]
    random.shuffle(other_sentences)
    
    # Reconstruct: put answer sentence somewhere in the middle
    insert_pos = random.randint(1, len(other_sentences))
    other_sentences.insert(insert_pos, paired[answer_idx])
    
    return ' '.join(other_sentences)


def create_answer_variations(context: str, answer_text: str, answer_start: int) -> List[Dict]:
    """Create variations of answer spans."""
    variations = [{"text": answer_text, "answer_start": answer_start}]
    
    # Try to extend or shorten answer span
    words = answer_text.split()
    
    if len(words) > 2:
        # Shorter answer (remove last word)
        shorter = ' '.join(words[:-1])
        short_start = context.find(shorter)
        if short_start >= 0:
            variations.append({"text": shorter, "answer_start": short_start})
    
    # Try to extend answer to include more context
    end_pos = answer_start + len(answer_text)
    if end_pos < len(context) - 1:
        # Check if we can include the next word/phrase
        remaining = context[end_pos:end_pos + 50]
        match = re.match(r'\s*[\u0C00-\u0C7F]+', remaining)
        if match:
            extended = answer_text + match.group(0)
            variations.append({"text": extended, "answer_start": answer_start})
    
    return variations[:3]  # Return max 3 variations


def augment_qa_pair(qa: Dict, multiplier: int = 2) -> List[Dict]:
    """Augment a single QA pair using multiple techniques."""
    augmented = [qa]  # Include original
    
    context = qa.get("context", "")
    question = qa.get("question", "")
    answers = qa.get("answers", [])
    
    if not answers:
        return augmented
    
    original_answer = answers[0]
    answer_text = original_answer.get("text", "")
    answer_start = original_answer.get("answer_start", 0)
    
    # Technique 1: Question paraphrasing
    paraphrases = paraphrase_question(question)
    for para_q in paraphrases[1:multiplier]:  # Skip original
        aug_qa = copy.deepcopy(qa)
        aug_qa["id"] = qa["id"] + f"_para_{paraphrases.index(para_q)}"
        aug_qa["question"] = para_q
        augmented.append(aug_qa)
    
    # Technique 2: Context truncation
    truncated = truncate_context(context, answer_start, answer_text)
    if truncated != context and answer_text in truncated:
        new_start = truncated.find(answer_text)
        if new_start >= 0:
            aug_qa = copy.deepcopy(qa)
            aug_qa["id"] = qa["id"] + "_trunc"
            aug_qa["context"] = truncated
            aug_qa["answers"] = [{"text": answer_text, "answer_start": new_start}]
            augmented.append(aug_qa)
    
    # Technique 3: Sentence shuffling
    shuffled = shuffle_sentences(context, answer_text)
    if shuffled and answer_text in shuffled:
        new_start = shuffled.find(answer_text)
        if new_start >= 0:
            aug_qa = copy.deepcopy(qa)
            aug_qa["id"] = qa["id"] + "_shuf"
            aug_qa["context"] = shuffled
            aug_qa["answers"] = [{"text": answer_text, "answer_start": new_start}]
            augmented.append(aug_qa)
    
    return augmented[:multiplier + 1]


def validate_span(context: str, answer_text: str, answer_start: int) -> bool:
    """Validate that the answer span matches the context exactly."""
    if answer_start < 0 or answer_start >= len(context):
        return False
    return context[answer_start:answer_start + len(answer_text)] == answer_text


def augment_squad_file(input_path: Path, multiplier: int = 2) -> Dict:
    """Augment a SQuAD-format JSON file.
    
    IMPORTANT: Context-modifying augmentations (truncation, shuffling) create
    NEW paragraphs with their own context, so answer_start always matches.
    Question paraphrases stay in the original paragraph.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented_paragraphs = []
    total_original = 0
    total_augmented = 0
    skipped_invalid = 0
    
    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            original_qas = para.get("qas", [])
            total_original += len(original_qas)
            
            # Collect QAs for original context (original + paraphrased questions)
            same_context_qas = []
            # Collect QAs with modified contexts (each gets its own paragraph)
            new_paragraphs = []
            
            for qa in original_qas:
                answers = qa.get("answers", [])
                if not answers:
                    continue
                
                answer_text = answers[0].get("text", "")
                answer_start = answers[0].get("answer_start", 0)
                question = qa.get("question", "")
                
                # Validate original span first
                if not validate_span(context, answer_text, answer_start):
                    # Try to fix by re-finding
                    new_start = context.find(answer_text)
                    if new_start >= 0:
                        answer_start = new_start
                        qa["answers"] = [{"text": answer_text, "answer_start": new_start}]
                    else:
                        skipped_invalid += 1
                        continue
                
                # Add validated original
                same_context_qas.append(qa)
                
                # Technique 1: Question paraphrases (same context)
                paraphrases = paraphrase_question(question)
                for para_q in paraphrases[1:multiplier]:
                    aug_qa = copy.deepcopy(qa)
                    aug_qa["id"] = qa["id"] + f"_p{paraphrases.index(para_q)}"
                    aug_qa["question"] = para_q
                    same_context_qas.append(aug_qa)
                
                # Technique 2: Context truncation (NEW paragraph)
                truncated = truncate_context(context, answer_start, answer_text)
                if truncated != context and answer_text in truncated:
                    new_start = truncated.find(answer_text)
                    if new_start >= 0 and validate_span(truncated, answer_text, new_start):
                        trunc_qa = copy.deepcopy(qa)
                        trunc_qa["id"] = qa["id"] + "_trunc"
                        trunc_qa["answers"] = [{"text": answer_text, "answer_start": new_start}]
                        new_paragraphs.append({
                            "context": truncated,
                            "qas": [trunc_qa]
                        })
                
                # Technique 3: Sentence shuffling (NEW paragraph)
                shuffled = shuffle_sentences(context, answer_text)
                if shuffled and answer_text in shuffled:
                    new_start = shuffled.find(answer_text)
                    if new_start >= 0 and validate_span(shuffled, answer_text, new_start):
                        shuf_qa = copy.deepcopy(qa)
                        shuf_qa["id"] = qa["id"] + "_shuf"
                        shuf_qa["answers"] = [{"text": answer_text, "answer_start": new_start}]
                        new_paragraphs.append({
                            "context": shuffled,
                            "qas": [shuf_qa]
                        })
            
            # Add original context paragraph with all its QAs
            if same_context_qas:
                total_augmented += len(same_context_qas)
                augmented_paragraphs.append({
                    "context": context,
                    "qas": same_context_qas
                })
            
            # Add new paragraphs with modified contexts
            for np in new_paragraphs:
                total_augmented += len(np["qas"])
                augmented_paragraphs.append(np)
    
    if skipped_invalid > 0:
        print(f"   Skipped {skipped_invalid} pairs with unfixable broken spans")
    
    return {
        "version": data.get("version", "2.0"),
        "domain": data.get("domain", "unknown"),
        "augmented_at": datetime.now().isoformat(),
        "original_count": total_original,
        "augmented_count": total_augmented,
        "multiplier": multiplier,
        "data": [{
            "title": "Augmented Telugu QA",
            "paragraphs": augmented_paragraphs
        }]
    }


def augment_all_domains(multiplier: int = 2):
    """Augment QA pairs for all domains."""
    domains = ["news", "government", "literature"]
    
    for domain in domains:
        qa_dir = Path(f"data/domain/{domain}/qa_pairs")
        if not qa_dir.exists():
            print(f"No QA pairs found for {domain}")
            continue
        
        # Find non-raw, non-augmented files
        for json_file in qa_dir.glob("*.json"):
            if json_file.name.startswith("raw_") or json_file.name.startswith("augmented_"):
                continue
            
            print(f"\n📝 Augmenting {json_file.name}")
            augmented = augment_squad_file(json_file, multiplier)
            
            # Save augmented file
            output_path = qa_dir / f"augmented_{json_file.name}"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)
            
            print(f"   Original: {augmented['original_count']} QA pairs")
            print(f"   Augmented: {augmented['augmented_count']} QA pairs")
            print(f"   Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment Telugu QA pairs")
    parser.add_argument("--input", type=str, help="Input JSON file to augment")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--all", action="store_true", help="Augment all domain QA files")
    parser.add_argument("--multiplier", type=int, default=2, 
                        help="Augmentation multiplier (default: 2)")
    parser.add_argument("--verify", action="store_true", help="Verify augmented data")
    
    args = parser.parse_args()
    
    if args.verify:
        for domain in ["news", "government", "literature"]:
            qa_dir = Path(f"data/domain/{domain}/qa_pairs")
            for json_file in qa_dir.glob("augmented_*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📄 {json_file.name}")
                print(f"   Original: {data.get('original_count', 0)}")
                print(f"   Augmented: {data.get('augmented_count', 0)}")
    elif args.all:
        augment_all_domains(args.multiplier)
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"File not found: {input_path}")
            return
        
        augmented = augment_squad_file(input_path, args.multiplier)
        
        output_path = args.output or f"augmented_{input_path.name}"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)
        
        print(f"Augmented {augmented['original_count']} -> {augmented['augmented_count']} QA pairs")
        print(f"Saved to: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
