"""
TeQuAD Data Format Analyzer

Analyzes the actual TeQuAD data format (text files, not JSON).
This script reveals the data structure for building the preprocessing pipeline.

Data Format Discovery:
- Each line = 1 sample
- Files are aligned (line N in each file = same sample)
- Span format: start_word_idx \t end_word_idx (word-level, not char-level!)
"""

import os
from pathlib import Path
from collections import Counter
from typing import List, Tuple


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_lines(file_path: Path) -> List[str]:
    """Load all lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def analyze_alignment(data_dir: Path, prefix: str = "real_") -> dict:
    """Check if all files have the same number of lines (aligned)."""
    print(f"\n{'='*60}")
    print(f"📊 File Alignment Check - {data_dir.name}")
    print(f"{'='*60}")
    
    files = list(data_dir.glob(f"{prefix}*.txt"))
    line_counts = {}
    
    for f in sorted(files):
        lines = load_lines(f)
        line_counts[f.name] = len(lines)
        print(f"  {f.name}: {len(lines):,} lines")
    
    counts = list(line_counts.values())
    if len(set(counts)) == 1:
        print(f"\n  ✅ All files aligned with {counts[0]:,} samples")
    else:
        print(f"\n  ❌ Misaligned files!")
    
    return line_counts


def verify_span_format(data_dir: Path, prefix: str = "real_") -> None:
    """Analyze span format and verify against context/answer."""
    print(f"\n{'='*60}")
    print(f"🎯 Span Format Analysis - {data_dir.name}")
    print(f"{'='*60}")
    
    # Load files
    con_file = data_dir / f"{prefix}con_tel.txt"
    ans_file = data_dir / f"{prefix}ans_tel.txt"
    span_file = data_dir / f"{prefix}span_tel.txt"
    
    if not all(f.exists() for f in [con_file, span_file]):
        # Try alternate naming
        con_file = data_dir / "real_con_tel.txt"
        ans_file = data_dir / "real_ans_tel.txt" if (data_dir / "real_ans_tel.txt").exists() else data_dir / "corrected_ans_tel.txt"
        span_file = data_dir / "real_span_tel.txt" if (data_dir / "real_span_tel.txt").exists() else data_dir / "span_tel.txt"
    
    if not con_file.exists():
        print(f"  ⚠️ Context file not found")
        return
    
    contexts = load_lines(con_file)
    spans = load_lines(span_file)
    answers = load_lines(ans_file) if ans_file.exists() else None
    
    print(f"\n  Span format examples (first 5):")
    for i in range(min(5, len(spans))):
        print(f"    Line {i}: '{spans[i]}'")
    
    # Parse span format
    print(f"\n  Verifying span extraction (checking first 10 samples):")
    matches = 0
    partial_matches = 0
    mismatches = 0
    
    for i in range(min(10, len(contexts))):
        try:
            parts = spans[i].split()
            if len(parts) >= 2:
                start_idx = int(parts[0])
                end_idx = int(parts[1])
                
                # Split context into words
                words = contexts[i].split()
                
                # Extract answer span using word indices
                if start_idx < len(words) and end_idx < len(words):
                    extracted = ' '.join(words[start_idx:end_idx+1])
                    actual_answer = answers[i] if answers else "N/A"
                    
                    if extracted == actual_answer:
                        status = "✅ EXACT"
                        matches += 1
                    elif actual_answer in extracted or extracted in actual_answer:
                        status = "🔶 PARTIAL"
                        partial_matches += 1
                    else:
                        status = "❌ MISMATCH"
                        mismatches += 1
                    
                    print(f"\n    Sample {i}:")
                    print(f"      Span: {start_idx}-{end_idx}")
                    print(f"      Extracted: '{extracted[:60]}...' " if len(extracted) > 60 else f"      Extracted: '{extracted}'")
                    print(f"      Answer:    '{actual_answer[:60]}...' " if len(actual_answer) > 60 else f"      Answer:    '{actual_answer}'")
                    print(f"      Status: {status}")
                else:
                    print(f"\n    Sample {i}: Index out of bounds (start={start_idx}, end={end_idx}, words={len(words)})")
                    
        except (ValueError, IndexError) as e:
            print(f"\n    Sample {i}: Parse error - {e}")
    
    print(f"\n  Summary: {matches} exact, {partial_matches} partial, {mismatches} mismatches")


def analyze_text_lengths(data_dir: Path, prefix: str = "real_") -> None:
    """Analyze text lengths for model configuration."""
    print(f"\n{'='*60}")
    print(f"📏 Text Length Analysis - {data_dir.name}")
    print(f"{'='*60}")
    
    con_file = data_dir / f"{prefix}con_tel.txt"
    que_file = data_dir / f"{prefix}que_tel.txt"
    ans_file = data_dir / f"{prefix}ans_tel.txt"
    
    if not con_file.exists():
        con_file = data_dir / "real_con_tel.txt"
        que_file = data_dir / "real_que_tel.txt"
        ans_file = data_dir / "real_ans_tel.txt" if (data_dir / "real_ans_tel.txt").exists() else data_dir / "corrected_ans_tel.txt"
    
    for file_path, name in [(con_file, "Context"), (que_file, "Question"), (ans_file, "Answer")]:
        if not file_path.exists():
            print(f"\n  {name}: File not found")
            continue
            
        lines = load_lines(file_path)
        
        # Character lengths
        char_lengths = [len(l) for l in lines]
        
        # Word lengths
        word_lengths = [len(l.split()) for l in lines]
        
        if not char_lengths:
            continue
            
        char_sorted = sorted(char_lengths)
        word_sorted = sorted(word_lengths)
        n = len(char_lengths)
        
        print(f"\n  {name}:")
        print(f"    Samples: {n:,}")
        print(f"    Characters - Min: {min(char_lengths)}, Max: {max(char_lengths)}, Avg: {sum(char_lengths)/n:.0f}")
        print(f"    Characters - P90: {char_sorted[int(n*0.9)]}, P95: {char_sorted[int(n*0.95)]}, P99: {char_sorted[int(n*0.99)]}")
        print(f"    Words - Min: {min(word_lengths)}, Max: {max(word_lengths)}, Avg: {sum(word_lengths)/n:.0f}")
        print(f"    Words - P90: {word_sorted[int(n*0.9)]}, P95: {word_sorted[int(n*0.95)]}")


def analyze_telugu_encoding(data_dir: Path, prefix: str = "real_") -> None:
    """Analyze Telugu text encoding and special characters."""
    print(f"\n{'='*60}")
    print(f"🔤 Telugu Encoding Analysis - {data_dir.name}")
    print(f"{'='*60}")
    
    con_file = data_dir / f"{prefix}con_tel.txt"
    if not con_file.exists():
        con_file = data_dir / "real_con_tel.txt"
    
    if not con_file.exists():
        print("  Context file not found")
        return
    
    contexts = load_lines(con_file)[:100]  # Sample first 100
    
    special_chars = Counter()
    telugu_chars = Counter()
    
    TELUGU_RANGE = range(0x0C00, 0x0C80)
    
    for ctx in contexts:
        for char in ctx:
            code = ord(char)
            if code in TELUGU_RANGE:
                telugu_chars[char] += 1
            elif not char.isalnum() and not char.isspace():
                special_chars[char] += 1
    
    print(f"\n  Top Telugu characters:")
    for char, count in telugu_chars.most_common(15):
        print(f"    '{char}' (U+{ord(char):04X}): {count}")
    
    print(f"\n  Special/punctuation characters:")
    for char, count in special_chars.most_common(15):
        char_repr = repr(char) if ord(char) < 32 or ord(char) in [0x200c, 0x200d, 0x200b] else char
        print(f"    {char_repr} (U+{ord(char):04X}): {count}")
    
    # Check for zero-width characters
    problematic = {
        '\u200c': 'ZWNJ (Zero Width Non-Joiner)',
        '\u200d': 'ZWJ (Zero Width Joiner)',
        '\u200b': 'ZWSP (Zero Width Space)',
    }
    
    print(f"\n  ⚠️ Zero-width characters found:")
    for char, name in problematic.items():
        if char in special_chars:
            print(f"    - {name}: {special_chars[char]} occurrences")


def print_sample_triplets(data_dir: Path, prefix: str = "real_", n: int = 3) -> None:
    """Print sample Context-Question-Answer triplets."""
    print(f"\n{'='*60}")
    print(f"📝 Sample Triplets - {data_dir.name}")
    print(f"{'='*60}")
    
    con_file = data_dir / f"{prefix}con_tel.txt"
    que_file = data_dir / f"{prefix}que_tel.txt"
    ans_file = data_dir / f"{prefix}ans_tel.txt"
    
    if not con_file.exists():
        con_file = data_dir / "real_con_tel.txt"
        que_file = data_dir / "real_que_tel.txt"
        ans_file = data_dir / "real_ans_tel.txt" if (data_dir / "real_ans_tel.txt").exists() else data_dir / "corrected_ans_tel.txt"
    
    contexts = load_lines(con_file)
    questions = load_lines(que_file)
    answers = load_lines(ans_file)
    
    for i in range(min(n, len(contexts))):
        print(f"\n  {'─'*50}")
        print(f"  Sample {i+1}:")
        print(f"  {'─'*50}")
        
        ctx = contexts[i][:300] + "..." if len(contexts[i]) > 300 else contexts[i]
        print(f"\n  Context:\n    {ctx}")
        print(f"\n  Question:\n    {questions[i]}")
        print(f"\n  Answer:\n    {answers[i]}")


def generate_recommendations() -> None:
    """Print recommendations based on analysis."""
    print(f"\n{'='*60}")
    print(f"💡 RECOMMENDATIONS FOR PREPROCESSING")
    print(f"{'='*60}")
    
    print("""
  Based on the TeQuAD data format analysis:

  1. DATA FORMAT:
     - Text files (NOT JSON) - one line per sample
     - Files are aligned (same line number = same sample)
     - Span format: word-based indices (start_word, end_word)
     - NOT SQuAD format - needs conversion!

  2. PREPROCESSING PIPELINE:
     a) Load aligned text files (context, question, answer, span)
     b) Convert word-level spans to character-level spans
     c) Verify answer text matches span extraction
     d) Convert to HuggingFace datasets format
     e) Apply Telugu Unicode normalization

  3. SPAN HANDLING:
     - Spans are WORD indices, not character indices!
     - Need to convert: word_idx → char_offset
     - Formula: char_start = sum(len(word)+1 for word in words[:start_idx])

  4. RECOMMENDED MODEL CONFIG:
     - max_length: 384 (based on P95 context length)
     - doc_stride: 128 (for sliding window)
     - max_answer_length: 30 words

  5. FILES TO CREATE:
     - src/data/tequad_loader.py (load text files)
     - src/data/preprocessing.py (convert to SQuAD format)
     - src/data/squad_converter.py (HuggingFace format)
""")


def main():
    print("="*60)
    print("🔍 TeQuAD Data Format Analyzer")
    print("="*60)
    
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    
    train_dir = raw_dir / "Train"
    test_dir = raw_dir / "Test"
    
    # Analyze Train data
    if train_dir.exists():
        analyze_alignment(train_dir, "real_")
        verify_span_format(train_dir, "real_")
        analyze_text_lengths(train_dir, "real_")
        analyze_telugu_encoding(train_dir, "real_")
        print_sample_triplets(train_dir, "real_", n=2)
    else:
        print(f"\n⚠️ Train directory not found: {train_dir}")
    
    # Analyze Test data
    test_translated = test_dir / "Translated & Corrected"
    if test_translated.exists():
        analyze_alignment(test_translated, "")
        verify_span_format(test_translated, "")
        analyze_text_lengths(test_translated, "")
    else:
        print(f"\n⚠️ Test directory not found: {test_translated}")
    
    # Print recommendations
    generate_recommendations()


if __name__ == "__main__":
    main()
