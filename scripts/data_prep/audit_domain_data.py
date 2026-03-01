"""Quick audit of domain QA data quality."""
import json
from pathlib import Path

domains = {
    'government': 'augmented_government_qa_20260222_122628.json',
    'literature': 'augmented_literature_qa_20260222_122628.json',
    'news': 'augmented_news_qa_20260222_131418.json'
}

for domain, filename in domains.items():
    path = Path(f'data/domain/{domain}/qa_pairs/{filename}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    paragraphs = data['data'][0]['paragraphs']
    
    all_qs = []
    all_as = []
    all_contexts = []
    answer_in_context = 0
    total = 0
    short_context = 0
    empty_answer = 0
    
    for p in paragraphs:
        ctx = p['context']
        all_contexts.append(ctx)
        for qa in p.get('qas', []):
            total += 1
            q = qa['question']
            all_qs.append(q)
            
            if qa['answers']:
                ans = qa['answers'][0]
                a_text = ans['text']
                a_start = ans['answer_start']
                all_as.append(a_text)
                
                if a_start >= 0 and ctx[a_start:a_start+len(a_text)] == a_text:
                    answer_in_context += 1
                
                if not a_text.strip():
                    empty_answer += 1
            else:
                empty_answer += 1
            
            if len(ctx) < 100:
                short_context += 1
    
    unique_qs = len(set(all_qs))
    unique_as = len(set(all_as))
    unique_ctxs = len(set(all_contexts))
    avg_ctx_len = sum(len(c) for c in all_contexts) / len(all_contexts) if all_contexts else 0
    avg_q_len = sum(len(q) for q in all_qs) / len(all_qs) if all_qs else 0
    avg_a_len = sum(len(a) for a in all_as) / len(all_as) if all_as else 0
    
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{domain.upper()} DOMAIN AUDIT")
    print(f"{sep}")
    print(f"Total QA pairs:     {total}")
    print(f"Unique contexts:    {unique_ctxs}")
    print(f"Unique questions:   {unique_qs} ({unique_qs/total*100:.1f}%)")
    print(f"Unique answers:     {unique_as} ({unique_as/total*100:.1f}%)")
    print(f"Answer span valid:  {answer_in_context}/{total} ({answer_in_context/total*100:.1f}%)")
    print(f"Empty answers:      {empty_answer}")
    print(f"Short contexts:     {short_context} (<100 chars)")
    print(f"Avg context len:    {avg_ctx_len:.0f} chars")
    print(f"Avg question len:   {avg_q_len:.0f} chars")
    print(f"Avg answer len:     {avg_a_len:.0f} chars")
    print(f"QA/context ratio:   {total/unique_ctxs:.1f}")
    
    # Check for problematic patterns
    issues = []
    if answer_in_context / total < 0.9:
        issues.append(f"LOW SPAN ACCURACY: {answer_in_context/total*100:.0f}% answers verifiable in context")
    if unique_qs / total < 0.15:
        issues.append(f"LOW Q DIVERSITY: only {unique_qs/total*100:.1f}% unique questions")
    if avg_a_len < 3:
        issues.append(f"VERY SHORT ANSWERS: avg {avg_a_len:.0f} chars")
    if short_context > total * 0.1:
        issues.append(f"MANY SHORT CONTEXTS: {short_context} (<100 chars)")
    if empty_answer > 0:
        issues.append(f"EMPTY ANSWERS: {empty_answer}")
    
    if issues:
        print(f"\n  ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  No critical issues found.")

# Show 3 sample Q&A from each domain
print(f"\n{'='*60}")
print("SAMPLES")
print(f"{'='*60}")
for domain, filename in domains.items():
    path = Path(f'data/domain/{domain}/qa_pairs/{filename}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paragraphs = data['data'][0]['paragraphs']
    print(f"\n--- {domain.upper()} ---")
    shown = 0
    for p in paragraphs:
        if shown >= 3:
            break
        for qa in p.get('qas', []):
            if shown >= 3:
                break
            print(f"  Q: {qa['question'][:80]}")
            if qa['answers']:
                print(f"  A: {qa['answers'][0]['text'][:60]}")
                # Verify span
                ctx = p['context']
                a = qa['answers'][0]
                span_ok = ctx[a['answer_start']:a['answer_start']+len(a['text'])] == a['text']
                print(f"  Span OK: {span_ok}")
            print()
            shown += 1
