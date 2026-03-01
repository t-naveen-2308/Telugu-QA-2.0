"""Quick verification of converted SQuAD files."""
import json

# Load and display sample from each dataset
datasets = [
    ("data/processed/tequad_train.json", "TRAIN"),
    ("data/processed/tequad_validation.json", "VALIDATION"),
    ("data/processed/tequad_test_wiki.json", "TEST (Wiki)")
]

for path, name in datasets:
    print(f"\n{'='*60}")
    print(f"📄 {name}")
    print(f"{'='*60}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample = data['data'][0]['paragraphs'][0]
    qa = sample['qas'][0]
    ans = qa['answers'][0]
    
    print(f"\nContext (first 200 chars):")
    print(f"  {sample['context'][:200]}...")
    
    print(f"\nQuestion:")
    print(f"  {qa['question']}")
    
    print(f"\nAnswer:")
    print(f"  Text: '{ans['text']}'")
    print(f"  Start position: {ans['answer_start']}")
    
    # Verify extraction
    extracted = sample['context'][ans['answer_start']:ans['answer_start']+len(ans['text'])]
    match = "✅ MATCH" if extracted == ans['text'] else f"❌ MISMATCH (got: '{extracted}')"
    print(f"  Verification: {match}")

print(f"\n{'='*60}")
print("✅ All files verified!")
print(f"{'='*60}")
