"""
Export Domain QA Data for Google Colab LoRA Training

Creates a lightweight zip containing:
- Individual domain QA files (government, literature, news)
- A COMBINED domain_all_qa.json merging all 3 domains into one training set
- Domain-specific validation test sets for before/after comparison

Usage:
    python scripts/export_domain_for_colab.py
"""

import os
import json
import zipfile
import random
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def find_latest_augmented_file(domain: str, project_root: Path) -> Path:
    """Find the latest augmented QA file for a domain."""
    qa_dir = project_root / "data" / "domain" / domain / "qa_pairs"
    
    if not qa_dir.exists():
        return None
    
    # Get all augmented files
    augmented_files = list(qa_dir.glob("augmented_*.json"))
    
    if not augmented_files:
        # Fall back to any QA file
        qa_files = [f for f in qa_dir.glob("*.json") if not f.name.startswith("raw_")]
        if qa_files:
            return max(qa_files, key=lambda x: x.stat().st_mtime)
        return None
    
    # Return the most recent augmented file
    return max(augmented_files, key=lambda x: x.stat().st_mtime)


def load_paragraphs(filepath: Path):
    """Load paragraphs from a SQuAD-format JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('data', [{}])[0].get('paragraphs', [])


def create_combined_dataset(all_paragraphs: dict) -> dict:
    """Merge all domain paragraphs into a single SQuAD-format dataset.
    
    Also creates a held-out validation set (5% per domain) for 
    before/after comparison.
    """
    train_paragraphs = []
    val_paragraphs = []
    
    for domain, paragraphs in all_paragraphs.items():
        # Shuffle and split 95/5 for train/val
        shuffled = list(paragraphs)
        random.seed(42)
        random.shuffle(shuffled)
        
        split_idx = max(1, int(len(shuffled) * 0.05))
        val_paragraphs.extend(shuffled[:split_idx])
        train_paragraphs.extend(shuffled[split_idx:])
    
    # Shuffle the combined train set
    random.seed(42)
    random.shuffle(train_paragraphs)
    
    train_data = {
        "version": "2.0",
        "description": "Combined domain QA (government + literature + news) for LoRA training",
        "generated_at": datetime.now().isoformat(),
        "data": [{"title": "Telugu Domain QA - Combined", "paragraphs": train_paragraphs}]
    }
    
    val_data = {
        "version": "2.0",
        "description": "Domain validation set (held-out 5%) for before/after comparison",
        "generated_at": datetime.now().isoformat(),
        "data": [{"title": "Telugu Domain QA - Validation", "paragraphs": val_paragraphs}]
    }
    
    return train_data, val_data


def export_domain_data():
    """Create a zip file with domain QA data for Colab."""
    project_root = get_project_root()
    
    # Output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"domain-qa-colab-{timestamp}.zip"
    zip_path = project_root / zip_filename
    
    print("="*60)
    print("📦 Exporting Domain QA Data for Colab LoRA Training")
    print("="*60)
    
    domains = ["government", "literature", "news"]
    files_added = 0
    total_qa_pairs = 0
    all_paragraphs = {}
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1) Add individual domain files
        for domain in domains:
            qa_file = find_latest_augmented_file(domain, project_root)
            
            if qa_file and qa_file.exists():
                paragraphs = load_paragraphs(qa_file)
                qa_count = sum(len(p.get('qas', [])) for p in paragraphs)
                total_qa_pairs += qa_count
                all_paragraphs[domain] = paragraphs
                
                # Add individual domain file
                arcname = f"domain_data/{domain}_qa.json"
                zipf.write(qa_file, arcname)
                files_added += 1
                
                print(f"  ✓ {domain}: {qa_count} QA pairs ({qa_file.name})")
            else:
                print(f"  ✗ {domain}: No QA data found")
        
        # 2) Create and add combined training dataset + validation set
        if all_paragraphs:
            train_data, val_data = create_combined_dataset(all_paragraphs)
            
            train_paras = train_data['data'][0]['paragraphs']
            train_qa_count = sum(len(p.get('qas', [])) for p in train_paras)
            
            val_paras = val_data['data'][0]['paragraphs']
            val_qa_count = sum(len(p.get('qas', [])) for p in val_paras)
            
            # Write combined train
            train_json = json.dumps(train_data, ensure_ascii=False, indent=2)
            zipf.writestr("domain_data/domain_all_train.json", train_json)
            print(f"\n  ✓ COMBINED TRAIN: {train_qa_count} QA pairs ({len(train_paras)} contexts)")
            
            # Write validation set
            val_json = json.dumps(val_data, ensure_ascii=False, indent=2)
            zipf.writestr("domain_data/domain_all_val.json", val_json)
            print(f"  ✓ VALIDATION SET:  {val_qa_count} QA pairs ({len(val_paras)} contexts)")
        
        # 3) Add per-domain test sets for before/after F1/EM comparison
        test_dir = project_root / "data" / "domain"
        for domain in domains:
            test_file = test_dir / domain / "test" / f"{domain}_test.json"
            if test_file.exists():
                arcname = f"domain_data/{domain}_test.json"
                zipf.write(test_file, arcname)
                # Count QAs
                with open(test_file, 'r', encoding='utf-8') as f:
                    td = json.load(f)
                tqa = sum(len(p.get('qas', [])) for p in td.get('data', [{}])[0].get('paragraphs', []))
                print(f"  ✓ {domain} TEST: {tqa} QA pairs")
            else:
                print(f"  ✗ {domain} test set not found at {test_file}")

        # 4) Also save validation test set locally for app usage
        if all_paragraphs:
            for domain in domains:
                domain_test_dir = test_dir / domain / "test"
                domain_test_dir.mkdir(parents=True, exist_ok=True)
            
            # Save combined val set locally
            val_path = test_dir / "domain_validation.json"
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)
            print(f"\n  Saved local validation set: {val_path}")
    
    # Calculate zip size
    zip_size = zip_path.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"✅ Export complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {zip_path}")
    print(f"Domains: {files_added}")
    print(f"Total QA pairs: {total_qa_pairs}")
    print(f"Size: {zip_size:.2f} MB")
    
    print(f"\n📋 Training Plan:")
    print(f"  Run 1: MuRIL + LoRA on domain_all_train.json")
    print(f"  Run 2: mBERT + LoRA on domain_all_train.json")
    print(f"\n📋 Next Steps:")
    print(f"1. Open Google Colab")
    print(f"2. Open notebooks/04_domain_lora_training.ipynb")
    print(f"3. Upload {zip_filename} when prompted")
    print(f"4. Run 1: Select MuRIL, train on combined data")
    print(f"5. Run 2: Select mBERT, train on combined data")
    
    return zip_path


if __name__ == "__main__":
    export_domain_data()
