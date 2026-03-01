"""
Download Telugu datasets from Kaggle and HuggingFace for domain expansion.

Datasets:
- Telugu News (Kaggle): ~10K news articles from Andhra Jyothi
- Telugu Books (Kaggle): 500 novels from TeluguOne
- AI4Bharat IndicQA (HuggingFace): Telugu QA benchmark
- AI4Bharat Sangraha (HuggingFace): Large Telugu corpus

Usage:
    python scripts/data_collection/download_kaggle.py --all
    python scripts/data_collection/download_kaggle.py --dataset news
    python scripts/data_collection/download_kaggle.py --verify
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Kaggle API token from functional requirements
KAGGLE_API_TOKEN = ""

# Dataset configurations
KAGGLE_DATASETS = {
    "news": {
        "dataset": "sudalairajkumar/telugu-nlp",
        "description": "Telugu News from Andhra Jyothi (~10K articles)",
        "output_dir": "data/domain/news/raw",
        "files": ["telugu_news.csv"]
    },
    "books": {
        "dataset": "sudalairajkumar/telugu-nlp", 
        "description": "Telugu Books from TeluguOne (500 novels)",
        "output_dir": "data/domain/literature/raw",
        "files": ["telugu_books.csv"]
    }
}

HUGGINGFACE_DATASETS = {
    "indicqa": {
        "dataset": "ai4bharat/IndicQA",
        "description": "Telugu QA benchmark (~1K pairs)",
        "output_dir": "data/domain/qa_benchmark",
        "subset": "indicqa.te"
    },
    "sangraha": {
        "dataset": "ai4bharat/sangraha",
        "description": "Large Telugu corpus (3.7B tokens)",
        "output_dir": "data/domain/corpus",
        "subset": "telugu"
    }
}


def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Check if credentials already exist
    if kaggle_json.exists():
        print("Kaggle credentials already configured.")
        return True
    
    # Create credentials file
    # Note: The token format is KGAT_xxx, but kaggle.json needs username and key
    # User needs to get these from kaggle.com/account
    print("\n⚠️  Kaggle API Setup Required")
    print("=" * 50)
    print("The Kaggle API token provided needs to be configured.")
    print("\nTo set up Kaggle API:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'") 
    print("4. Download kaggle.json and place it in ~/.kaggle/")
    print(f"   (Path: {kaggle_dir})")
    print("\nAlternatively, set environment variables:")
    print("  KAGGLE_USERNAME=your_username")
    print("  KAGGLE_KEY=your_api_key")
    print("=" * 50)
    
    return False


def download_kaggle_dataset(dataset_key: str, force: bool = False) -> bool:
    """Download a dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Installing kaggle package...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        from kaggle.api.kaggle_api_extended import KaggleApi
    
    if dataset_key not in KAGGLE_DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(KAGGLE_DATASETS.keys())}")
        return False
    
    config = KAGGLE_DATASETS[dataset_key]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if not force and any(output_dir.iterdir()):
        print(f"✓ {dataset_key}: Already downloaded in {output_dir}")
        return True
    
    print(f"\nDownloading {config['description']}...")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Output: {output_dir}")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        api.dataset_download_files(
            config["dataset"],
            path=str(output_dir),
            unzip=True
        )
        
        print(f"✓ Downloaded {dataset_key} successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_key}: {e}")
        return False


def download_huggingface_dataset(dataset_key: str, force: bool = False) -> bool:
    """Download a dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets package...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset
    
    if dataset_key not in HUGGINGFACE_DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {list(HUGGINGFACE_DATASETS.keys())}")
        return False
    
    config = HUGGINGFACE_DATASETS[dataset_key]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{dataset_key}.json"
    
    # Check if already downloaded
    if not force and output_file.exists():
        print(f"✓ {dataset_key}: Already downloaded at {output_file}")
        return True
    
    print(f"\nDownloading {config['description']}...")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Subset: {config.get('subset', 'default')}")
    print(f"  Output: {output_dir}")
    
    try:
        # Handle different dataset structures
        if dataset_key == "indicqa":
            dataset = load_dataset(
                config["dataset"],
                config["subset"],
                trust_remote_code=True
            )
            # Save as JSON
            if "test" in dataset:
                data = [dict(item) for item in dataset["test"]]
            else:
                data = [dict(item) for item in dataset["train"]]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Downloaded {len(data)} samples to {output_file}")
            
        elif dataset_key == "sangraha":
            # Sangraha is very large, download streaming
            print("  Note: Sangraha is large (3.7B tokens). Downloading sample...")
            dataset = load_dataset(
                config["dataset"],
                "verified",  # Use verified subset
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            # Filter for Telugu and take sample
            telugu_samples = []
            for i, item in enumerate(dataset):
                if item.get("source_lang", "") == "te" or "telugu" in str(item).lower():
                    telugu_samples.append({
                        "text": item.get("text", ""),
                        "source": item.get("source", ""),
                        "id": i
                    })
                if len(telugu_samples) >= 10000:  # Limit for initial testing
                    break
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(telugu_samples, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Downloaded {len(telugu_samples)} Telugu samples to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_key}: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_all(force: bool = False):
    """Download all datasets."""
    print("=" * 60)
    print("Telugu Domain Expansion - Dataset Download")
    print("=" * 60)
    
    results = {}
    
    # Setup Kaggle credentials
    kaggle_ready = setup_kaggle_credentials()
    
    # Download Kaggle datasets
    print("\n📦 Kaggle Datasets")
    print("-" * 40)
    if kaggle_ready or os.environ.get("KAGGLE_USERNAME"):
        for key in KAGGLE_DATASETS:
            results[f"kaggle_{key}"] = download_kaggle_dataset(key, force)
    else:
        print("Skipping Kaggle datasets (credentials not configured)")
        for key in KAGGLE_DATASETS:
            results[f"kaggle_{key}"] = False
    
    # Download HuggingFace datasets
    print("\n🤗 HuggingFace Datasets")
    print("-" * 40)
    for key in HUGGINGFACE_DATASETS:
        results[f"hf_{key}"] = download_huggingface_dataset(key, force)
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    success = sum(1 for v in results.values() if v)
    total = len(results)
    
    for key, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {key}")
    
    print(f"\nTotal: {success}/{total} datasets downloaded")
    
    return success == total


def verify_downloads():
    """Verify downloaded datasets."""
    print("=" * 60)
    print("Verifying Downloaded Datasets")
    print("=" * 60)
    
    all_dirs = [
        "data/domain/news/raw",
        "data/domain/literature/raw",
        "data/domain/qa_benchmark",
        "data/domain/corpus"
    ]
    
    for dir_path in all_dirs:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*"))
            if files:
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"✓ {dir_path}: {len(files)} files ({size_mb:.2f} MB)")
                for f in files[:3]:  # Show first 3 files
                    print(f"    - {f.name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
            else:
                print(f"⚠ {dir_path}: Empty")
        else:
            print(f"✗ {dir_path}: Not found")


def main():
    parser = argparse.ArgumentParser(description="Download Telugu datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, help="Download specific dataset (news, books, indicqa, sangraha)")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_downloads()
    elif args.all:
        download_all(args.force)
    elif args.dataset:
        if args.dataset in KAGGLE_DATASETS:
            download_kaggle_dataset(args.dataset, args.force)
        elif args.dataset in HUGGINGFACE_DATASETS:
            download_huggingface_dataset(args.dataset, args.force)
        else:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(KAGGLE_DATASETS.keys()) + list(HUGGINGFACE_DATASETS.keys())}")
    else:
        parser.print_help()
        print("\n📋 Available datasets:")
        print("\nKaggle:")
        for key, config in KAGGLE_DATASETS.items():
            print(f"  - {key}: {config['description']}")
        print("\nHuggingFace:")
        for key, config in HUGGINGFACE_DATASETS.items():
            print(f"  - {key}: {config['description']}")


if __name__ == "__main__":
    main()
