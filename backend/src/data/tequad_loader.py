"""
TeQuAD Dataset Loader

Loads the converted SQuAD-format JSON files for training.
Compatible with HuggingFace datasets and transformers.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict

from utils.helpers import get_project_root, load_config


def load_json_file(file_path: Union[str, Path]) -> Dict:
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def squad_to_dataset(squad_data: Dict) -> Dataset:
    """
    Convert SQuAD-format dict to HuggingFace Dataset.
    
    Args:
        squad_data: Dictionary in SQuAD format
        
    Returns:
        HuggingFace Dataset with columns: id, context, question, answers
    """
    examples = {
        "id": [],
        "context": [],
        "question": [],
        "answers": []
    }
    
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            
            for qa in paragraph["qas"]:
                examples["id"].append(qa["id"])
                examples["context"].append(context)
                examples["question"].append(qa["question"])
                
                # Format answers as expected by HuggingFace
                answers = {
                    "text": [ans["text"] for ans in qa["answers"]],
                    "answer_start": [ans["answer_start"] for ans in qa["answers"]]
                }
                examples["answers"].append(answers)
    
    return Dataset.from_dict(examples)


def load_tequad_dataset(
    split: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None
) -> Union[Dataset, DatasetDict]:
    """
    Load the TeQuAD dataset.
    
    Args:
        split: Which split to load ("train", "validation", "test", or None for all)
        data_dir: Custom data directory (defaults to project's data/processed)
        
    Returns:
        Dataset or DatasetDict with the requested splits
    """
    if data_dir is None:
        project_root = get_project_root()
        data_dir = project_root / "data" / "processed"
    else:
        data_dir = Path(data_dir)
    
    # Map split names to files
    split_files = {
        "train": data_dir / "tequad_train.json",
        "validation": data_dir / "tequad_validation.json",
        "test": data_dir / "tequad_test_wiki.json"
    }
    
    if split is not None:
        # Load single split
        if split not in split_files:
            raise ValueError(f"Unknown split: {split}. Choose from {list(split_files.keys())}")
        
        file_path = split_files[split]
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        squad_data = load_json_file(file_path)
        return squad_to_dataset(squad_data)
    
    else:
        # Load all splits
        datasets = {}
        for split_name, file_path in split_files.items():
            if file_path.exists():
                squad_data = load_json_file(file_path)
                datasets[split_name] = squad_to_dataset(squad_data)
        
        return DatasetDict(datasets)


class TeQuADDataset:
    """
    Wrapper class for TeQuAD dataset with convenient methods.
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize TeQuAD dataset.
        
        Args:
            data_dir: Custom data directory
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._datasets = None
    
    @property
    def datasets(self) -> DatasetDict:
        """Lazy load all datasets."""
        if self._datasets is None:
            self._datasets = load_tequad_dataset(data_dir=self.data_dir)
        
        return self._datasets
    
    @property
    def train(self) -> Dataset:
        """Get training dataset."""
        return self.datasets["train"]
    
    @property
    def validation(self) -> Dataset:
        """Get validation dataset."""
        return self.datasets["validation"]
    
    @property
    def test(self) -> Dataset:
        """Get test dataset."""
        return self.datasets["test"]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {}
        for split_name, dataset in self.datasets.items():
            stats[split_name] = {
                "num_examples": len(dataset),
                "avg_context_length": sum(len(ex["context"]) for ex in dataset) / len(dataset),
                "avg_question_length": sum(len(ex["question"]) for ex in dataset) / len(dataset),
                "avg_answer_length": sum(
                    len(ex["answers"]["text"][0]) if ex["answers"]["text"] else 0 
                    for ex in dataset
                ) / len(dataset)
            }
        return stats
    
    def __repr__(self) -> str:
        splits = list(self.datasets.keys())
        sizes = [len(self.datasets[s]) for s in splits]
        return f"TeQuADDataset(splits={splits}, sizes={sizes})"


# Quick test
if __name__ == "__main__":
    print("Loading TeQuAD dataset...")
    dataset = TeQuADDataset()
    print(dataset)
    print("\nStatistics:")
    for split, stats in dataset.get_statistics().items():
        print(f"\n{split}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nSample from train:")
    sample = dataset.train[0]
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answers']['text'][0]}")
