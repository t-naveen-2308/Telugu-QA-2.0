"""
Training Module for Telugu QA System

Wraps HuggingFace Trainer with QA-specific configurations.
Optimized for Colab Free Tier (T4 GPU).
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
    default_data_collator
)
from datasets import Dataset

from src.utils.helpers import load_config, get_project_root, is_colab
from src.data.tequad_loader import load_tequad_dataset
from src.data.preprocessing import preprocess_for_training, preprocess_validation
from src.models.model_factory import load_model_and_tokenizer
from src.training.metrics import compute_qa_metrics, postprocess_qa_predictions


@dataclass
class TeluguQATrainingConfig:
    """Configuration for Telugu QA training."""
    
    # Model
    model_key: str = "muril"
    
    # Training params
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    
    # Preprocessing
    max_seq_length: int = 384
    doc_stride: int = 128
    
    # Optimization
    fp16: bool = True
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 2
    
    # Output
    output_dir: str = "models/checkpoints"
    
    @classmethod
    def from_config_file(cls) -> "TeluguQATrainingConfig":
        """Load configuration from YAML file."""
        config = load_config("training_config")
        training = config.get("training", {})
        preprocessing = config.get("preprocessing", {})
        output = config.get("output", {})
        
        return cls(
            learning_rate=training.get("learning_rate", 3e-5),
            num_train_epochs=training.get("num_train_epochs", 3),
            per_device_train_batch_size=training.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=training.get("per_device_eval_batch_size", 16),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 2),
            max_seq_length=preprocessing.get("max_seq_length", 384),
            doc_stride=preprocessing.get("doc_stride", 128),
            fp16=training.get("fp16", True),
            warmup_ratio=training.get("warmup_ratio", 0.1),
            weight_decay=training.get("weight_decay", 0.01),
            evaluation_strategy=training.get("evaluation_strategy", "steps"),
            eval_steps=training.get("eval_steps", 1000),
            save_steps=training.get("save_steps", 1000),
            save_total_limit=training.get("save_total_limit", 2),
            output_dir=output.get("output_dir", "models/checkpoints")
        )


def create_training_args(
    config: TeluguQATrainingConfig,
    model_key: str,
    output_dir: Optional[str] = None
) -> TrainingArguments:
    """
    Create TrainingArguments from config.
    
    Args:
        config: Training configuration
        model_key: Model identifier for output directory
        output_dir: Override output directory
        
    Returns:
        HuggingFace TrainingArguments
    """
    if output_dir is None:
        project_root = get_project_root()
        output_dir = project_root / config.output_dir / model_key
    
    return TrainingArguments(
        output_dir=str(output_dir),
        
        # Training
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimization
        fp16=config.fp16 and torch.cuda.is_available(),
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        
        # Evaluation & Saving
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.evaluation_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Logging
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        
        # Other
        remove_unused_columns=True,
        dataloader_num_workers=2 if not is_colab() else 0,
    )


class TeluguQATrainer:
    """
    High-level trainer for Telugu QA models.
    
    Usage:
        trainer = TeluguQATrainer(model_key="muril")
        trainer.train()
        trainer.evaluate()
        trainer.save("path/to/save")
    """
    
    def __init__(
        self,
        model_key: str = "muril",
        config: Optional[TeluguQATrainingConfig] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_key: Model to train ("muril", "indicbert", "xlmr")
            config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.model_key = model_key
        self.config = config or TeluguQATrainingConfig.from_config_file()
        self.config.model_key = model_key
        
        # Load model and tokenizer
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(model_key)
        
        # Datasets (lazy loaded)
        self._train_dataset = None
        self._eval_dataset = None
        self._raw_datasets = None
        
        # HuggingFace Trainer
        self._trainer = None
    
    def prepare_datasets(self) -> None:
        """Load and preprocess datasets."""
        print("Loading datasets...")
        self._raw_datasets = load_tequad_dataset()
        
        print("Preprocessing training data...")
        self._train_dataset = preprocess_for_training(
            self._raw_datasets["train"],
            self.tokenizer,
            max_length=self.config.max_seq_length,
            doc_stride=self.config.doc_stride,
            num_proc=1  # Use 1 for Colab compatibility
        )
        
        print("Preprocessing validation data...")
        self._eval_dataset = preprocess_validation(
            self._raw_datasets["validation"],
            self.tokenizer,
            max_length=self.config.max_seq_length,
            doc_stride=self.config.doc_stride,
            num_proc=1
        )
        
        print(f"Train examples: {len(self._train_dataset)}")
        print(f"Eval examples: {len(self._eval_dataset)}")
    
    def _create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer instance."""
        if self._train_dataset is None:
            self.prepare_datasets()
        
        training_args = create_training_args(self.config, self.model_key)
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
        )
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training metrics
        """
        if self._trainer is None:
            self._trainer = self._create_trainer()
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.model_key}")
        print(f"{'='*60}")
        
        train_result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self._trainer.save_model()
        
        return train_result.metrics
    
    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataset: Dataset to evaluate on (defaults to validation)
            
        Returns:
            Evaluation metrics (EM, F1)
        """
        if self._trainer is None:
            self._trainer = self._create_trainer()
        
        print("\nEvaluating...")
        metrics = self._trainer.evaluate(eval_dataset=dataset)
        
        return metrics
    
    def save(self, output_path: str) -> None:
        """Save model and tokenizer."""
        print(f"Saving model to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
    
    def push_to_hub(self, repo_name: str, private: bool = True) -> None:
        """Push model to HuggingFace Hub."""
        self.model.push_to_hub(repo_name, private=private)
        self.tokenizer.push_to_hub(repo_name, private=private)


def create_trainer(model_key: str = "muril") -> TeluguQATrainer:
    """
    Factory function to create a trainer.
    
    Args:
        model_key: Model to train
        
    Returns:
        TeluguQATrainer instance
    """
    return TeluguQATrainer(model_key=model_key)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Telugu QA Model")
    parser.add_argument("--model", type=str, default="muril",
                        choices=["muril", "indicbert", "xlmr"],
                        help="Model to train")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    trainer = create_trainer(args.model)
    trainer.train(resume_from_checkpoint=args.resume)
