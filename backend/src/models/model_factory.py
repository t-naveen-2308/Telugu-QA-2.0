"""
Model Factory for Telugu QA System

Provides unified interface to load different transformer models
for Question Answering: MuRIL, IndicBERT, XLM-RoBERTa.
"""

from typing import Tuple, Dict, Optional, List
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch

from utils.helpers import load_config, get_device

# Model registry with HuggingFace model names
MODEL_REGISTRY = {
    "muril": {
        "name": "google/muril-base-cased",
        "display_name": "MuRIL",
        "description": "Best for Indian languages including Telugu"
    },
    "indicbert": {
        "name": "ai4bharat/indic-bert",
        "display_name": "IndicBERT", 
        "description": "Lightweight model for Indian languages"
    },
    "xlmr": {
        "name": "xlm-roberta-base",
        "display_name": "XLM-RoBERTa",
        "description": "Cross-lingual model with broad language coverage"
    },
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "display_name": "mBERT",
        "description": "Google's multilingual BERT covering 104 languages"
    }
}


def get_available_models() -> List[str]:
    """Get list of available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_key: str) -> Dict:
    """Get information about a specific model."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {get_available_models()}")
    return MODEL_REGISTRY[model_key]


def load_model_and_tokenizer(
    model_key: str = "muril",
    from_checkpoint: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer for Question Answering.
    
    Args:
        model_key: Key from MODEL_REGISTRY ("muril", "indicbert", "xlmr")
        from_checkpoint: Path to a fine-tuned checkpoint (optional)
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    # Get model name from registry
    model_info = get_model_info(model_key)
    model_name = model_info["name"]
    
    print(f"Loading {model_info['display_name']}...")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    
    # Load from checkpoint or pretrained
    if from_checkpoint:
        print(f"  Loading from checkpoint: {from_checkpoint}")
        model = AutoModelForQuestionAnswering.from_pretrained(from_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(from_checkpoint)
    else:
        print(f"  Loading pretrained model...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move to device
    model = model.to(device)
    
    print(f"  ✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


class ModelFactory:
    """
    Factory class for managing model loading and caching.
    """
    
    def __init__(self):
        self._cache = {}
    
    def get_model(
        self,
        model_key: str = "muril",
        from_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Get a model and tokenizer, with optional caching.
        
        Args:
            model_key: Key from MODEL_REGISTRY
            from_checkpoint: Path to checkpoint
            device: Device to use
            use_cache: Whether to cache loaded models
            
        Returns:
            Tuple of (model, tokenizer)
        """
        cache_key = f"{model_key}_{from_checkpoint}_{device}"
        
        if use_cache and cache_key in self._cache:
            print(f"Using cached model: {model_key}")
            return self._cache[cache_key]
        
        model, tokenizer = load_model_and_tokenizer(
            model_key=model_key,
            from_checkpoint=from_checkpoint,
            device=device
        )
        
        if use_cache:
            self._cache[cache_key] = (model, tokenizer)
        
        return model, tokenizer
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        self._cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    @staticmethod
    def list_models() -> None:
        """Print available models with descriptions."""
        print("\nAvailable Models:")
        print("-" * 60)
        for key, info in MODEL_REGISTRY.items():
            print(f"  {key:12} - {info['display_name']}")
            print(f"               {info['description']}")
        print("-" * 60)


# Quick test
if __name__ == "__main__":
    print("Testing Model Factory...")
    
    # List available models
    ModelFactory.list_models()
    
    # Test loading (will download if not cached)
    print("\nLoading MuRIL model...")
    model, tokenizer = load_model_and_tokenizer("muril")
    
    # Test tokenization
    test_text = "హైదరాబాద్ తెలంగాణ రాజధాని"
    tokens = tokenizer(test_text)
    print(f"\nTest tokenization:")
    print(f"  Input: {test_text}")
    print(f"  Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'])}")
