"""
Telugu QA Inference Engine

Loads the trained model and provides question answering functionality.
Includes morphology-aware answer refinement for improved boundary detection.
Supports both base checkpoints and LoRA (PEFT) domain-adapted models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from utils.helpers import check_question_relevance
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Import PEFT for LoRA adapter loading
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Import morphology processor for answer refinement
try:
    from ..morphology.processor import TeluguMorphologyProcessor, RefinementResult
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False
    TeluguMorphologyProcessor = None

import unicodedata


# Mapping from domain model keys to base model keys and adapter paths
DOMAIN_MODEL_CONFIG = {
    "muril-domain": {
        "base_key": "muril",
        "adapter_dir": "muril-domain",
        "hf_tokenizer": "google/muril-base-cased",
    },
    "mbert-domain": {
        "base_key": "mbert",
        "adapter_dir": "mbert-domain",
        "hf_tokenizer": "google/bert-base-multilingual-cased",
    },
}




class TeluguQAEngine:
    """
    Telugu Question Answering Engine.
    
    Loads a trained model and provides inference capabilities.
    Includes optional morphology-aware answer refinement for improved
    answer boundary detection in Telugu's agglutinative morphology.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_key: str = "muril",
        device: Optional[str] = None,
        use_morphology: bool = True,
        morphology_aggressive: bool = False
    ):
        """
        Initialize the QA engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_key: Model key (muril, mbert, muril-domain, mbert-domain, etc.)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            use_morphology: Enable morphology-aware answer refinement
            morphology_aggressive: Use aggressive suffix removal
        """
        self.model_key = model_key
        self.is_domain_model = model_key in DOMAIN_MODEL_CONFIG
        self.use_morphology = use_morphology and MORPHOLOGY_AVAILABLE
        
        # Initialize morphology processor - independent of which model is used
        # Morphology refinement is a post-processing step that works the same for all models
        if self.use_morphology:
            self.morphology_processor = TeluguMorphologyProcessor(
                aggressive_mode=morphology_aggressive
            )
        else:
            self.morphology_processor = None
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model based on type
        if self.is_domain_model:
            self._load_domain_model(model_key, model_path)
        else:
            self._load_base_model(model_key, model_path)
        
        print("[OK] Model loaded successfully!")
    
    def _load_base_model(self, model_key: str, model_path: Optional[Union[str, Path]] = None):
        """Load a base (non-LoRA) QA model."""
        if model_path is None:
            model_path = self._get_default_model_path(model_key)
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        print(f"Loading base model from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForQuestionAnswering.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def _load_domain_model(self, model_key: str, adapter_path: Optional[Union[str, Path]] = None):
        """Load a LoRA domain-adapted model (base checkpoint + PEFT adapter)."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft package required for domain models. Install with: pip install peft"
            )
        
        config = DOMAIN_MODEL_CONFIG[model_key]
        base_key = config["base_key"]
        
        # Find base checkpoint
        base_path = self._get_default_model_path(base_key)
        
        # Find adapter
        if adapter_path is None:
            adapter_path = self._get_adapter_path(config["adapter_dir"])
        adapter_path = Path(adapter_path)
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found at: {adapter_path}")
        
        self.model_path = adapter_path  # For get_model_info()
        
        print(f"Loading domain model: {model_key}")
        print(f"  Base checkpoint: {base_path}")
        print(f"  LoRA adapter: {adapter_path}")
        print(f"  Device: {self.device}")
        
        # Load tokenizer (from HuggingFace or base checkpoint)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(base_path))
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(config["hf_tokenizer"])
        
        # Load base model then apply LoRA adapter
        base_model = AutoModelForQuestionAnswering.from_pretrained(str(base_path))
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model = self.model.merge_and_unload()  # Merge LoRA weights for faster inference
        
        self.model.to(self.device)
        self.model.eval()
        
        self.pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def _get_default_model_path(self, model_key: str) -> Path:
        """Get default model path based on model key."""
        # Try to find project root
        current = Path(__file__).resolve()
        for parent in current.parents:
            checkpoint_dir = parent / "models" / "checkpoints" / model_key
            if checkpoint_dir.exists():
                # Find the best checkpoint (prefer 'final', then highest number)
                if (checkpoint_dir / "final").exists():
                    return checkpoint_dir / "final"
                
                checkpoints = sorted(
                    [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                    key=lambda x: int(x.name.split("-")[1]),
                    reverse=True
                )
                if checkpoints:
                    return checkpoints[0]
                
                # If no subdirectories, use the checkpoint_dir itself
                if (checkpoint_dir / "model.safetensors").exists():
                    return checkpoint_dir
        
        raise FileNotFoundError(f"No checkpoint found for model: {model_key}")
    
    def _get_adapter_path(self, adapter_dir: str) -> Path:
        """Get adapter path from models/adapters/<adapter_dir>/."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            adapter_path = parent / "models" / "adapters" / adapter_dir
            if adapter_path.exists():
                # Check for adapter_model.safetensors or adapter_model.bin
                if (adapter_path / "adapter_model.safetensors").exists() or \
                   (adapter_path / "adapter_model.bin").exists():
                    return adapter_path
        
        raise FileNotFoundError(f"No adapter found at models/adapters/{adapter_dir}/")
    
    def answer(
        self,
        question: str,
        context: str,
        max_answer_length: int = 100,
        top_k: int = 1,
        apply_refinement: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Answer a question given a context.
        
        Args:
            question: The question in Telugu
            context: The context/passage to search for the answer
            max_answer_length: Maximum length of the answer
            top_k: Number of top answers to return
            apply_refinement: Apply morphology-aware answer refinement
            
        Returns:
            Dictionary with answer, score, start, end positions
            Or list of dictionaries if top_k > 1
        """
        # Normalize inputs for consistent Unicode handling
        question = self._normalize_input(question)
        context = self._normalize_input(context)
        
        result = self.pipeline(
            question=question,
            context=context,
            max_answer_len=max_answer_length,
            top_k=top_k
        )
        
        # Apply morphology refinement if enabled
        if apply_refinement and self.use_morphology and self.morphology_processor:
            if isinstance(result, list):
                # Multiple answers
                for r in result:
                    refinement = self.morphology_processor.refine_answer(
                        r['answer'], question, context, r['score']
                    )
                    r['original_score'] = r['score']  # Store original confidence
                    r['answer'] = refinement.refined_answer
                    r['original_answer'] = refinement.original_answer
                    r['score'] = min(1.0, max(0.0, r['score'] + refinement.confidence_adjustment))
                    r['confidence_adjustment'] = refinement.confidence_adjustment
                    r['refinement_applied'] = refinement.refinement_applied
                    r['removed_suffixes'] = refinement.removed_suffixes
            else:
                # Single answer
                refinement = self.morphology_processor.refine_answer(
                    result['answer'], question, context, result['score']
                )
                result['original_score'] = result['score']  # Store original confidence
                result['answer'] = refinement.refined_answer
                result['original_answer'] = refinement.original_answer
                result['score'] = min(1.0, max(0.0, result['score'] + refinement.confidence_adjustment))
                result['confidence_adjustment'] = refinement.confidence_adjustment
                result['refinement_applied'] = refinement.refinement_applied
                result['removed_suffixes'] = refinement.removed_suffixes
        
        return result
    
    def batch_answer(
        self,
        questions: List[str],
        contexts: List[str],
        max_answer_length: int = 100,
        apply_refinement: bool = True
    ) -> List[Dict]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            contexts: List of contexts (same length as questions)
            max_answer_length: Maximum length of each answer
            apply_refinement: Apply morphology-aware answer refinement
            
        Returns:
            List of answer dictionaries
        """
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must have the same length")
        
        inputs = [
            {"question": q, "context": c}
            for q, c in zip(questions, contexts)
        ]
        
        results = self.pipeline(inputs, max_answer_len=max_answer_length)
        
        # Apply morphology refinement if enabled
        if apply_refinement and self.use_morphology and self.morphology_processor:
            for r, q, c in zip(results, questions, contexts):
                refinement = self.morphology_processor.refine_answer(
                    r['answer'], q, c, r['score']
                )
                r['answer'] = refinement.refined_answer
                r['original_answer'] = refinement.original_answer
                r['score'] = min(1.0, max(0.0, r['score'] + refinement.confidence_adjustment))
                r['refinement_applied'] = refinement.refinement_applied
                r['removed_suffixes'] = refinement.removed_suffixes
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            "model_key": self.model_key,
            "model_path": str(self.model_path),
            "device": self.device,
            "model_type": self.model.config.model_type,
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "morphology_enabled": self.use_morphology,
            "is_domain_model": self.is_domain_model,
        }
        if self.is_domain_model:
            config = DOMAIN_MODEL_CONFIG[self.model_key]
            info["base_model"] = config["base_key"]
            info["adapter_dir"] = config["adapter_dir"]
        return info
    
    def analyze_morphology_errors(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: List[str]
    ) -> Dict:
        """
        Analyze morphology-related errors in predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            questions: List of questions
            
        Returns:
            Analysis dictionary with error statistics
        """
        if not self.morphology_processor:
            return {"error": "Morphology processing not enabled"}
        
        return self.morphology_processor.analyze_morphology_errors(
            predictions, ground_truths, questions
        )
    
    def answer_with_rescore(
        self,
        question: str,
        context: str,
        max_answer_length: int = 100
    ) -> Dict:
        """
        Answer question with morphology refinement.
        
        This method:
        1. Normalizes inputs for consistent Unicode handling
        2. Gets the raw answer from the model
        3. Applies morphology refinement (suffix removal, etc.)
        4. Returns refined answer with confidence adjustment
        
        Args:
            question: The question in Telugu
            context: The context/passage
            max_answer_length: Maximum answer length
            
        Returns:
            Dict with answer, original_answer, score, refinement details
        """
        # Normalize inputs
        question = self._normalize_input(question)
        context = self._normalize_input(context)
        
        is_relevant = check_question_relevance(question, context)
        
        if not is_relevant:
            return {
                'answer': '',
                'original_answer': '',
                'score': 0.0,
                'original_score': 0.0,
                'start': 0,
                'end': 0,
                'refinement_applied': False,
                'removed_suffixes': [],
                'confidence_improvement': 0.0,
                'is_relevant': False
            }
        
        # Get raw answer
        raw_result = self.pipeline(
            question=question,
            context=context,
            max_answer_len=max_answer_length
        )
        
        original_answer = raw_result['answer']
        original_score = raw_result['score']
        
        # Apply morphology refinement if enabled
        if self.use_morphology and self.morphology_processor:
            refinement = self.morphology_processor.refine_answer(
                original_answer, question, context, original_score
            )
            refined_answer = refinement.refined_answer
            removed_suffixes = refinement.removed_suffixes
            refinement_applied = refinement.refinement_applied
            confidence_adjustment = refinement.confidence_adjustment
        else:
            refined_answer = original_answer
            removed_suffixes = []
            refinement_applied = False
            confidence_adjustment = 0.0
        
        # Apply confidence adjustment
        final_score = min(1.0, max(0.0, original_score + confidence_adjustment))
        
        return {
            'answer': refined_answer,
            'original_answer': original_answer,
            'score': final_score,
            'original_score': original_score,
            'start': raw_result['start'],
            'end': raw_result['end'],
            'refinement_applied': refinement_applied,
            'removed_suffixes': removed_suffixes,
            'confidence_improvement': final_score - original_score,
            'is_relevant': True
        }
    
    @staticmethod
    def _normalize_input(text: str) -> str:
        """Normalize input text (question/context) for consistent processing."""
        if not text:
            return text
        # Unicode NFC normalization for consistent representation
        return unicodedata.normalize('NFC', text)


def load_qa_engine(
    model_key: str = "muril",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_morphology: bool = True,
    morphology_aggressive: bool = False
) -> TeluguQAEngine:
    """
    Convenience function to load the QA engine.
    
    Args:
        model_key: Which model to load (muril, indicbert, xlmr, mbert)
        model_path: Optional custom path to model
        device: Device to use (auto-detected if None)
        use_morphology: Enable morphology-aware answer refinement
        morphology_aggressive: Use aggressive suffix removal
        
    Returns:
        Initialized TeluguQAEngine
    """
    return TeluguQAEngine(
        model_path=model_path,
        model_key=model_key,
        device=device,
        use_morphology=use_morphology,
        morphology_aggressive=morphology_aggressive
    )


# Quick test when run directly
if __name__ == "__main__":
    print("="*60)
    print("Telugu QA Engine - Test (with Morphology Refinement)")
    print("="*60)
    
    # Load engine with morphology enabled
    engine = load_qa_engine("muril", use_morphology=True)
    
    # Test question
    context = "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని. ఇది దక్కన్ పీఠభూమిపై ఉంది. హైదరాబాద్ జనాభా దాదాపు 1 కోటి."
    question = "తెలంగాణ రాజధాని ఏది?"
    
    print(f"\nContext: {context}")
    print(f"Question: {question}")
    
    # Test with refinement
    result = engine.answer(question, context)
    
    print(f"\nRefined Answer: {result['answer']}")
    if 'original_answer' in result and result.get('refinement_applied'):
        print(f"Original Answer: {result['original_answer']}")
        print(f"Removed Suffixes: {result.get('removed_suffixes', [])}")
    print(f"Score: {result['score']:.4f}")
    
    # Test without refinement for comparison
    result_raw = engine.answer(question, context, apply_refinement=False)
    print(f"\nRaw Answer (no refinement): {result_raw['answer']}")
    print(f"Raw Score: {result_raw['score']:.4f}")
    
    print("\n" + "="*60)
    print(f"Morphology enabled: {engine.use_morphology}")
