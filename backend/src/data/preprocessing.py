"""
Preprocessing for QA Training

Converts raw text data into tokenized format suitable for transformer training.
Handles Telugu-specific considerations and sliding window for long documents.
"""

from typing import Dict, List, Tuple, Optional, Any
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import Dataset
import unicodedata


def normalize_telugu_text(text: str) -> str:
    """
    Normalize Telugu text for consistent processing.
    
    - Applies NFC normalization
    - Handles zero-width characters
    """
    # NFC normalization for consistent Unicode representation
    text = unicodedata.normalize("NFC", text)
    
    # Optionally remove zero-width spaces (can cause tokenization issues)
    # text = text.replace("\u200b", "")  # Zero-width space
    
    return text


def prepare_train_features(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    max_query_length: int = 64,
    pad_to_max_length: bool = True
) -> Dict[str, List]:
    """
    Prepare training features from SQuAD-format examples.
    
    This function:
    1. Tokenizes question and context
    2. Handles long contexts with sliding window
    3. Finds token positions for answer spans
    
    Args:
        examples: Batch of examples with context, question, answers
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        doc_stride: Overlap between chunks for sliding window
        max_query_length: Maximum question length
        pad_to_max_length: Whether to pad to max_length
        
    Returns:
        Tokenized features with start_positions and end_positions
    """
    # Normalize text
    questions = [normalize_telugu_text(q) for q in examples["question"]]
    contexts = [normalize_telugu_text(c) for c in examples["context"]]
    
    # Tokenize with sliding window for long documents
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )
    
    # Map from feature index back to example index
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    
    # Initialize output
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    tokenized["example_id"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # Get the example this feature came from
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]
        
        # Store example ID for evaluation
        tokenized["example_id"].append(examples["id"][sample_idx])
        
        # If no answer, set to CLS token (index 0)
        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
            continue
        
        # Get answer span (use first answer)
        answer_start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end_char = answer_start_char + len(answer_text)
        
        # Find token positions
        sequence_ids = tokenized.sequence_ids(i)
        
        # Find context start and end in tokens
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
        
        # Check if answer is within this chunk
        if (offsets[context_start][0] > answer_end_char or 
            offsets[context_end][1] < answer_start_char):
            # Answer not in this chunk
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            # Find token indices for answer
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= answer_start_char:
                start_token += 1
            start_token -= 1
            
            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= answer_end_char:
                end_token -= 1
            end_token += 1
            
            tokenized["start_positions"].append(start_token)
            tokenized["end_positions"].append(end_token)
    
    return tokenized


def prepare_validation_features(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    max_query_length: int = 64,
    pad_to_max_length: bool = True
) -> Dict[str, List]:
    """
    Prepare validation features (similar to training but keeps offset mapping).
    
    For validation, we need to keep track of offset mappings to convert
    token predictions back to character positions.
    """
    # Normalize text
    questions = [normalize_telugu_text(q) for q in examples["question"]]
    contexts = [normalize_telugu_text(c) for c in examples["context"]]
    
    # Tokenize with sliding window
    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    
    # Keep offset mapping for post-processing
    tokenized["example_id"] = []
    
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_idx])
        
        # Set sequence IDs to None for non-context tokens
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    
    return tokenized


def preprocess_for_training(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    num_proc: int = 4
) -> Dataset:
    """
    Preprocess entire dataset for training.
    
    Args:
        dataset: Raw dataset with context, question, answers
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        doc_stride: Sliding window stride
        num_proc: Number of processes for parallel processing
        
    Returns:
        Tokenized dataset ready for training
    """
    tokenized_dataset = dataset.map(
        lambda examples: prepare_train_features(
            examples, 
            tokenizer, 
            max_length=max_length,
            doc_stride=doc_stride
        ),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing training data"
    )
    
    return tokenized_dataset


def preprocess_validation(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    num_proc: int = 4
) -> Dataset:
    """
    Preprocess dataset for validation/evaluation.
    
    Args:
        dataset: Raw dataset with context, question, answers
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        doc_stride: Sliding window stride
        num_proc: Number of processes for parallel processing
        
    Returns:
        Tokenized dataset with offset mappings for evaluation
    """
    tokenized_dataset = dataset.map(
        lambda examples: prepare_validation_features(
            examples,
            tokenizer,
            max_length=max_length,
            doc_stride=doc_stride
        ),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing validation data"
    )
    
    return tokenized_dataset
