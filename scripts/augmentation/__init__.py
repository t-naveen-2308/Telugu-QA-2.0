"""Data augmentation module for Telugu QA pairs."""
from .augment_qa import (
    paraphrase_question,
    augment_qa_pair,
    augment_squad_file,
    QUESTION_PARAPHRASES,
    QUESTION_TEMPLATES
)

__all__ = [
    "paraphrase_question",
    "augment_qa_pair",
    "augment_squad_file",
    "QUESTION_PARAPHRASES",
    "QUESTION_TEMPLATES"
]
