"""QA Generation module for Telugu domain-specific QA pairs."""
from .generate_qa_pairs import (
    QAPair,
    generate_template_qa,
    generate_entity_based_qa,
    generate_synthetic_qa_pairs,
    convert_to_squad
)

__all__ = [
    "QAPair",
    "generate_template_qa", 
    "generate_entity_based_qa",
    "generate_synthetic_qa_pairs",
    "convert_to_squad"
]
