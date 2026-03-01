"""Data loading and preprocessing for Telugu QA System."""

from .tequad_loader import load_tequad_dataset, TeQuADDataset
from .preprocessing import preprocess_for_training, preprocess_validation
