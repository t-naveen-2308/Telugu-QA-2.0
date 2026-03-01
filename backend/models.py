"""
Pydantic Models for Telugu QA API

Request/Response models for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class QARequest(BaseModel):
    """Question Answering request."""
    context: str = Field(..., description="Telugu context text", min_length=1)
    question: str = Field(..., description="Telugu question", min_length=1)
    model_key: str = Field(default="muril", description="Model to use: muril, mbert, xlmr, indicbert")


class QAResponse(BaseModel):
    """Question Answering response."""
    answer: str = Field(..., description="Extracted/refined answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    start: int = Field(..., description="Start position in context")
    end: int = Field(..., description="End position in context")
    
    # Morphology refinement details
    original_answer: str = Field(..., description="Raw model output before refinement")
    original_confidence: float = Field(..., description="Original confidence score")
    refinement_applied: bool = Field(default=False, description="Whether morphology refinement was applied")
    removed_suffixes: list[str] = Field(default=[], description="Suffixes removed during refinement")
    confidence_improvement: float = Field(default=0.0, description="Confidence improvement from refinement")
    model_key: str = Field(..., description="Model used for prediction")
    is_relevant: bool = Field(default=True, description="Whether the question is relevant to the context")


class TransliterateRequest(BaseModel):
    """Transliteration request."""
    text: str = Field(..., description="ITRANS/English text to transliterate")


class TransliterateResponse(BaseModel):
    """Transliteration response."""
    original: str = Field(..., description="Original input text")
    options: list[str] = Field(default=[], description="Possible Telugu transliterations")


class TTSRequest(BaseModel):
    """Text-to-Speech request."""
    text: str = Field(..., description="Telugu text to speak", min_length=1)
    slow: bool = Field(default=False, description="Speak slowly")


class ModelInfo(BaseModel):
    """Model information."""
    key: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Model description")
    exact_match: float = Field(..., description="Exact Match score (%)")
    f1_score: float = Field(..., description="F1 score (%)")
    recommended: bool = Field(default=False, description="Whether this is the recommended model")


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for a model."""
    model_key: str
    exact_match: float
    f1_score: float
    total_samples: int = 0


class ModelTrainingLoss(BaseModel):
    """Training loss data for a single model."""
    steps: list[int]
    training_loss: list[float]
    validation_loss: list[float]
    best_step: int
    best_val_loss: float


class TrainingData(BaseModel):
    """Training data for all models."""
    muril: ModelTrainingLoss
    xlmr: ModelTrainingLoss
    indicbert: ModelTrainingLoss
    mbert: ModelTrainingLoss
    muril_domain: ModelTrainingLoss | None = None
    mbert_domain: ModelTrainingLoss | None = None
