from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class TokenWeight(BaseModel):
    term: str = Field(..., description="Token/word")
    weight: float = Field(..., description="Influence weight")


class SentimentResponse(BaseModel):
    label: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment classification")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    tokens: List[TokenWeight] = Field(..., description="Most influential tokens")


class ReviewItem(BaseModel):
    id: str = Field(..., description="Review ID")
    text: str = Field(..., description="Review text")
    label: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment classification")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    created_at: str = Field(..., description="Review creation timestamp")


class ReviewSummary(BaseModel):
    pos: int = Field(..., description="Number of positive reviews")
    neu: int = Field(..., description="Number of neutral reviews")
    neg: int = Field(..., description="Number of negative reviews")


class ReviewAnalysisResponse(BaseModel):
    items: List[ReviewItem] = Field(..., description="Analyzed reviews")
    summary: ReviewSummary = Field(..., description="Summary statistics")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    amazon_enabled: bool = Field(..., description="Whether Amazon integration is enabled")
