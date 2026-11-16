from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TopK(BaseModel):
    label: str
    score: float


class PredictionResult(BaseModel):
    id: int
    predicted: str
    confidence: float
    top_k: list[TopK]
    inference_ms: float
    model_version: str


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    predicted: str
    confidence: float
    inference_ms: float
    model_version: str
    created_at: datetime


class HistoryPage(BaseModel):
    total: int
    items: list[HistoryItem]


class ClassCount(BaseModel):
    label: str
    count: int


class DailyCount(BaseModel):
    date: str
    count: int


class Stats(BaseModel):
    total_predictions: int
    avg_confidence: float
    avg_inference_ms: float
    by_class: list[ClassCount]
    daily: list[DailyCount]
