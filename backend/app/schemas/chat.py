from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .prediction import PredictionResult

Role = Literal["system", "user", "assistant"]


class TopKItem(BaseModel):
    label: str
    score: float


class FeedbackRequest(BaseModel):
    prediction_id: Optional[int] = None
    original_label: str
    correct_label: str
    is_correct: bool
    image_path: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    message: str
    correction_saved: bool


class ChatBotResponse(BaseModel):
    prediction_id: Optional[int] = None
    chatbot_message: str
    predicted: Optional[str] = None
    confidence: Optional[float] = None
    top_k: Optional[List[TopKItem]] = None
    engine_used: Optional[str] = None
    is_low_confidence: Optional[bool] = None
    is_human: Optional[bool] = False
    inference_ms: Optional[float] = None
    model_version: Optional[str] = None


class RetrainStatus(BaseModel):
    status: str
    message: str
    trained_on: Optional[int] = None
    new_classes: Optional[List[str]] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[int] = None
    message: str = Field(min_length=1, max_length=2000)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Role
    content: str
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: List[MessageOut]