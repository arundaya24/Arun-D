from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255))
    predicted: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)
    top_k_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    inference_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("idx_created", "created_at"),
        Index("idx_predicted", "predicted"),
    )
