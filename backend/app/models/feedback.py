from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.app.core.database import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_label: Mapped[str] = mapped_column(String(100), nullable=False)
    correct_label: Mapped[str] = mapped_column(String(100), nullable=False)
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_feedback_created_at", "created_at"),
        Index("ix_feedback_correct_label", "correct_label"),
    )

    def __repr__(self) -> str:
        return (
            f"<Feedback(id={self.id!r}, "
            f"prediction_id={self.prediction_id!r}, "
            f"original_label={self.original_label!r}, "
            f"correct_label={self.correct_label!r}, "
            f"is_correct={self.is_correct!r})>"
        )
