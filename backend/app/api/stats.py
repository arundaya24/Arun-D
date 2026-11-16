from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models import Prediction
from backend.app.schemas.prediction import ClassCount, DailyCount, Stats

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats", response_model=Stats)
def stats(db: Session = Depends(get_db)):
    total, avg_conf, avg_ms = db.execute(
        select(
            func.count(Prediction.id),
            func.coalesce(func.avg(Prediction.confidence), 0.0),
            func.coalesce(func.avg(Prediction.inference_ms), 0.0),
        )
    ).one()

    by_class = db.execute(
        select(Prediction.predicted, func.count(Prediction.id))
        .group_by(Prediction.predicted)
        .order_by(func.count(Prediction.id).desc())
    ).all()

    since = datetime.now() - timedelta(days=14)
    day = func.date(Prediction.created_at)
    daily = db.execute(
        select(day, func.count(Prediction.id))
        .where(Prediction.created_at >= since)
        .group_by(day)
        .order_by(day)
    ).all()

    return Stats(
        total_predictions=total,
        avg_confidence=round(float(avg_conf), 4),
        avg_inference_ms=round(float(avg_ms), 2),
        by_class=[ClassCount(label=l, count=c) for l, c in by_class],
        daily=[DailyCount(date=str(d), count=c) for d, c in daily],
    )
