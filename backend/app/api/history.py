from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models import Prediction
from backend.app.schemas.prediction import HistoryPage

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history", response_model=HistoryPage)
def history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    total = db.scalar(select(func.count(Prediction.id))) or 0
    rows = db.scalars(
        select(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset)
    ).all()
    return HistoryPage(total=total, items=rows)


@router.delete("/history/{pred_id}", status_code=204)
def delete_item(pred_id: int, db: Session = Depends(get_db)):
    row = db.get(Prediction, pred_id)
    if row is None:
        raise HTTPException(404, "Data tidak ditemukan.")
    db.delete(row)
    db.commit()
