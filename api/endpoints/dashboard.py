"""
Dashboard API endpoints.
"""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from fixture import Fixture
from prediction import Prediction

router = APIRouter()

@router.get("/summary")
def get_dashboard_summary(
    db: Session = Depends(get_db),
    date_str: Optional[str] = None
):
    """Get dashboard summary."""
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        target_date = datetime.now().date()

    # Get date range
    start_date = datetime.combine(target_date, datetime.min.time())
    end_date = start_date + timedelta(days=1)

    # Get fixtures count
    fixtures_count = db.query(func.count(Fixture.id)).filter(
        Fixture.date >= start_date,
        Fixture.date < end_date
    ).scalar()

    # Get predictions count
    predictions_count = db.query(func.count(Prediction.id)).join(Fixture).filter(
        Fixture.date >= start_date,
        Fixture.date < end_date
    ).scalar()

    # Get high confidence predictions count
    high_confidence_count = db.query(func.count(Prediction.id)).filter(
        Prediction.fixture_id.in_(
            db.query(Fixture.fixture_id).filter(
                Fixture.date >= start_date,
                Fixture.date < end_date
            )
        ),
        (
            (Prediction.home_win_pred >= 0.7) |
            (Prediction.draw_pred >= 0.7) |
            (Prediction.away_win_pred >= 0.7) |
            (Prediction.over_2_5_pred >= 0.7) |
            (Prediction.under_2_5_pred >= 0.7) |
            (Prediction.btts_yes_pred >= 0.7) |
            (Prediction.btts_no_pred >= 0.7)
        )
    ).scalar()

    return {
        "date": target_date.isoformat(),
        "fixtures_count": fixtures_count,
        "predictions_count": predictions_count,
        "high_confidence_count": high_confidence_count
    }

@router.get("/high-confidence")
def get_high_confidence_predictions(
    db: Session = Depends(get_db),
    date_str: Optional[str] = None,
    confidence_threshold: float = 0.7
):
    """Get high confidence predictions."""
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        target_date = datetime.now().date()

    # Get date range
    start_date = datetime.combine(target_date, datetime.min.time())
    end_date = start_date + timedelta(days=1)

    # Get fixtures with high confidence predictions
    fixtures = db.query(Fixture).filter(
        Fixture.date >= start_date,
        Fixture.date < end_date,
        Fixture.predictions.any(
            (Prediction.home_win_pred >= confidence_threshold) |
            (Prediction.draw_pred >= confidence_threshold) |
            (Prediction.away_win_pred >= confidence_threshold) |
            (Prediction.over_2_5_pred >= confidence_threshold) |
            (Prediction.under_2_5_pred >= confidence_threshold) |
            (Prediction.btts_yes_pred >= confidence_threshold) |
            (Prediction.btts_no_pred >= confidence_threshold)
        )
    ).all()

    # Build response
    result = []

    for fixture in fixtures:
        prediction = db.query(Prediction).filter(Prediction.fixture_id == fixture.fixture_id).first()

        if not prediction:
            continue

        # Get high confidence predictions
        high_confidence_predictions = []

        if prediction.home_win_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "Match Result",
                "prediction": "HOME",
                "confidence": prediction.home_win_pred
            })

        if prediction.draw_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "Match Result",
                "prediction": "DRAW",
                "confidence": prediction.draw_pred
            })

        if prediction.away_win_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "Match Result",
                "prediction": "AWAY",
                "confidence": prediction.away_win_pred
            })

        if prediction.over_2_5_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "Over/Under",
                "prediction": "OVER",
                "confidence": prediction.over_2_5_pred
            })

        if prediction.under_2_5_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "Over/Under",
                "prediction": "UNDER",
                "confidence": prediction.under_2_5_pred
            })

        if prediction.btts_yes_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "BTTS",
                "prediction": "YES",
                "confidence": prediction.btts_yes_pred
            })

        if prediction.btts_no_pred >= confidence_threshold:
            high_confidence_predictions.append({
                "type": "BTTS",
                "prediction": "NO",
                "confidence": prediction.btts_no_pred
            })

        result.append({
            "fixture": fixture.to_dict(),
            "high_confidence_predictions": high_confidence_predictions
        })

    return result
