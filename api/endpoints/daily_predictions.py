"""
Daily Predictions API - Database-based predictions
Frontend fetches pre-generated predictions from database.
"""

import logging
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session

from database import get_db
from services.daily_predictions_service import DailyPredictionsService, DailyPrediction, DailyPredictionSummary

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize service
daily_service = DailyPredictionsService()

@router.get("/today")
def get_todays_predictions_from_db(db: Session = Depends(get_db)):
    """
    Get today's predictions from database (fast response).
    
    Returns:
        Pre-generated predictions for today
    """
    try:
        today = datetime.now().date()
        
        # Get summary
        summary = db.query(DailyPredictionSummary).filter(
            DailyPredictionSummary.prediction_date == today
        ).first()
        
        if not summary:
            return {
                "status": "no_predictions",
                "date": today.isoformat(),
                "message": "No predictions generated for today. Use /generate to create them.",
                "predictions": []
            }
        
        if summary.generation_status != "completed":
            return {
                "status": "pending",
                "date": today.isoformat(),
                "message": f"Predictions are {summary.generation_status}",
                "summary": daily_service._summary_to_dict(summary),
                "predictions": []
            }
        
        # Get predictions
        predictions = db.query(DailyPrediction).filter(
            DailyPrediction.prediction_date == today
        ).all()
        
        # Format predictions
        formatted_predictions = []
        for pred in predictions:
            formatted_pred = {
                "fixture_info": {
                    "fixture_id": pred.fixture_id,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "league": pred.league_name,
                    "date": pred.fixture_date.isoformat(),
                    "status": pred.fixture_status
                },
                "ml_predictions": json.loads(pred.ml_predictions),
                "betting_categories": {
                    "2_odds": json.loads(pred.betting_2_odds) if pred.betting_2_odds else {},
                    "5_odds": json.loads(pred.betting_5_odds) if pred.betting_5_odds else {},
                    "10_odds": json.loads(pred.betting_10_odds) if pred.betting_10_odds else {},
                    "rollover": json.loads(pred.betting_rollover) if pred.betting_rollover else {}
                },
                "model_summary": {
                    "total_predictions": pred.total_models_used,
                    "highest_confidence": pred.highest_confidence
                }
            }
            formatted_predictions.append(formatted_pred)
        
        return {
            "status": "success",
            "date": today.isoformat(),
            "source": "database",
            "cached": True,
            "summary": daily_service._summary_to_dict(summary),
            "predictions": formatted_predictions
        }
        
    except Exception as e:
        logger.error(f"Error getting today's predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/categories")
def get_betting_categories_from_db(db: Session = Depends(get_db)):
    """
    Get today's betting categories from database.
    
    Returns:
        Categorized betting recommendations
    """
    try:
        today = datetime.now().date()
        
        # Get predictions
        predictions = db.query(DailyPrediction).filter(
            DailyPrediction.prediction_date == today
        ).all()
        
        if not predictions:
            return {
                "status": "no_predictions",
                "date": today.isoformat(),
                "message": "No predictions found for today",
                "categories": {"2_odds": [], "5_odds": [], "10_odds": [], "rollover": []}
            }
        
        # Extract betting categories
        categorized_results = {
            '2_odds': [],
            '5_odds': [],
            '10_odds': [],
            'rollover': []
        }
        
        for pred in predictions:
            fixture_info = {
                "fixture_id": pred.fixture_id,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "league": pred.league_name,
                "date": pred.fixture_date.isoformat(),
                "status": pred.fixture_status
            }
            
            # Process each betting category
            betting_data = {
                "2_odds": json.loads(pred.betting_2_odds) if pred.betting_2_odds else {},
                "5_odds": json.loads(pred.betting_5_odds) if pred.betting_5_odds else {},
                "10_odds": json.loads(pred.betting_10_odds) if pred.betting_10_odds else {},
                "rollover": json.loads(pred.betting_rollover) if pred.betting_rollover else {}
            }
            
            for category, category_data in betting_data.items():
                if category_data.get('selected', False):
                    categorized_results[category].append({
                        'fixture': fixture_info,
                        'prediction': category_data.get('prediction', {}),
                        'confidence': category_data.get('prediction', {}).get('confidence', 0),
                        'risk_level': category_data.get('risk_level', 'UNKNOWN'),
                        'expected_odds': category_data.get('expected_odds', 0),
                        'recommendation': category_data.get('recommendation', 'UNKNOWN')
                    })
        
        return {
            "status": "success",
            "date": today.isoformat(),
            "source": "database",
            "categories": categorized_results,
            "summary": {
                "2_odds_count": len(categorized_results['2_odds']),
                "5_odds_count": len(categorized_results['5_odds']),
                "10_odds_count": len(categorized_results['10_odds']),
                "rollover_count": len(categorized_results['rollover']),
                "total_selections": sum(len(cat) for cat in categorized_results.values())
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting betting categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/generate")
def generate_daily_predictions(
    background_tasks: BackgroundTasks,
    target_date: str = Query(None, description="Date in YYYY-MM-DD format (default: today)"),
    force: bool = Query(False, description="Force regeneration even if predictions exist")
):
    """
    Generate predictions for a specific date (admin/manual trigger).
    
    Args:
        target_date: Date to generate predictions for
        force: Force regeneration even if predictions exist
        
    Returns:
        Generation status
    """
    try:
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Validate date format
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        logger.info(f"🎯 Manual prediction generation requested for {target_date}")
        
        # Check if predictions already exist (unless force)
        if not force:
            db = next(get_db())
            existing = db.query(DailyPredictionSummary).filter(
                DailyPredictionSummary.prediction_date == datetime.strptime(target_date, "%Y-%m-%d").date()
            ).first()
            
            if existing and existing.generation_status == "completed":
                return {
                    "status": "already_exists",
                    "date": target_date,
                    "message": "Predictions already exist. Use force=true to regenerate.",
                    "summary": daily_service._summary_to_dict(existing)
                }
        
        # Generate predictions (can be slow, so run in background for production)
        result = daily_service.generate_daily_predictions(target_date)
        
        return {
            "status": "completed",
            "date": target_date,
            "message": "Predictions generated successfully",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/status")
def get_prediction_status(
    target_date: str = Query(None, description="Date in YYYY-MM-DD format (default: today)"),
    db: Session = Depends(get_db)
):
    """
    Get prediction generation status for a date.
    
    Args:
        target_date: Date to check status for
        
    Returns:
        Status information
    """
    try:
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        prediction_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        # Get summary
        summary = db.query(DailyPredictionSummary).filter(
            DailyPredictionSummary.prediction_date == prediction_date
        ).first()
        
        if not summary:
            return {
                "status": "not_generated",
                "date": target_date,
                "message": "No predictions generated for this date"
            }
        
        return {
            "status": summary.generation_status,
            "date": target_date,
            "summary": daily_service._summary_to_dict(summary)
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error getting prediction status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/history")
def get_prediction_history(
    days: int = Query(7, description="Number of days to retrieve"),
    db: Session = Depends(get_db)
):
    """
    Get prediction history for the last N days.
    
    Args:
        days: Number of days to retrieve
        
    Returns:
        Historical prediction summaries
    """
    try:
        # Get summaries for last N days
        summaries = db.query(DailyPredictionSummary).order_by(
            DailyPredictionSummary.prediction_date.desc()
        ).limit(days).all()
        
        history = []
        for summary in summaries:
            history.append(daily_service._summary_to_dict(summary))
        
        return {
            "status": "success",
            "days_requested": days,
            "days_found": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.delete("/clear")
def clear_predictions(
    target_date: str = Query(None, description="Date in YYYY-MM-DD format (default: today)"),
    db: Session = Depends(get_db)
):
    """
    Clear predictions for a specific date (admin function).
    
    Args:
        target_date: Date to clear predictions for
        
    Returns:
        Deletion status
    """
    try:
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        prediction_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        # Delete predictions
        deleted_predictions = db.query(DailyPrediction).filter(
            DailyPrediction.prediction_date == prediction_date
        ).delete()
        
        # Delete summary
        deleted_summary = db.query(DailyPredictionSummary).filter(
            DailyPredictionSummary.prediction_date == prediction_date
        ).delete()
        
        db.commit()
        
        return {
            "status": "success",
            "date": target_date,
            "message": f"Cleared {deleted_predictions} predictions and {deleted_summary} summary"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error clearing predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
