"""
Basketball Predictions API Endpoints

This module provides REST API endpoints for basketball predictions,
following the same pattern as the football predictions API.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from fastapi import APIRouter, Query, HTTPException, Request
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

from basketball.prediction_service import BasketballPredictionService
from utils.error_handling import handle_database_error, BetSightlyError, ValidationError
from utils.security import check_rate_limit, sanitize_input, log_security_event

router = APIRouter()

# Enums for better type safety
class BasketballResponseFormat(str, Enum):
    SIMPLE = "simple"
    DETAILED = "detailed"
    SUMMARY = "summary"

class ConfidenceLevel(str, Enum):
    HIGH = "high_confidence"
    MEDIUM = "medium_confidence"
    LOW = "low_confidence"
    ALL = "all"

def _get_confidence_metadata(level: str) -> Dict[str, Any]:
    """Get metadata for a confidence level."""
    metadata = {
        "high_confidence": {
            "name": "High Confidence",
            "description": "Predictions with >75% confidence",
            "min_confidence": 0.75,
            "risk_level": "low"
        },
        "medium_confidence": {
            "name": "Medium Confidence", 
            "description": "Predictions with 60-75% confidence",
            "min_confidence": 0.60,
            "risk_level": "medium"
        },
        "low_confidence": {
            "name": "Low Confidence",
            "description": "Predictions with <60% confidence",
            "min_confidence": 0.0,
            "risk_level": "high"
        }
    }
    return metadata.get(level, {})

def _standardize_basketball_response(
    predictions_data: Dict[str, Any],
    confidence_level: Optional[str] = None,
    format_type: BasketballResponseFormat = BasketballResponseFormat.SIMPLE
) -> Dict[str, Any]:
    """
    Standardize basketball prediction response format.
    
    Args:
        predictions_data: Raw predictions data
        confidence_level: Confidence level filter
        format_type: Response format type
        
    Returns:
        Standardized response dictionary
    """
    if predictions_data.get('status') != 'success':
        return predictions_data
    
    predictions = predictions_data.get('predictions', [])
    
    # Filter by confidence level if specified
    if confidence_level and confidence_level != 'all':
        categories = predictions_data.get('categories', {})
        predictions = categories.get(confidence_level, [])
    
    response = {
        "status": "success",
        "date": predictions_data.get('date'),
        "sport": "basketball",
        "league": "NBA",
        "count": len(predictions),
        "predictions": predictions
    }
    
    if confidence_level:
        response["metadata"] = _get_confidence_metadata(confidence_level)
    
    if format_type == BasketballResponseFormat.DETAILED:
        response["statistics"] = {
            "avg_confidence": sum(p.get('overall_confidence', 0) for p in predictions) / len(predictions) if predictions else 0,
            "high_confidence_count": len([p for p in predictions if p.get('overall_confidence', 0) > 0.75]),
            "medium_confidence_count": len([p for p in predictions if 0.60 < p.get('overall_confidence', 0) <= 0.75]),
            "low_confidence_count": len([p for p in predictions if p.get('overall_confidence', 0) <= 0.60])
        }
        
        response["models_status"] = predictions_data.get('models_used', {})
    
    elif format_type == BasketballResponseFormat.SUMMARY:
        response = {
            "status": "success",
            "date": predictions_data.get('date'),
            "sport": "basketball",
            "league": "NBA",
            "summary": {
                "total_games": predictions_data.get('total_games', 0),
                "confidence_breakdown": {
                    category: len(games) 
                    for category, games in predictions_data.get('categories', {}).items()
                },
                "models_available": predictions_data.get('models_used', {}),
                "best_predictions": sorted(
                    predictions, 
                    key=lambda x: x.get('overall_confidence', 0), 
                    reverse=True
                )[:3]
            }
        }
    
    return response

@router.get("/")
def get_basketball_predictions(
    request: Request,
    date: Optional[date] = Query(None, description="Date to get predictions for (YYYY-MM-DD)"),
    confidence: Optional[ConfidenceLevel] = Query(None, description="Filter by confidence level"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of predictions"),
    format: BasketballResponseFormat = Query(BasketballResponseFormat.SIMPLE, description="Response format")
):
    """
    **Basketball Predictions Endpoint**
    
    Get NBA basketball predictions with flexible filtering and formatting options.
    
    **Examples:**
    - `/api/basketball-predictions/` - All predictions for today
    - `/api/basketball-predictions/?confidence=high_confidence` - High confidence predictions only
    - `/api/basketball-predictions/?format=detailed` - Detailed response with statistics
    - `/api/basketball-predictions/?date=2024-01-15` - Predictions for specific date
    """
    try:
        # Apply rate limiting
        check_rate_limit(request)
        
        # Input validation and sanitization
        if date and date > datetime.now().date() + timedelta(days=7):
            raise ValidationError("Date cannot be more than 7 days in the future")
        
        if limit > 50:
            raise ValidationError("Limit cannot exceed 50 for basketball predictions")
        
        # Log request for monitoring
        logger.info(f"Basketball predictions request: date={date}, confidence={confidence}, limit={limit}")
        
        # Initialize prediction service
        prediction_service = BasketballPredictionService()
        
        # Generate predictions
        date_str = date.strftime("%Y-%m-%d") if date else None
        predictions_data = prediction_service.generate_predictions(date_str)
        
        if predictions_data.get('status') != 'success':
            return predictions_data
        
        # Apply limit to predictions
        if 'predictions' in predictions_data:
            predictions_data['predictions'] = predictions_data['predictions'][:limit]
        
        # Standardize response
        response = _standardize_basketball_response(
            predictions_data,
            confidence.value if confidence else None,
            format
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Basketball predictions validation error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Error getting basketball predictions: {str(e)}")
        raise handle_database_error(e, "getting basketball predictions")

@router.get("/summary")
def get_basketball_summary(
    request: Request,
    date: Optional[date] = Query(None, description="Date to get summary for (YYYY-MM-DD)")
):
    """
    **Basketball Predictions Summary**
    
    Get a summary of basketball predictions including confidence breakdown
    and model availability status.
    """
    try:
        # Apply rate limiting
        check_rate_limit(request)
        
        # Input validation
        if date and date > datetime.now().date() + timedelta(days=7):
            raise ValidationError("Date cannot be more than 7 days in the future")
        
        logger.info(f"Basketball summary request: date={date}")
        
        # Initialize prediction service
        prediction_service = BasketballPredictionService()
        
        # Get prediction summary
        date_str = date.strftime("%Y-%m-%d") if date else None
        summary_data = prediction_service.get_prediction_summary(date_str)
        
        return {
            "status": "success",
            "date": summary_data.get('date'),
            "sport": "basketball",
            "league": "NBA",
            "summary": summary_data
        }
        
    except ValidationError as e:
        logger.warning(f"Basketball summary validation error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Error getting basketball summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/confidence/{confidence_level}")
def get_basketball_predictions_by_confidence(
    confidence_level: ConfidenceLevel,
    request: Request,
    date: Optional[date] = Query(None, description="Date to get predictions for (YYYY-MM-DD)"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of predictions")
):
    """
    **Basketball Predictions by Confidence Level**
    
    Get basketball predictions filtered by confidence level.
    
    **Confidence Levels:**
    - `high_confidence`: >75% confidence
    - `medium_confidence`: 60-75% confidence  
    - `low_confidence`: <60% confidence
    - `all`: All predictions
    """
    try:
        # Apply rate limiting
        check_rate_limit(request)
        
        # Input validation
        if date and date > datetime.now().date() + timedelta(days=7):
            raise ValidationError("Date cannot be more than 7 days in the future")
        
        logger.info(f"Basketball confidence predictions request: level={confidence_level}, date={date}")
        
        # Initialize prediction service
        prediction_service = BasketballPredictionService()
        
        # Generate predictions
        date_str = date.strftime("%Y-%m-%d") if date else None
        predictions_data = prediction_service.generate_predictions(date_str)
        
        if predictions_data.get('status') != 'success':
            return predictions_data
        
        # Filter by confidence level
        if confidence_level != ConfidenceLevel.ALL:
            categories = predictions_data.get('categories', {})
            filtered_predictions = categories.get(confidence_level.value, [])
        else:
            filtered_predictions = predictions_data.get('predictions', [])
        
        # Apply limit
        filtered_predictions = filtered_predictions[:limit]
        
        response = {
            "status": "success",
            "date": predictions_data.get('date'),
            "sport": "basketball",
            "league": "NBA",
            "confidence_level": confidence_level.value,
            "count": len(filtered_predictions),
            "predictions": filtered_predictions,
            "metadata": _get_confidence_metadata(confidence_level.value)
        }
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Basketball confidence predictions validation error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Error getting basketball confidence predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/models/status")
def get_basketball_models_status(request: Request):
    """
    **Basketball Models Status**
    
    Get the status of available basketball prediction models.
    """
    try:
        # Apply rate limiting
        check_rate_limit(request)
        
        logger.info("Basketball models status request")
        
        # Initialize prediction service
        prediction_service = BasketballPredictionService()
        
        # Get models status
        models_status = prediction_service._get_models_status()
        
        return {
            "status": "success",
            "sport": "basketball",
            "league": "NBA",
            "models": models_status,
            "total_models": len(models_status),
            "available_models": len([m for m in models_status.values() if m]),
            "model_details": {
                "win_loss_xgboost": {
                    "description": "XGBoost model for Win/Loss prediction",
                    "available": models_status.get('win_loss_xgboost', False)
                },
                "over_under_lightgbm": {
                    "description": "LightGBM model for Over/Under prediction", 
                    "available": models_status.get('over_under_lightgbm', False)
                },
                "neural_network": {
                    "description": "Neural Network for advanced predictions",
                    "available": models_status.get('neural_network', False)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting basketball models status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/train")
def train_basketball_models(request: Request):
    """
    **Train Basketball Models**
    
    Trigger training of basketball prediction models.
    Note: This is a long-running operation.
    """
    try:
        # Apply rate limiting (stricter for training)
        check_rate_limit(request, limit=5, window=3600)  # 5 requests per hour
        
        logger.info("Basketball model training request")
        
        # Initialize prediction service
        prediction_service = BasketballPredictionService()
        
        # Train models
        training_results = prediction_service.train_models()
        
        return {
            "status": "success",
            "sport": "basketball",
            "league": "NBA",
            "training_results": training_results
        }
        
    except Exception as e:
        logger.error(f"Error training basketball models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
