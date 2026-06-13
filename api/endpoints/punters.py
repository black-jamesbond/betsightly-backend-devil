"""
Punters API Endpoints

This module defines the API endpoints for punters and their predictions.
"""

import logging
from typing import List, Optional
from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from database import get_db
from services.punter_service import punter_service
from punter import Punter
from punter_prediction import PunterPrediction
from schemas.punter import PunterCreate, PunterUpdate, PunterResponse, PunterListResponse
from schemas.prediction import PunterPredictionResponse, PunterPredictionListResponse
from utils.common import setup_logging

# Set up logging
logger = setup_logging(__name__)

# Create router
router = APIRouter()

@router.get("/", response_model=PunterListResponse)
def get_punters(
    db: Session = Depends(get_db),
    skip: int = Query(0, description="Number of punters to skip"),
    limit: int = Query(100, description="Maximum number of punters to return")
):
    """
    Get all punters.

    Args:
        skip: Number of punters to skip
        limit: Maximum number of punters to return
    """
    try:
        punters = punter_service.get_all_punters()

        return {
            "status": "success",
            "punters": punters[skip:skip+limit],
            "total": len(punters),
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting punters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting punters: {str(e)}")

@router.get("/top", response_model=PunterListResponse)
def get_top_punters(
    db: Session = Depends(get_db),
    limit: int = Query(10, description="Maximum number of punters to return")
):
    """
    Get top punters based on win rate.

    Args:
        limit: Maximum number of punters to return
    """
    try:
        punters = punter_service.get_top_punters(limit)

        return {
            "status": "success",
            "punters": punters,
            "total": len(punters),
            "skip": 0,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting top punters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting top punters: {str(e)}")

@router.get("/{punter_id}", response_model=PunterResponse)
def get_punter(
    punter_id: str,
    db: Session = Depends(get_db)
):
    """
    Get punter by ID.

    Args:
        punter_id: Punter ID
    """
    try:
        punter = punter_service.get_punter_by_id(punter_id)

        if not punter:
            raise HTTPException(status_code=404, detail=f"Punter with ID {punter_id} not found")

        return {
            "status": "success",
            "punter": punter
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting punter {punter_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting punter: {str(e)}")

@router.post("/", response_model=PunterResponse)
def create_punter(
    punter: PunterCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new punter.

    Args:
        punter: Punter data
    """
    try:
        created_punter = punter_service.create_punter(punter.dict())

        if not created_punter:
            raise HTTPException(status_code=500, detail="Failed to create punter")

        return {
            "status": "success",
            "punter": created_punter
        }
    except Exception as e:
        logger.error(f"Error creating punter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating punter: {str(e)}")

@router.put("/{punter_id}", response_model=PunterResponse)
def update_punter(
    punter_id: str,
    punter: PunterUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an existing punter.

    Args:
        punter_id: Punter ID
        punter: Updated punter data
    """
    try:
        updated_punter = punter_service.update_punter(punter_id, punter.dict(exclude_unset=True))

        if not updated_punter:
            raise HTTPException(status_code=404, detail=f"Punter with ID {punter_id} not found")

        return {
            "status": "success",
            "punter": updated_punter
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating punter {punter_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating punter: {str(e)}")

@router.delete("/{punter_id}")
def delete_punter(
    punter_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a punter.

    Args:
        punter_id: Punter ID
    """
    try:
        success = punter_service.delete_punter(punter_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Punter with ID {punter_id} not found")

        return {
            "status": "success",
            "message": f"Punter with ID {punter_id} deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting punter {punter_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting punter: {str(e)}")

@router.get("/{punter_id}/predictions", response_model=PunterPredictionListResponse)
def get_punter_predictions(
    punter_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(10, description="Maximum number of predictions to return")
):
    """
    Get predictions by a punter.

    Args:
        punter_id: Punter ID
        limit: Maximum number of predictions to return
    """
    try:
        # Check if punter exists
        punter = punter_service.get_punter_by_id(punter_id)

        if not punter:
            raise HTTPException(status_code=404, detail=f"Punter with ID {punter_id} not found")

        # Get predictions
        predictions = punter_service.get_punter_predictions(punter_id, limit)

        return {
            "status": "success",
            "predictions": predictions,
            "total": len(predictions),
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for punter {punter_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {str(e)}")

@router.get("/{punter_id}/performance")
def get_punter_performance(
    punter_id: str,
    db: Session = Depends(get_db)
):
    """
    Get performance metrics for a punter.

    Args:
        punter_id: Punter ID
    """
    try:
        # Check if punter exists
        punter = punter_service.get_punter_by_id(punter_id)

        if not punter:
            raise HTTPException(status_code=404, detail=f"Punter with ID {punter_id} not found")

        # Get performance metrics
        performance = punter_service.get_punter_performance(punter_id)

        return {
            "status": "success",
            "performance": performance
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance for punter {punter_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance: {str(e)}")



@router.post("/{punter_id}/predictions/{prediction_id}/status")
def update_prediction_status(
    punter_id: str,
    prediction_id: str,
    status: str = Query(..., description="New status (won, lost, pending, void)"),
    db: Session = Depends(get_db)
):
    """
    Update prediction status.

    Args:
        punter_id: Punter ID
        prediction_id: Prediction ID
        status: New status (won, lost, pending, void)
    """
    try:
        # Check if punter exists
        punter = punter_service.get_punter_by_id(punter_id)

        if not punter:
            raise HTTPException(status_code=404, detail=f"Punter with ID {punter_id} not found")

        # Validate status
        valid_statuses = ["won", "lost", "pending", "void"]

        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}")

        # Update prediction status
        updated_prediction = punter_service.update_prediction_status(prediction_id, status)

        if not updated_prediction:
            raise HTTPException(status_code=404, detail=f"Prediction with ID {prediction_id} not found")

        return {
            "status": "success",
            "prediction": updated_prediction
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prediction status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating prediction status: {str(e)}")
