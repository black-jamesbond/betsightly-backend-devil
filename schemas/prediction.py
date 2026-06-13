"""
Prediction Schemas

This module defines the Pydantic schemas for predictions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class PunterPredictionBase(BaseModel):
    """Base schema for punter prediction data."""
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    prediction_type: str = Field(..., description="Type of prediction (e.g., match_result, over_under, btts)")
    prediction: Optional[str] = Field(None, description="Prediction value")
    odds: Optional[float] = Field(None, description="Odds for the prediction")
    match_datetime: Optional[datetime] = Field(None, description="Date and time of the match")
    source: Optional[str] = Field(None, description="Source of the prediction (e.g., telegram, twitter)")
    source_id: Optional[str] = Field(None, description="ID of the source message/post")
    source_text: Optional[str] = Field(None, description="Original text of the prediction")
    confidence: float = Field(0.5, description="Confidence score (0-1)")
    status: str = Field("pending", description="Status of the prediction (pending, won, lost, void)")

class PunterPredictionCreate(PunterPredictionBase):
    """Schema for creating a punter prediction."""
    punter_id: str = Field(..., description="ID of the punter who made the prediction")

class PunterPredictionUpdate(BaseModel):
    """Schema for updating a punter prediction."""
    prediction: Optional[str] = Field(None, description="Prediction value")
    odds: Optional[float] = Field(None, description="Odds for the prediction")
    match_datetime: Optional[datetime] = Field(None, description="Date and time of the match")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    status: Optional[str] = Field(None, description="Status of the prediction (pending, won, lost, void)")

class PunterPredictionInDB(PunterPredictionBase):
    """Schema for punter prediction data in the database."""
    id: str = Field(..., description="Prediction ID")
    punter_id: str = Field(..., description="ID of the punter who made the prediction")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

class PunterPredictionResponse(BaseModel):
    """Schema for punter prediction response."""
    status: str = Field(..., description="Response status")
    prediction: Dict[str, Any] = Field(..., description="Prediction data")

class PunterPredictionListResponse(BaseModel):
    """Schema for punter prediction list response."""
    status: str = Field(..., description="Response status")
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    limit: int = Field(..., description="Maximum number of predictions returned")
