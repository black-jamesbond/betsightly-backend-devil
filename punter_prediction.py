"""
Punter Prediction Model

This module defines the PunterPrediction model for storing predictions from punters.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy import Column, String, DateTime, Float, Text, ForeignKey, Integer
from sqlalchemy.orm import relationship

from database import Base

class PunterPrediction(Base):
    """
    PunterPrediction model for storing predictions from punters.

    Attributes:
        id: Unique identifier
        punter_id: ID of the punter who made the prediction
        home_team: Home team name
        away_team: Away team name
        prediction_type: Type of prediction (e.g., match_result, over_under, btts)
        prediction: Prediction value
        odds: Odds for the prediction
        match_datetime: Date and time of the match
        source: Source of the prediction (e.g., telegram, twitter)
        source_id: ID of the source message/post
        source_text: Original text of the prediction
        confidence: Confidence score (0-1)
        status: Status of the prediction (pending, won, lost, void)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        punter: Relationship to punter
    """

    __tablename__ = "punter_predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    punter_id = Column(Integer, ForeignKey("punters.id"), nullable=False)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    prediction_type = Column(String(50), nullable=False)
    prediction = Column(String(50), nullable=True)
    odds = Column(Float, nullable=True)
    match_datetime = Column(DateTime, nullable=True)
    source = Column(String(50), nullable=True)
    source_id = Column(String(100), nullable=True)
    source_text = Column(Text, nullable=True)
    confidence = Column(Float, default=0.5)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    punter = relationship("Punter", back_populates="punter_predictions")

    def __init__(
        self,
        punter_id: int,
        home_team: str,
        away_team: str,
        prediction_type: str,
        prediction: Optional[str] = None,
        odds: Optional[float] = None,
        match_datetime: Optional[datetime] = None,
        source: Optional[str] = None,
        source_id: Optional[str] = None,
        source_text: Optional[str] = None,
        confidence: float = 0.5,
        status: str = "pending",
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a punter prediction.

        Args:
            punter_id: ID of the punter who made the prediction
            home_team: Home team name
            away_team: Away team name
            prediction_type: Type of prediction (e.g., match_result, over_under, btts)
            prediction: Prediction value
            odds: Odds for the prediction
            match_datetime: Date and time of the match
            source: Source of the prediction (e.g., telegram, twitter)
            source_id: ID of the source message/post
            source_text: Original text of the prediction
            confidence: Confidence score (0-1)
            status: Status of the prediction (pending, won, lost, void)
            created_at: Creation timestamp
        """
        self.punter_id = punter_id
        self.home_team = home_team
        self.away_team = away_team
        self.prediction_type = prediction_type
        self.prediction = prediction
        self.odds = odds
        self.match_datetime = match_datetime
        self.source = source
        self.source_id = source_id
        self.source_text = source_text
        self.confidence = confidence
        self.status = status
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert punter prediction to dictionary.

        Returns:
            Dictionary representation of punter prediction
        """
        return {
            "id": self.id,
            "punter_id": self.punter_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prediction_type": self.prediction_type,
            "prediction": self.prediction,
            "odds": self.odds,
            "match_datetime": self.match_datetime.isoformat() if self.match_datetime else None,
            "source": self.source,
            "source_id": self.source_id,
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self) -> str:
        """
        Get string representation of punter prediction.

        Returns:
            String representation
        """
        return f"<PunterPrediction(id='{self.id}', type='{self.prediction_type}', teams='{self.home_team} vs {self.away_team}')>"
