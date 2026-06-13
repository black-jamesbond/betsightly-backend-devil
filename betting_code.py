"""
Betting Code Model

This module defines the BettingCode model for storing betting/booking codes.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from database import Base

class BettingCode(Base):
    """
    BettingCode model for storing betting/booking codes.

    Attributes:
        id: Unique identifier
        code: Betting/booking code
        punter_id: ID of the punter who provided the code
        bookmaker_id: ID of the bookmaker
        odds: Odds value
        event_date: Date of the event
        status: Status of the code (pending, won, lost)
        confidence: Confidence level (1-10)
        featured: Whether the code is featured
        notes: Additional notes
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "betting_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(100), nullable=False)
    punter_id = Column(Integer, ForeignKey("punters.id"), nullable=False)
    bookmaker_id = Column(Integer, ForeignKey("bookmakers.id"), nullable=True)
    odds = Column(Float, nullable=True)
    event_date = Column(DateTime, nullable=True)
    status = Column(String(20), default="pending")  # pending, won, lost
    confidence = Column(Integer, nullable=True)  # 1-10 scale
    featured = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    punter = relationship("Punter", lazy="select")
    bookmaker = relationship("Bookmaker", lazy="select")

    def __init__(
        self,
        code: str,
        punter_id: int,
        bookmaker_id: Optional[int] = None,
        odds: Optional[float] = None,
        event_date: Optional[datetime] = None,
        status: str = "pending",
        confidence: Optional[int] = None,
        featured: bool = False,
        notes: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a betting code.

        Args:
            code: Betting/booking code
            punter_id: ID of the punter who provided the code
            bookmaker_id: ID of the bookmaker
            odds: Odds value
            event_date: Date of the event
            status: Status of the code (pending, won, lost)
            confidence: Confidence level (1-10)
            featured: Whether the code is featured
            notes: Additional notes
            created_at: Creation timestamp
        """
        self.code = code
        self.punter_id = punter_id
        self.bookmaker_id = bookmaker_id
        self.odds = odds
        self.event_date = event_date
        self.status = status
        self.confidence = confidence
        self.featured = featured
        self.notes = notes
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert betting code to dictionary.

        Returns:
            Dictionary representation of betting code
        """
        try:
            punter_name = self.punter.name if hasattr(self, 'punter') and self.punter else None
        except:
            punter_name = None

        try:
            bookmaker_name = self.bookmaker.name if hasattr(self, 'bookmaker') and self.bookmaker else None
        except:
            bookmaker_name = None

        return {
            "id": self.id,
            "code": self.code,
            "punter_id": self.punter_id,
            "punter_name": punter_name,
            "bookmaker_id": self.bookmaker_id,
            "bookmaker_name": bookmaker_name,
            "odds": self.odds,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "status": self.status,
            "confidence": self.confidence,
            "featured": self.featured,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self) -> str:
        """
        Get string representation of betting code.

        Returns:
            String representation
        """
        return f"<BettingCode(id={self.id}, code='{self.code}', punter_id={self.punter_id})>"
