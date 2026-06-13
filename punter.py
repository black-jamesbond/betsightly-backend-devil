"""
Punter Model

This module defines the Punter model for storing information about prediction providers.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, Integer, Boolean, Float, Text
from sqlalchemy.orm import relationship

from database import Base

class Punter(Base):
    """
    Punter model for storing information about prediction providers.
    Used by both the main API and the Telegram bot.
    """

    __tablename__ = "punters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    nickname = Column(String(100), nullable=True)
    telegram_username = Column(String(100), nullable=True, unique=True)
    country = Column(String(100), default="Nigeria")
    specialty = Column(String(100), nullable=True)
    verified = Column(Boolean, default=False)

    # Extended profile fields (used by frontend + service)
    image_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    social_media = Column(Text, nullable=True)  # JSON string: {"twitter": "url", "instagram": "url"}

    # Computed stats (updated when predictions are resolved)
    success_rate = Column(Float, default=0.0)    # win percentage 0-100
    popularity = Column(Integer, default=0)       # total predictions count
    total_won = Column(Integer, default=0)
    total_lost = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    betting_codes = relationship("BettingCode", lazy="select", overlaps="punter")
    punter_predictions = relationship("PunterPrediction", back_populates="punter", lazy="select")

    def update_stats(self):
        """Recalculate success_rate and popularity from predictions."""
        won = self.total_won
        lost = self.total_lost
        total = won + lost
        self.popularity = total
        self.success_rate = round((won / total) * 100, 1) if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert punter to dictionary."""
        import json as _json
        social = {}
        if self.social_media:
            try:
                social = _json.loads(self.social_media)
            except (ValueError, TypeError):
                social = {}

        return {
            "id": self.id,
            "name": self.name,
            "nickname": self.nickname,
            "telegram_username": self.telegram_username,
            "country": self.country,
            "specialty": self.specialty,
            "verified": self.verified,
            "image_url": self.image_url,
            "bio": self.bio,
            "social_media": social,
            "success_rate": self.success_rate or 0.0,
            "popularity": self.popularity or 0,
            "total_won": self.total_won or 0,
            "total_lost": self.total_lost or 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        """
        Get string representation of punter.

        Returns:
            String representation
        """
        return f"<Punter(id={self.id}, name='{self.name}', specialty='{self.specialty}')>"
