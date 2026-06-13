"""
Bookmaker Model

This module defines the Bookmaker model for storing information about betting companies.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, DateTime, Integer

from database import Base

class Bookmaker(Base):
    """
    Bookmaker model for storing information about betting companies.

    Attributes:
        id: Unique identifier
        name: Bookmaker name
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "bookmakers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships will be set up after all models are defined

    def __init__(
        self,
        name: str,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a bookmaker.

        Args:
            name: Bookmaker name
            created_at: Creation timestamp
        """
        self.name = name
        self.created_at = created_at or datetime.now()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert bookmaker to dictionary.

        Returns:
            Dictionary representation of bookmaker
        """
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self) -> str:
        """
        Get string representation of bookmaker.

        Returns:
            String representation
        """
        return f"<Bookmaker(id={self.id}, name='{self.name}')>"
