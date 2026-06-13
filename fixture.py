"""
Fixture model for storing football match fixtures.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from database import Base

# Forward declaration to avoid circular imports
prediction_table = None

class Fixture(Base):
    """Fixture model for storing football match fixtures."""

    __tablename__ = "fixtures"

    id = Column(Integer, primary_key=True)
    fixture_id = Column(Integer, unique=True, index=True)
    date = Column(DateTime)
    league_id = Column(Integer, index=True)
    league_name = Column(String(255))
    home_team_id = Column(Integer, index=True)
    home_team = Column(String(255))
    away_team_id = Column(Integer, index=True)
    away_team = Column(String(255))
    home_odds = Column(Float, default=0)
    draw_odds = Column(Float, default=0)
    away_odds = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships will be set up after both classes are defined
    predictions = []

    def __repr__(self):
        return f"<Fixture {self.home_team} vs {self.away_team} on {self.date}>"

    @classmethod
    def from_api(cls, fixture_data):
        """Create a Fixture instance from API data."""
        return cls(
            fixture_id=fixture_data["fixture_id"],
            date=datetime.fromisoformat(fixture_data["date"].replace("Z", "+00:00")),
            league_id=fixture_data["league_id"],
            league_name=fixture_data["league_name"],
            home_team_id=fixture_data["home_team_id"],
            home_team=fixture_data["home_team"],
            away_team_id=fixture_data["away_team_id"],
            away_team=fixture_data["away_team"],
            home_odds=fixture_data.get("home_odds", 0),
            draw_odds=fixture_data.get("draw_odds", 0),
            away_odds=fixture_data.get("away_odds", 0)
        )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "fixture_id": self.fixture_id,
            "date": self.date.isoformat(),
            "league_id": self.league_id,
            "league_name": self.league_name,
            "home_team_id": self.home_team_id,
            "home_team": self.home_team,
            "away_team_id": self.away_team_id,
            "away_team": self.away_team,
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
