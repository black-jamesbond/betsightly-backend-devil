"""
Prediction model for storing football match predictions.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from database import Base
from fixture import Fixture

class Prediction(Base):
    """Prediction model for storing football match predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.fixture_id"), index=True)
    match_result_pred = Column(String(10))  # HOME, DRAW, AWAY
    home_win_pred = Column(Float)
    draw_pred = Column(Float)
    away_win_pred = Column(Float)
    over_under_pred = Column(String(10))  # OVER, UNDER
    over_2_5_pred = Column(Float)
    under_2_5_pred = Column(Float)
    btts_pred = Column(String(10))  # YES, NO
    btts_yes_pred = Column(Float)
    btts_no_pred = Column(Float)

    # New fields for categorization
    prediction_type = Column(String(20))  # 2_odds, 5_odds, 10_odds, rollover
    odds = Column(Float, default=0)
    confidence = Column(Float, default=0)
    combined_odds = Column(Float, default=0)
    combined_confidence = Column(Float, default=0)
    combo_id = Column(String(50))
    rollover_day = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)
    # updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # Removed due to migration issues

    # Relationships
    fixture = relationship("Fixture")

    def __repr__(self):
        return f"<Prediction for fixture {self.fixture_id}: {self.match_result_pred}>"

    def to_dict(self):
        """Convert prediction to dictionary."""
        # Calculate prediction percentages
        home_win_pct = round(self.home_win_pred * 100, 1) if self.home_win_pred else 0
        draw_pct = round(self.draw_pred * 100, 1) if self.draw_pred else 0
        away_win_pct = round(self.away_win_pred * 100, 1) if self.away_win_pred else 0
        over_2_5_pct = round(self.over_2_5_pred * 100, 1) if self.over_2_5_pred else 0
        under_2_5_pct = round(self.under_2_5_pred * 100, 1) if self.under_2_5_pred else 0
        btts_yes_pct = round(self.btts_yes_pred * 100, 1) if self.btts_yes_pred else 0
        btts_no_pct = round(self.btts_no_pred * 100, 1) if self.btts_no_pred else 0

        # Calculate confidence percentage
        confidence_pct = round(self.confidence * 100, 1) if self.confidence else 0

        # Determine prediction description based on prediction type
        prediction_description = ""
        if self.prediction_type == "btts":
            prediction_description = f"Both teams to score: {self.btts_pred}"
        elif self.prediction_type == "over_2_5":
            prediction_description = f"Over 2.5 goals"
        elif self.prediction_type == "under_2_5":
            prediction_description = f"Under 2.5 goals"
        elif self.prediction_type == "home_win":
            if self.fixture:
                prediction_description = f"{self.fixture.home_team} to win"
        elif self.prediction_type == "draw":
            prediction_description = f"Match to end in a draw"
        elif self.prediction_type == "away_win":
            if self.fixture:
                prediction_description = f"{self.fixture.away_team} to win"

        result = {
            "id": self.id,
            "fixture_id": self.fixture_id,
            "match_result_pred": self.match_result_pred,
            "home_win_pred": self.home_win_pred,
            "draw_pred": self.draw_pred,
            "away_win_pred": self.away_win_pred,
            "over_under_pred": self.over_under_pred,
            "over_2_5_pred": self.over_2_5_pred,
            "under_2_5_pred": self.under_2_5_pred,
            "btts_pred": self.btts_pred,
            "btts_yes_pred": self.btts_yes_pred,
            "btts_no_pred": self.btts_no_pred,
            "prediction_type": self.prediction_type,
            "odds": self.odds,
            "confidence": self.confidence,
            "combined_odds": self.combined_odds,
            "combined_confidence": self.combined_confidence,
            "combo_id": self.combo_id,
            "rollover_day": self.rollover_day,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,

            # Add prediction percentages
            "home_win_pct": home_win_pct,
            "draw_pct": draw_pct,
            "away_win_pct": away_win_pct,
            "over_2_5_pct": over_2_5_pct,
            "under_2_5_pct": under_2_5_pct,
            "btts_yes_pct": btts_yes_pct,
            "btts_no_pct": btts_no_pct,
            "confidence_pct": confidence_pct,

            # Add prediction description
            "description": prediction_description,
            "explanation": prediction_description
        }

        # Add fixture data if available
        if self.fixture:
            # Format date for display
            match_date = self.fixture.date.strftime("%b %d, %Y") if self.fixture.date else None

            result["game"] = {
                "id": self.fixture.fixture_id,
                "home_team": self.fixture.home_team,
                "away_team": self.fixture.away_team,
                "league_name": self.fixture.league_name,
                "startTime": self.fixture.date.isoformat() if self.fixture.date else None,
                "date": match_date
            }

            # Add these fields at the top level too for compatibility with frontend
            result["homeTeam"] = self.fixture.home_team
            result["awayTeam"] = self.fixture.away_team
            result["league"] = self.fixture.league_name
            result["matchDate"] = self.fixture.date.isoformat() if self.fixture.date else None
            result["date"] = match_date

        return result

    @classmethod
    def from_prediction_data(cls, prediction_data):
        """Create a Prediction instance from prediction data."""
        return cls(
            fixture_id=prediction_data["fixture_id"],
            match_result_pred=prediction_data["match_result_pred"],
            home_win_pred=prediction_data["home_win_pred"],
            draw_pred=prediction_data["draw_pred"],
            away_win_pred=prediction_data["away_win_pred"],
            over_under_pred=prediction_data["over_under_pred"],
            over_2_5_pred=prediction_data["over_2_5_pred"],
            under_2_5_pred=prediction_data["under_2_5_pred"],
            btts_pred=prediction_data["btts_pred"],
            btts_yes_pred=prediction_data["btts_yes_pred"],
            btts_no_pred=prediction_data["btts_no_pred"]
        )


