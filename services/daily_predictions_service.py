"""
Daily Predictions Service
Generates predictions once per day and stores them in database.
"""

import logging
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base

from database import get_db, Base, engine
from services.apifootball_service import APIFootballService
from api.endpoints.ml_predictions import RealMLPredictionService
from services.accumulator_builder import AccumulatorBuilder

# Set up logging
logger = logging.getLogger(__name__)

# Database Models
class DailyPrediction(Base):
    """Store daily predictions in database."""
    __tablename__ = "daily_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    fixture_id = Column(Integer, nullable=False, index=True)
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    league_name = Column(String(255), nullable=False)
    fixture_date = Column(DateTime, nullable=False)
    fixture_status = Column(String(50), nullable=False)
    
    # ML Predictions (JSON)
    ml_predictions = Column(Text, nullable=False)  # JSON string
    
    # Betting Categories
    betting_2_odds = Column(Text, nullable=True)  # JSON string
    betting_5_odds = Column(Text, nullable=True)  # JSON string
    betting_10_odds = Column(Text, nullable=True)  # JSON string
    betting_over_1_5 = Column(Text, nullable=True)  # JSON string
    betting_rollover = Column(Text, nullable=True)  # JSON string
    
    # Metadata
    total_models_used = Column(Integer, default=0)
    highest_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DailyPredictionSummary(Base):
    """Store daily prediction summary."""
    __tablename__ = "daily_prediction_summary"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(Date, nullable=False, unique=True, index=True)
    total_fixtures = Column(Integer, default=0)
    upcoming_fixtures = Column(Integer, default=0)
    predictions_generated = Column(Integer, default=0)
    models_used = Column(Integer, default=0)
    
    # Category counts
    betting_2_odds_count = Column(Integer, default=0)
    betting_5_odds_count = Column(Integer, default=0)
    betting_10_odds_count = Column(Integer, default=0)
    betting_over_1_5_count = Column(Integer, default=0)
    betting_rollover_count = Column(Integer, default=0)
    
    generation_status = Column(String(50), default="pending")  # pending, completed, failed
    generation_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DailyPredictionsService:
    """Service to generate and manage daily predictions."""
    
    def __init__(self):
        """Initialize the service."""
        self.apifootball_service = APIFootballService()
        self.ml_service = RealMLPredictionService()
        self.accumulator_builder = AccumulatorBuilder()

        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Daily predictions tables created")
    
    def generate_daily_predictions(self, target_date: str = None) -> Dict[str, Any]:
        """
        Generate predictions for a specific date and store in database.
        
        Args:
            target_date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Generation result summary
        """
        try:
            if target_date is None:
                target_date = datetime.now().strftime("%Y-%m-%d")
            
            prediction_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            
            logger.info(f"🎯 Generating daily predictions for {target_date}")
            
            # Get database session
            db = next(get_db())
            
            # Check if predictions already exist for this date
            existing_summary = db.query(DailyPredictionSummary).filter(
                DailyPredictionSummary.prediction_date == prediction_date
            ).first()
            
            if existing_summary and existing_summary.generation_status == "completed":
                logger.info(f"✅ Predictions already exist for {target_date}")
                return {
                    "status": "already_exists",
                    "date": target_date,
                    "message": "Predictions already generated for this date",
                    "summary": self._summary_to_dict(existing_summary)
                }
            
            # Create or update summary record
            if existing_summary:
                summary = existing_summary
                summary.generation_status = "pending"
                summary.updated_at = datetime.utcnow()
            else:
                summary = DailyPredictionSummary(
                    prediction_date=prediction_date,
                    generation_status="pending"
                )
                db.add(summary)
            
            db.commit()
            
            # Get fixtures for the date
            all_fixtures = self.apifootball_service.get_daily_fixtures(target_date)
            summary.total_fixtures = len(all_fixtures)
            
            # Filter for upcoming fixtures
            upcoming_fixtures = self._filter_upcoming_fixtures(all_fixtures)
            summary.upcoming_fixtures = len(upcoming_fixtures)
            
            logger.info(f"📊 Found {len(upcoming_fixtures)} upcoming fixtures")
            
            # Generate predictions for each fixture
            predictions_count = 0
            all_predictions = []

            # Clear existing predictions for this date
            db.query(DailyPrediction).filter(
                DailyPrediction.prediction_date == prediction_date
            ).delete()

            # Generate ML predictions for all fixtures
            for fixture in upcoming_fixtures:
                try:
                    # Generate ML prediction
                    prediction_result = self.ml_service.generate_predictions_for_fixture(fixture)

                    if 'error' not in prediction_result:
                        all_predictions.append(prediction_result)
                        predictions_count += 1

                except Exception as e:
                    logger.error(f"Error processing fixture {fixture.get('fixture_id')}: {str(e)}")
                    continue

            # Build accumulators from all predictions
            logger.info(f"🎯 Building accumulators from {len(all_predictions)} predictions")
            accumulator_result = self.accumulator_builder.build_accumulators(all_predictions)
            accumulators = accumulator_result.get('accumulators', {})

            # Count successful accumulators
            category_counts = {"2_odds": 0, "5_odds": 0, "10_odds": 0, "over_1_5": 0, "rollover": 0}
            for category, accumulator in accumulators.items():
                if accumulator.get('selected', False):
                    category_counts[category] = 1  # 1 accumulator per category

            # Store individual predictions with accumulator data
            for prediction_result in all_predictions:
                try:
                    # Add accumulator info to prediction
                    prediction_result['accumulator_info'] = {
                        'included_in_accumulators': [],
                        'accumulator_summary': accumulator_result.get('summary', {})
                    }

                    # Check which accumulators include this fixture
                    fixture_id = prediction_result['fixture_info'].get('fixture_id')
                    for category, accumulator in accumulators.items():
                        if accumulator.get('selected', False):
                            for game in accumulator.get('games', []):
                                if game.get('fixture_id') == fixture_id:
                                    prediction_result['accumulator_info']['included_in_accumulators'].append(category)

                    # Store prediction in database
                    db_prediction = self._create_db_prediction_with_accumulators(
                        prediction_result, prediction_date, accumulators
                    )
                    db.add(db_prediction)

                except Exception as e:
                    logger.error(f"Error storing prediction: {str(e)}")
                    continue
            
            # Update summary
            summary.predictions_generated = predictions_count
            summary.models_used = len(self.ml_service.api_models)
            summary.betting_2_odds_count = category_counts["2_odds"]
            summary.betting_5_odds_count = category_counts["5_odds"]
            summary.betting_10_odds_count = category_counts["10_odds"]
            summary.betting_over_1_5_count = category_counts["over_1_5"]
            summary.betting_rollover_count = category_counts["rollover"]
            summary.generation_status = "completed"
            summary.generation_time = datetime.utcnow()
            
            db.commit()

            logger.info(f"✅ Generated {predictions_count} predictions for {target_date}")

            # Build response before closing session (object is still bound)
            result = {
                "status": "success",
                "date": target_date,
                "message": f"Generated {predictions_count} predictions",
                "summary": self._summary_to_dict(summary)
            }
            db.close()
            return result
            
        except Exception as e:
            logger.error(f"Error generating daily predictions: {str(e)}")
            
            # Update summary with error
            try:
                if 'summary' in locals():
                    summary.generation_status = "failed"
                    summary.error_message = str(e)
                    db.commit()
                    db.close()
            except:
                pass
            
            return {
                "status": "error",
                "date": target_date,
                "message": f"Error: {str(e)}"
            }
    
    def _filter_upcoming_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        """Filter for upcoming (not-started) fixtures only.

        API-Football short status codes:
            NS = Not Started, TBD = Time To Be Defined
            1H/2H/HT/ET/BT/P/SUSP/INT = in-play
            FT/AET/PEN = finished
            PST/CANC/ABD/AWD/WO = cancelled/postponed
        """
        upcoming_codes = {"NS", "TBD"}
        upcoming = []

        for fixture in fixtures:
            status = fixture.get("status", "").strip()
            if status in upcoming_codes:
                upcoming.append(fixture)
                continue
            # Fallback for long-form status strings
            if status.lower() in ("not started", "scheduled"):
                upcoming.append(fixture)

        return upcoming
    
    def _create_db_prediction_with_accumulators(self, prediction_result: Dict, prediction_date: date, accumulators: Dict) -> DailyPrediction:
        """Create database prediction record with accumulator data."""
        fixture_info = prediction_result['fixture_info']
        ml_predictions = prediction_result['ml_predictions']

        # Use accumulator data instead of individual betting categories
        betting_categories = {}
        for category, accumulator in accumulators.items():
            betting_categories[category] = accumulator

        # Calculate highest confidence
        highest_confidence = 0.0
        for pred_data in ml_predictions.values():
            confidence = pred_data.get('confidence', 0.0)
            if confidence > highest_confidence:
                highest_confidence = confidence

        return DailyPrediction(
            prediction_date=prediction_date,
            fixture_id=fixture_info.get('fixture_id', 0),
            home_team=fixture_info.get('home_team', ''),
            away_team=fixture_info.get('away_team', ''),
            league_name=fixture_info.get('league', ''),
            fixture_date=datetime.fromisoformat(fixture_info.get('date', datetime.now().isoformat())),
            fixture_status=fixture_info.get('status', ''),
            ml_predictions=json.dumps(ml_predictions),
            betting_2_odds=json.dumps(betting_categories.get('2_odds', {})),
            betting_5_odds=json.dumps(betting_categories.get('5_odds', {})),
            betting_10_odds=json.dumps(betting_categories.get('10_odds', {})),
            betting_over_1_5=json.dumps(betting_categories.get('over_1_5', {})),
            betting_rollover=json.dumps(betting_categories.get('rollover', {})),
            total_models_used=prediction_result['model_summary']['total_predictions'],
            highest_confidence=highest_confidence
        )

    def _create_db_prediction(self, prediction_result: Dict, prediction_date: date) -> DailyPrediction:
        """Create database prediction record."""
        fixture_info = prediction_result['fixture_info']
        ml_predictions = prediction_result['ml_predictions']
        betting_categories = prediction_result['betting_categories']

        # Calculate highest confidence
        highest_confidence = 0.0
        for pred_data in ml_predictions.values():
            confidence = pred_data.get('confidence', 0.0)
            if confidence > highest_confidence:
                highest_confidence = confidence

        return DailyPrediction(
            prediction_date=prediction_date,
            fixture_id=fixture_info.get('fixture_id', 0),
            home_team=fixture_info.get('home_team', ''),
            away_team=fixture_info.get('away_team', ''),
            league_name=fixture_info.get('league', ''),
            fixture_date=datetime.fromisoformat(fixture_info.get('date', datetime.now().isoformat())),
            fixture_status=fixture_info.get('status', ''),
            ml_predictions=json.dumps(ml_predictions),
            betting_2_odds=json.dumps(betting_categories.get('2_odds', {})),
            betting_5_odds=json.dumps(betting_categories.get('5_odds', {})),
            betting_10_odds=json.dumps(betting_categories.get('10_odds', {})),
            betting_over_1_5=json.dumps(betting_categories.get('over_1_5', {})),
            betting_rollover=json.dumps(betting_categories.get('rollover', {})),
            total_models_used=prediction_result['model_summary']['total_predictions'],
            highest_confidence=highest_confidence
        )
    
    def _summary_to_dict(self, summary: DailyPredictionSummary) -> Dict:
        """Convert summary to dictionary."""
        return {
            "prediction_date": summary.prediction_date.isoformat(),
            "total_fixtures": summary.total_fixtures,
            "upcoming_fixtures": summary.upcoming_fixtures,
            "predictions_generated": summary.predictions_generated,
            "models_used": summary.models_used,
            "betting_counts": {
                "2_odds": summary.betting_2_odds_count,
                "5_odds": summary.betting_5_odds_count,
                "10_odds": summary.betting_10_odds_count,
                "over_1_5": summary.betting_over_1_5_count,
                "rollover": summary.betting_rollover_count
            },
            "status": summary.generation_status,
            "generation_time": summary.generation_time.isoformat() if summary.generation_time else None
        }
