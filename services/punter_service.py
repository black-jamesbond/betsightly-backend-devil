"""
Punter Service

This module provides functionality to manage punters and their predictions.
It handles punter creation, prediction extraction, and performance tracking.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

# Import models directly
from punter import Punter
from prediction import Prediction
from database import get_db
from utils.common import setup_logging
from utils.config import settings

# Set up logging
logger = setup_logging(__name__)

class PunterService:
    """
    Service for managing punters and their predictions.

    Features:
    - Create and update punters
    - Extract predictions from Telegram
    - Track punter performance
    - Validate and store predictions
    """

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize the punter service.

        Args:
            db: Database session
        """
        self.db = db or next(get_db())

        # Ensure cache directory exists
        self.cache_dir = os.path.join(settings.punter.CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_all_punters(self, include_codes: bool = True) -> List[Dict[str, Any]]:
        """
        Get all punters, optionally with their recent betting codes.

        Returns:
            List of punters (each with a ``betting_codes`` list when *include_codes* is True)
        """
        try:
            punters = self.db.query(Punter).all()
            result = []

            for punter in punters:
                data = punter.to_dict()
                if include_codes:
                    from betting_code import BettingCode
                    codes = (
                        self.db.query(BettingCode)
                        .filter(BettingCode.punter_id == punter.id)
                        .order_by(BettingCode.created_at.desc())
                        .limit(10)
                        .all()
                    )
                    data["betting_codes"] = [c.to_dict() for c in codes]
                result.append(data)

            return result

        except Exception as e:
            logger.error(f"Error getting all punters: {str(e)}")
            return []

    def get_punter_by_id(self, punter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get punter by ID.

        Args:
            punter_id: Punter ID

        Returns:
            Punter data or None if not found
        """
        try:
            punter = self.db.query(Punter).filter(Punter.id == punter_id).first()

            if not punter:
                return None

            return punter.to_dict()

        except Exception as e:
            logger.error(f"Error getting punter by ID {punter_id}: {str(e)}")
            return None



    def create_punter(self, punter_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new punter.

        Args:
            punter_data: Punter data

        Returns:
            Created punter data or None if creation failed
        """
        try:
            # Create new punter
            punter = Punter(
                name=punter_data.get("name"),
                nickname=punter_data.get("nickname"),
                country=punter_data.get("country", "Nigeria"),
                popularity=punter_data.get("popularity", 0),
                specialty=punter_data.get("specialty"),
                success_rate=punter_data.get("success_rate"),
                image_url=punter_data.get("image_url"),
                social_media=punter_data.get("social_media"),
                bio=punter_data.get("bio"),
                verified=punter_data.get("verified", False),
                created_at=datetime.now()
            )

            # Add to database
            self.db.add(punter)
            self.db.commit()
            self.db.refresh(punter)

            logger.info(f"Created punter: {punter.name} (ID: {punter.id})")

            return punter.to_dict()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating punter: {str(e)}")
            return None

    def update_punter(self, punter_id: str, punter_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing punter.

        Args:
            punter_id: Punter ID
            punter_data: Updated punter data

        Returns:
            Updated punter data or None if update failed
        """
        try:
            # Get punter
            punter = self.db.query(Punter).filter(Punter.id == punter_id).first()

            if not punter:
                logger.warning(f"Punter with ID {punter_id} not found")
                return None

            # Update fields
            if "name" in punter_data:
                punter.name = punter_data["name"]

            if "nickname" in punter_data:
                punter.nickname = punter_data["nickname"]

            if "country" in punter_data:
                punter.country = punter_data["country"]

            if "popularity" in punter_data:
                punter.popularity = punter_data["popularity"]

            if "specialty" in punter_data:
                punter.specialty = punter_data["specialty"]

            if "success_rate" in punter_data:
                punter.success_rate = punter_data["success_rate"]

            if "image_url" in punter_data:
                punter.image_url = punter_data["image_url"]

            if "social_media" in punter_data:
                punter.social_media = punter_data["social_media"]

            if "bio" in punter_data:
                punter.bio = punter_data["bio"]

            if "verified" in punter_data:
                punter.verified = punter_data["verified"]

            # Update last updated timestamp
            punter.updated_at = datetime.now()

            # Commit changes
            self.db.commit()
            self.db.refresh(punter)

            logger.info(f"Updated punter: {punter.name} (ID: {punter.id})")

            return punter.to_dict()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating punter: {str(e)}")
            return None

    def delete_punter(self, punter_id: str) -> bool:
        """
        Delete a punter.

        Args:
            punter_id: Punter ID

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            # Get punter
            punter = self.db.query(Punter).filter(Punter.id == punter_id).first()

            if not punter:
                logger.warning(f"Punter with ID {punter_id} not found")
                return False

            # Delete punter
            self.db.delete(punter)
            self.db.commit()

            logger.info(f"Deleted punter: {punter.name} (ID: {punter.id})")

            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting punter: {str(e)}")
            return False

    def get_punter_predictions(self, punter_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get predictions by a punter.

        Args:
            punter_id: Punter ID
            limit: Maximum number of predictions to return

        Returns:
            List of predictions
        """
        try:
            # Get predictions
            predictions = (
                self.db.query(Prediction)
                .filter(Prediction.punter_id == punter_id)
                .order_by(desc(Prediction.created_at))
                .limit(limit)
                .all()
            )

            return [prediction.to_dict() for prediction in predictions]

        except Exception as e:
            logger.error(f"Error getting predictions for punter {punter_id}: {str(e)}")
            return []

    def get_top_punters(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top punters based on win rate.

        Args:
            limit: Maximum number of punters to return

        Returns:
            List of top punters
        """
        try:
            # Get punters with at least 10 predictions
            punters = (
                self.db.query(
                    Punter,
                    func.count(Prediction.id).label("total_predictions"),
                    func.sum(Prediction.status == "won").label("won_predictions")
                )
                .join(Prediction, Punter.id == Prediction.punter_id)
                .group_by(Punter.id)
                .having(func.count(Prediction.id) >= 10)
                .order_by(desc(func.sum(Prediction.status == "won") / func.count(Prediction.id)))
                .limit(limit)
                .all()
            )

            # Format results
            result = []

            for punter, total_predictions, won_predictions in punters:
                win_rate = won_predictions / total_predictions if total_predictions > 0 else 0

                result.append({
                    **punter.to_dict(),
                    "total_predictions": total_predictions,
                    "won_predictions": won_predictions,
                    "win_rate": win_rate
                })

            return result

        except Exception as e:
            logger.error(f"Error getting top punters: {str(e)}")
            return []



    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Save a prediction to the database.

        Args:
            prediction_data: Prediction data

        Returns:
            Saved prediction data or None if save failed
        """
        try:
            # Check if prediction already exists
            existing_prediction = (
                self.db.query(Prediction)
                .filter(
                    Prediction.punter_id == prediction_data["punter_id"],
                    Prediction.home_team == prediction_data["home_team"],
                    Prediction.away_team == prediction_data["away_team"],
                    Prediction.prediction_type == prediction_data["prediction_type"]
                )
                .first()
            )

            if existing_prediction:
                logger.warning(f"Prediction already exists: {existing_prediction.id}")
                return existing_prediction.to_dict()

            # Create new prediction
            prediction = Prediction(
                punter_id=prediction_data["punter_id"],
                home_team=prediction_data["home_team"],
                away_team=prediction_data["away_team"],
                prediction_type=prediction_data["prediction_type"],
                prediction=prediction_data.get("prediction"),
                odds=prediction_data.get("odds"),
                match_datetime=prediction_data.get("match_datetime"),
                source=prediction_data.get("source"),
                source_id=prediction_data.get("message_id"),
                source_text=prediction_data.get("message_text"),
                confidence=prediction_data.get("confidence_score", 0.5),
                status="pending",
                created_at=datetime.now()
            )

            # Add to database
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)

            logger.info(f"Saved prediction: {prediction.id}")

            # Save betting code if provided
            if prediction_data.get("bet_code") and prediction_data.get("odds"):
                from betting_code import BettingCode
                from bookmaker import Bookmaker

                # Get or create bookmaker if provided
                bookmaker_id = None
                if prediction_data.get("bookmaker"):
                    bookmaker = self.db.query(Bookmaker).filter(Bookmaker.name == prediction_data["bookmaker"]).first()

                    if not bookmaker:
                        # Create new bookmaker
                        bookmaker = Bookmaker(
                            name=prediction_data["bookmaker"],
                            created_at=datetime.now()
                        )
                        self.db.add(bookmaker)
                        self.db.commit()
                        self.db.refresh(bookmaker)

                    bookmaker_id = bookmaker.id

                # Check if betting code already exists
                existing_code = self.db.query(BettingCode).filter(BettingCode.code == prediction_data["bet_code"]).first()

                if not existing_code:
                    # Create new betting code
                    betting_code = BettingCode(
                        code=prediction_data["bet_code"],
                        punter_id=prediction_data["punter_id"],
                        bookmaker_id=bookmaker_id,
                        odds=prediction_data["odds"],
                        event_date=prediction_data.get("match_datetime"),
                        status="pending",
                        confidence=int(prediction_data.get("confidence_score", 0.5) * 10),  # Convert to 1-10 scale
                        featured=False,
                        notes=f"Match: {prediction_data['home_team']} vs {prediction_data['away_team']}\nPrediction: {prediction_data.get('prediction')}",
                        created_at=datetime.now()
                    )

                    self.db.add(betting_code)
                    self.db.commit()
                    self.db.refresh(betting_code)

                    logger.info(f"Saved betting code: {betting_code.code} (ID: {betting_code.id})")

            return prediction.to_dict()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving prediction: {str(e)}")
            return None

    def update_prediction_status(self, prediction_id: str, status: str) -> Optional[Dict[str, Any]]:
        """
        Update prediction status.

        Args:
            prediction_id: Prediction ID
            status: New status (won, lost, pending, void)

        Returns:
            Updated prediction data or None if update failed
        """
        try:
            # Get prediction
            prediction = self.db.query(Prediction).filter(Prediction.id == prediction_id).first()

            if not prediction:
                logger.warning(f"Prediction with ID {prediction_id} not found")
                return None

            # Update status
            prediction.status = status
            prediction.updated_at = datetime.now()

            # Commit changes
            self.db.commit()
            self.db.refresh(prediction)

            logger.info(f"Updated prediction status: {prediction.id} -> {status}")

            return prediction.to_dict()

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating prediction status: {str(e)}")
            return None

    def get_punter_performance(self, punter_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a punter.

        Args:
            punter_id: Punter ID

        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get punter
            punter = self.db.query(Punter).filter(Punter.id == punter_id).first()

            if not punter:
                logger.warning(f"Punter with ID {punter_id} not found")
                return {}

            # Get predictions
            predictions = self.db.query(Prediction).filter(Prediction.punter_id == punter_id).all()

            # Calculate metrics
            total_predictions = len(predictions)
            won_predictions = sum(1 for p in predictions if p.status == "won")
            lost_predictions = sum(1 for p in predictions if p.status == "lost")
            pending_predictions = sum(1 for p in predictions if p.status == "pending")
            void_predictions = sum(1 for p in predictions if p.status == "void")

            win_rate = won_predictions / (won_predictions + lost_predictions) if (won_predictions + lost_predictions) > 0 else 0

            # Calculate profit/loss
            profit_loss = sum((p.odds - 1) for p in predictions if p.status == "won") - lost_predictions

            # Calculate ROI
            roi = profit_loss / total_predictions if total_predictions > 0 else 0

            # Calculate average odds
            average_odds = sum(p.odds for p in predictions if p.odds) / sum(1 for p in predictions if p.odds) if sum(1 for p in predictions if p.odds) > 0 else 0

            # Calculate metrics by prediction type
            prediction_types = {}

            for prediction in predictions:
                prediction_type = prediction.prediction_type

                if prediction_type not in prediction_types:
                    prediction_types[prediction_type] = {
                        "total": 0,
                        "won": 0,
                        "lost": 0,
                        "pending": 0,
                        "void": 0
                    }

                prediction_types[prediction_type]["total"] += 1

                if prediction.status == "won":
                    prediction_types[prediction_type]["won"] += 1
                elif prediction.status == "lost":
                    prediction_types[prediction_type]["lost"] += 1
                elif prediction.status == "pending":
                    prediction_types[prediction_type]["pending"] += 1
                elif prediction.status == "void":
                    prediction_types[prediction_type]["void"] += 1

            # Calculate win rate for each prediction type
            for prediction_type, metrics in prediction_types.items():
                metrics["win_rate"] = metrics["won"] / (metrics["won"] + metrics["lost"]) if (metrics["won"] + metrics["lost"]) > 0 else 0

            # Return performance metrics
            return {
                "punter_id": punter_id,
                "punter_name": punter.name,
                "total_predictions": total_predictions,
                "won_predictions": won_predictions,
                "lost_predictions": lost_predictions,
                "pending_predictions": pending_predictions,
                "void_predictions": void_predictions,
                "win_rate": win_rate,
                "profit_loss": profit_loss,
                "roi": roi,
                "average_odds": average_odds,
                "prediction_types": prediction_types
            }

        except Exception as e:
            logger.error(f"Error getting punter performance: {str(e)}")
            return {}

# Create a singleton instance
punter_service = PunterService()
