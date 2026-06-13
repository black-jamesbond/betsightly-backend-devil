#!/usr/bin/env python3
"""
Quick Prediction Service

Uses the trained quick models to make predictions.
Integrates with your working API keys for live fixture data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import settings

logger = logging.getLogger(__name__)

class QuickPredictionService:
    """
    Quick prediction service using trained models and live API data.
    """
    
    def __init__(self):
        """Initialize the prediction service (models loaded lazily on first use)."""
        self.models = {}
        self.encoders = {}
        self.model_dir = Path("models/quick")
        self._loaded = False

    def _ensure_loaded(self):
        """Load models on first use."""
        if self._loaded:
            return
        self._load_models()
        self._loaded = True
        logger.info("Quick Prediction Service models loaded")
    
    def _load_models(self):
        """Load trained models and encoders."""
        try:
            model_types = ['match_result', 'over_under', 'btts']
            
            for model_type in model_types:
                model_path = self.model_dir / f"{model_type}_model.joblib"
                encoder_path = self.model_dir / f"{model_type}_encoder.joblib"
                
                if model_path.exists() and encoder_path.exists():
                    self.models[model_type] = joblib.load(model_path)
                    self.encoders[model_type] = joblib.load(encoder_path)
                    logger.info(f"✅ Loaded {model_type} model")
                else:
                    logger.warning(f"⚠️  {model_type} model not found")
            
            logger.info(f"📊 Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
    
    def get_predictions_for_date(self, date: str = None) -> Dict[str, Any]:
        """
        Get predictions for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with predictions and categories
        """
        self._ensure_loaded()

        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            # Get fixtures from API
            fixtures = self._get_fixtures_from_api(date)
            
            if not fixtures:
                return {
                    "status": "success",
                    "date": date,
                    "predictions": [],
                    "categories": {
                        "2_odds": [],
                        "5_odds": [],
                        "10_odds": [],
                        "rollover": []
                    },
                    "message": "No fixtures found for this date"
                }
            
            # Generate predictions
            predictions = []
            for fixture in fixtures:
                prediction = self._predict_fixture(fixture)
                if prediction:
                    predictions.append(prediction)
            
            # Categorize predictions
            categories = self._categorize_predictions(predictions)
            
            return {
                "status": "success",
                "date": date,
                "predictions": predictions,
                "categories": categories,
                "summary": {
                    "total_fixtures": len(fixtures),
                    "total_predictions": len(predictions),
                    "high_confidence_predictions": len([p for p in predictions if p.get("confidence", 0) > 70])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting predictions: {str(e)}")
            return {
                "status": "error",
                "date": date,
                "message": str(e)
            }
    
    def _get_fixtures_from_api(self, date: str) -> List[Dict]:
        """Get fixtures from API using your configured keys - NO MOCKS."""
        try:
            # Load environment variables
            import os
            from dotenv import load_dotenv
            load_dotenv()

            football_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
            api_football_key = os.getenv("API_FOOTBALL_API_KEY", "")

            # Try Football-Data.org first
            if football_key and "dummy" not in football_key and len(football_key) > 10:
                logger.info(f"🌐 Using Football-Data.org API (key: {football_key[:10]}...)")
                return self._get_fixtures_football_data(date)

            # Try API-Football as fallback
            elif api_football_key and "dummy" not in api_football_key and len(api_football_key) > 10:
                logger.info(f"🌐 Using API-Football API (key: {api_football_key[:10]}...)")
                return self._get_fixtures_api_football(date)

            else:
                # NO MOCKS - return empty if no API keys
                logger.warning("⚠️  No valid API keys found - returning no fixtures")
                logger.warning(f"Football key: {football_key[:10] if football_key else 'None'}...")
                logger.warning(f"API-Football key: {api_football_key[:10] if api_football_key else 'None'}...")
                return []

        except Exception as e:
            logger.error(f"❌ Error fetching fixtures: {str(e)}")
            return []
    
    def _get_fixtures_football_data(self, date: str) -> List[Dict]:
        """Get fixtures from Football-Data.org API."""
        import requests
        import os

        api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        headers = {"X-Auth-Token": api_key}
        url = "https://api.football-data.org/v4/matches"

        # Use a broader date range to catch more fixtures
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        end_date = date_obj + timedelta(days=1)

        params = {
            "dateFrom": date,
            "dateTo": end_date.strftime("%Y-%m-%d")
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        matches = data.get("matches", [])
        
        # Convert to standard format
        fixtures = []
        for match in matches:
            fixtures.append({
                "id": match.get("id"),
                "home_team": match.get("homeTeam", {}).get("name", "Unknown"),
                "away_team": match.get("awayTeam", {}).get("name", "Unknown"),
                "date": match.get("utcDate"),
                "competition": match.get("competition", {}).get("name", "Unknown")
            })
        
        logger.info(f"📡 Fetched {len(fixtures)} fixtures from Football-Data.org")
        return fixtures
    
    def _get_fixtures_api_football(self, date: str) -> List[Dict]:
        """Get fixtures from API-Football."""
        import requests
        import os

        api_key = os.getenv("API_FOOTBALL_API_KEY", "")
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v3.football.api-sports.io"
        }
        url = "https://v3.football.api-sports.io/fixtures"
        params = {"date": date}
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        fixtures_data = data.get("response", [])
        
        # Convert to standard format
        fixtures = []
        for fixture in fixtures_data:
            fixtures.append({
                "id": fixture.get("fixture", {}).get("id"),
                "home_team": fixture.get("teams", {}).get("home", {}).get("name", "Unknown"),
                "away_team": fixture.get("teams", {}).get("away", {}).get("name", "Unknown"),
                "date": fixture.get("fixture", {}).get("date"),
                "competition": fixture.get("league", {}).get("name", "Unknown")
            })
        
        logger.info(f"📡 Fetched {len(fixtures)} fixtures from API-Football")
        return fixtures
    

    
    def _predict_fixture(self, fixture: Dict) -> Optional[Dict]:
        """Make predictions for a single fixture."""
        try:
            # Create features for this fixture
            features = self._create_fixture_features(fixture)
            
            # Make predictions with all models
            predictions = {}
            
            for model_type, model in self.models.items():
                try:
                    # Get prediction probabilities
                    proba = model.predict_proba([features])[0]
                    
                    # Get class prediction
                    pred_class = model.predict([features])[0]
                    
                    # Decode prediction
                    encoder = self.encoders[model_type]
                    prediction = encoder.inverse_transform([pred_class])[0]
                    
                    # Get confidence (max probability)
                    confidence = max(proba) * 100
                    
                    predictions[model_type] = {
                        "prediction": prediction,
                        "confidence": round(confidence, 1),
                        "probabilities": [round(p, 3) for p in proba]
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error predicting {model_type}: {str(e)}")
            
            if predictions:
                return {
                    "fixture": fixture,
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": round(np.mean([p["confidence"] for p in predictions.values()]), 1)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error predicting fixture: {str(e)}")
            return None
    
    def _create_fixture_features(self, fixture: Dict) -> List[float]:
        """Create features for a fixture (simplified version)."""
        # Simple feature creation based on team names
        home_team = fixture.get("home_team", "Unknown")
        away_team = fixture.get("away_team", "Unknown")
        
        # Hash-based team strength (consistent but arbitrary)
        home_strength = (hash(home_team) % 100) / 100
        away_strength = (hash(away_team) % 100) / 100
        
        # Create features matching training format
        features = [
            hash(home_team) % 50,  # home_team_id
            hash(away_team) % 50,  # away_team_id
            1.5 + home_strength,   # home_goals_avg
            1.2 + away_strength,   # away_goals_avg
            2.7 + home_strength + away_strength,  # total_goals
            home_strength - away_strength,        # goal_difference
            1.5 + home_strength,   # home_team_strength
            1.2 + away_strength,   # away_team_strength
            home_strength - away_strength,  # strength_difference
            0.3  # home_advantage
        ]
        
        return features
    
    def _categorize_predictions(self, predictions: List[Dict]) -> Dict[str, List]:
        """Categorize predictions by odds/confidence."""
        categories = {
            "2_odds": [],
            "5_odds": [],
            "10_odds": [],
            "rollover": []
        }
        
        for prediction in predictions:
            confidence = prediction.get("confidence", 0)
            
            # Simple categorization based on confidence
            if confidence >= 70:
                categories["2_odds"].append(prediction)
            elif confidence >= 50:
                categories["5_odds"].append(prediction)
            elif confidence >= 30:
                categories["10_odds"].append(prediction)
            else:
                categories["rollover"].append(prediction)
        
        return categories


# Global service instance
quick_prediction_service = QuickPredictionService()


def get_predictions_for_date(date: str = None) -> Dict[str, Any]:
    """Convenience function to get predictions."""
    return quick_prediction_service.get_predictions_for_date(date)


if __name__ == "__main__":
    # Test the service
    print("🔍 Testing Quick Prediction Service")
    print("=" * 40)
    
    predictions = get_predictions_for_date()
    print(f"Status: {predictions.get('status')}")
    print(f"Predictions: {len(predictions.get('predictions', []))}")
    print(f"Categories: {[(k, len(v)) for k, v in predictions.get('categories', {}).items()]}")
