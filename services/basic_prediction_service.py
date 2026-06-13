"""
Basic Prediction Service for Render Deployment

A lightweight prediction service that provides real football data
without heavy ML dependencies for stable deployment.
"""

import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BasicPrediction:
    """Basic prediction data structure."""
    id: int
    home_team: str
    away_team: str
    league: str
    bet_type: str
    prediction: str
    confidence: float
    odds: float
    date: str
    reasoning: str = ""

class BasicPredictionService:
    """
    Basic prediction service using real football data with simple algorithms.
    
    This service provides real predictions without heavy ML dependencies,
    making it suitable for stable deployment on Render.
    """
    
    def __init__(self):
        self.football_data_api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        self.api_football_key = os.getenv("API_FOOTBALL_API_KEY")
        self.base_url_football_data = "https://api.football-data.org/v4"
        self.base_url_api_football = "https://v3.football.api-sports.io"
        
    def get_predictions_for_date(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Get predictions for a specific date using real football data.
        
        Args:
            date_str: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with categorized predictions
        """
        try:
            if not date_str:
                date_str = datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"Getting basic predictions for {date_str}")
            
            # Get real fixtures for the date
            fixtures = self._get_fixtures_for_date(date_str)
            
            if not fixtures:
                logger.warning(f"No fixtures found for {date_str}, using mock data")
                return self._get_mock_predictions(date_str)
            
            # Generate predictions from real fixtures
            predictions = self._generate_predictions_from_fixtures(fixtures, date_str)
            
            # Categorize predictions
            categorized = self._categorize_predictions(predictions)
            
            return {
                "status": "success",
                "date": date_str,
                "categories": categorized,
                "total_predictions": len(predictions),
                "data_source": "real_football_data",
                "service": "basic_prediction_service"
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            # Fallback to mock data
            return self._get_mock_predictions(date_str)
    
    def _get_fixtures_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Get fixtures from Football Data API."""
        try:
            if not self.football_data_api_key or self.football_data_api_key == "your-football-data-api-key-here":
                logger.warning("Football Data API key not configured")
                return []
            
            headers = {"X-Auth-Token": self.football_data_api_key}
            
            # Get fixtures for major leagues
            leagues = ["PL", "BL1", "SA", "PD", "FL1"]  # Premier League, Bundesliga, Serie A, La Liga, Ligue 1
            all_fixtures = []
            
            for league in leagues:
                try:
                    url = f"{self.base_url_football_data}/competitions/{league}/matches"
                    params = {
                        "dateFrom": date_str,
                        "dateTo": date_str,
                        "status": "SCHEDULED"
                    }
                    
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        fixtures = data.get("matches", [])
                        all_fixtures.extend(fixtures)
                        logger.info(f"Found {len(fixtures)} fixtures for {league}")
                    else:
                        logger.warning(f"API error for {league}: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error fetching {league} fixtures: {str(e)}")
                    continue
            
            return all_fixtures[:10]  # Limit to 10 fixtures
            
        except Exception as e:
            logger.error(f"Error fetching fixtures: {str(e)}")
            return []
    
    def _generate_predictions_from_fixtures(self, fixtures: List[Dict], date_str: str) -> List[BasicPrediction]:
        """Generate predictions from real fixtures using simple algorithms."""
        predictions = []
        
        for i, fixture in enumerate(fixtures):
            try:
                home_team = fixture.get("homeTeam", {}).get("name", "Unknown")
                away_team = fixture.get("awayTeam", {}).get("name", "Unknown")
                competition = fixture.get("competition", {}).get("name", "Unknown League")
                
                # Simple prediction algorithm based on team names and basic heuristics
                prediction_data = self._simple_prediction_algorithm(home_team, away_team)
                
                prediction = BasicPrediction(
                    id=i + 1,
                    home_team=home_team,
                    away_team=away_team,
                    league=competition,
                    bet_type=prediction_data["bet_type"],
                    prediction=prediction_data["prediction"],
                    confidence=prediction_data["confidence"],
                    odds=prediction_data["odds"],
                    date=date_str,
                    reasoning=prediction_data["reasoning"]
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error generating prediction for fixture: {str(e)}")
                continue
        
        return predictions
    
    def _simple_prediction_algorithm(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """
        Simple prediction algorithm based on team strength heuristics.
        
        This is a basic algorithm that can be enhanced with real ML models later.
        """
        # Simple team strength mapping (can be enhanced with real data)
        strong_teams = {
            "Manchester City": 0.9, "Arsenal": 0.85, "Liverpool": 0.85, "Chelsea": 0.8,
            "Manchester United": 0.75, "Tottenham": 0.7, "Newcastle": 0.7,
            "Bayern Munich": 0.9, "Borussia Dortmund": 0.8, "RB Leipzig": 0.75,
            "Real Madrid": 0.9, "Barcelona": 0.85, "Atletico Madrid": 0.8,
            "Juventus": 0.8, "AC Milan": 0.75, "Inter": 0.8, "Napoli": 0.75,
            "Paris Saint-Germain": 0.85, "Marseille": 0.7, "Lyon": 0.7
        }
        
        home_strength = strong_teams.get(home_team, 0.5)
        away_strength = strong_teams.get(away_team, 0.5)
        
        # Home advantage
        home_strength += 0.1
        
        # Determine prediction
        if home_strength > away_strength + 0.2:
            return {
                "bet_type": "Match Result",
                "prediction": f"{home_team} Win",
                "confidence": min(0.8, home_strength),
                "odds": 2.0 - (home_strength - away_strength),
                "reasoning": f"{home_team} has strong home advantage"
            }
        elif away_strength > home_strength + 0.1:
            return {
                "bet_type": "Match Result", 
                "prediction": f"{away_team} Win",
                "confidence": min(0.75, away_strength),
                "odds": 2.2 - (away_strength - home_strength),
                "reasoning": f"{away_team} is the stronger team"
            }
        else:
            return {
                "bet_type": "Over/Under",
                "prediction": "Over 2.5 Goals",
                "confidence": 0.65,
                "odds": 1.8,
                "reasoning": "Evenly matched teams often produce goals"
            }
    
    def _categorize_predictions(self, predictions: List[BasicPrediction]) -> Dict[str, List[Dict]]:
        """Categorize predictions by odds/risk level."""
        categories = {
            "2_odds": [],
            "5_odds": [],
            "10_odds": [],
            "rollover": []
        }
        
        for pred in predictions:
            pred_dict = {
                "id": pred.id,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "league": pred.league,
                "bet_type": pred.bet_type,
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "odds": pred.odds,
                "date": pred.date,
                "reasoning": pred.reasoning
            }
            
            # Categorize by odds
            if pred.odds <= 2.5:
                categories["2_odds"].append(pred_dict)
            elif pred.odds <= 5.0:
                categories["5_odds"].append(pred_dict)
            else:
                categories["10_odds"].append(pred_dict)
            
            # Add to rollover if confidence is high
            if pred.confidence >= 0.7:
                categories["rollover"].append(pred_dict)
        
        return categories
    
    def _get_mock_predictions(self, date_str: str) -> Dict[str, Any]:
        """Fallback mock predictions when real data is unavailable."""
        mock_predictions = [
            BasicPrediction(
                id=1,
                home_team="Arsenal",
                away_team="Chelsea", 
                league="Premier League",
                bet_type="Match Result",
                prediction="Arsenal Win",
                confidence=0.75,
                odds=2.1,
                date=date_str,
                reasoning="Arsenal has strong home form"
            ),
            BasicPrediction(
                id=2,
                home_team="Manchester City",
                away_team="Liverpool",
                league="Premier League", 
                bet_type="Over/Under",
                prediction="Over 2.5 Goals",
                confidence=0.68,
                odds=1.8,
                date=date_str,
                reasoning="Both teams have strong attacking records"
            )
        ]
        
        categorized = self._categorize_predictions(mock_predictions)
        
        return {
            "status": "success",
            "date": date_str,
            "categories": categorized,
            "total_predictions": len(mock_predictions),
            "data_source": "mock_data",
            "service": "basic_prediction_service"
        }

# Create global instance
basic_prediction_service = BasicPredictionService()
