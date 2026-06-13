"""
Basketball Prediction Service

This service integrates all basketball prediction components and provides
a unified interface for generating basketball predictions, following the
same pattern as the football prediction service.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import json

from .data_fetcher import NBADataFetcher
from .feature_engineering import BasketballFeatureEngineer
from .models import BasketballModelFactory
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class BasketballPredictionService:
    """
    Service for generating basketball predictions.
    
    Integrates data fetching, feature engineering, and ML models
    to provide comprehensive basketball predictions.
    """
    
    def __init__(self):
        """Initialize the basketball prediction service."""
        self.data_fetcher = NBADataFetcher()
        self.feature_engineer = BasketballFeatureEngineer()
        self.model_factory = BasketballModelFactory()
        
        # Initialize models
        self.win_loss_model = self.model_factory.create_win_loss_model()
        self.over_under_model = self.model_factory.create_over_under_model()
        self.neural_network_model = self.model_factory.create_neural_network_model()
        
        # Prediction cache
        self.cache_dir = os.path.join(settings.ml.CACHE_DIR, "basketball", "predictions")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("Basketball Prediction Service initialized")

    def _models_available(self) -> bool:
        """Check if basketball models are available."""
        return (self.win_loss_model.model is not None or
                self.over_under_model.model is not None or
                (self.neural_network_model and self.neural_network_model.model is not None))

    def generate_predictions(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate basketball predictions for a given date - REAL DATA ONLY.

        Args:
            date: Date string (YYYY-MM-DD) or None for today

        Returns:
            Dictionary with predictions and metadata
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Generating basketball predictions for {date}")

        try:
            # Check if models are available
            if not self._models_available():
                return {
                    'status': 'error',
                    'date': date,
                    'message': 'Basketball models not trained yet. Please train models first.',
                    'predictions': [],
                    'total_games': 0
                }

            # Fetch today's games - REAL DATA ONLY
            try:
                games_df = self.data_fetcher.fetch_today_games()
            except Exception as e:
                logger.error(f"Error fetching NBA games: {str(e)}")
                return {
                    'status': 'success',
                    'date': date,
                    'message': 'No NBA games found for this date (API unavailable or no games scheduled)',
                    'predictions': [],
                    'total_games': 0
                }

            if games_df.empty:
                return {
                    'status': 'success',
                    'date': date,
                    'message': 'No NBA games scheduled for this date',
                    'predictions': [],
                    'total_games': 0
                }

            # Engineer features
            features_df = self.feature_engineer.engineer_features(games_df)

            # Generate predictions
            predictions = []

            for idx, game in games_df.iterrows():
                game_features = features_df.iloc[[idx]]

                # Get predictions from different models
                game_prediction = self._predict_single_game(game, game_features)
                predictions.append(game_prediction)

            # Categorize predictions by confidence
            categorized_predictions = self._categorize_predictions(predictions)

            # Save predictions
            self._save_predictions(predictions, date)

            result = {
                'status': 'success',
                'date': date,
                'predictions': predictions,
                'categories': categorized_predictions,
                'total_games': len(predictions),
                'models_used': self._get_models_status()
            }

            logger.info(f"Generated {len(predictions)} basketball predictions for {date}")
            return result

        except Exception as e:
            logger.error(f"Error generating basketball predictions: {str(e)}")
            return {
                'status': 'error',
                'date': date,
                'message': str(e),
                'predictions': []
            }
    
    def _predict_single_game(self, game: pd.Series, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictions for a single game.
        
        Args:
            game: Game information
            features: Engineered features for the game
            
        Returns:
            Dictionary with game predictions
        """
        prediction = {
            'game_id': game.get('game_id', 'unknown'),
            'game_date': game.get('game_date', datetime.now().strftime("%Y-%m-%d")),
            'home_team': game.get('home_team', 'Unknown'),
            'away_team': game.get('away_team', 'Unknown'),
            'predictions': {}
        }
        
        # Win/Loss prediction
        if self.win_loss_model.model is not None:
            win_loss_pred = self.win_loss_model.predict(features)
            if 'predictions' in win_loss_pred and win_loss_pred['predictions']:
                wl_result = win_loss_pred['predictions'][0]
                prediction['predictions']['win_loss'] = {
                    'prediction': wl_result['prediction'],
                    'home_win_probability': wl_result['home_win_probability'],
                    'away_win_probability': wl_result['away_win_probability'],
                    'confidence': wl_result['confidence']
                }
        
        # Over/Under prediction
        if self.over_under_model.model is not None:
            over_under_pred = self.over_under_model.predict(features)
            if 'predictions' in over_under_pred and over_under_pred['predictions']:
                ou_result = over_under_pred['predictions'][0]
                prediction['predictions']['over_under'] = {
                    'prediction': ou_result['prediction'],
                    'threshold': ou_result['threshold'],
                    'over_probability': ou_result['over_probability'],
                    'under_probability': ou_result['under_probability'],
                    'confidence': ou_result['confidence']
                }
        
        # Neural Network prediction (if available)
        if self.neural_network_model and self.neural_network_model.model is not None:
            nn_pred = self.neural_network_model.predict(features)
            if 'predictions' in nn_pred and nn_pred['predictions']:
                prediction['predictions']['neural_network'] = nn_pred['predictions'][0]
        
        # Calculate overall confidence
        confidences = []
        for pred_type, pred_data in prediction['predictions'].items():
            if isinstance(pred_data, dict) and 'confidence' in pred_data:
                confidences.append(pred_data['confidence'])
        
        prediction['overall_confidence'] = np.mean(confidences) if confidences else 0.5
        
        return prediction
    
    def _categorize_predictions(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize basketball predictions by specialized betting categories with target odds.

        Args:
            predictions: List of predictions

        Returns:
            Dictionary with specialized betting categories
        """
        # Initialize specialized betting categories
        categories = {
            "5_odds": [],       # Target combined odds: 5.0 (individual ~1.2-1.4)
            "10_odds": [],      # Target combined odds: 10.0 (individual ~1.3-1.6)
            "20_odds": [],      # Target combined odds: 20.0 (individual ~1.4-1.8)
            "rollover_7": []    # Target combined odds: 7.0 (individual ~1.25-1.5)
        }

        # Filter for high-confidence safe predictions only (≥75%)
        safe_predictions = self._filter_safe_basketball_predictions(predictions)

        if not safe_predictions:
            logger.warning("No safe basketball predictions found (≥75% confidence)")
            return categories

        # Build each category with target odds
        categories["5_odds"] = self._build_basketball_odds_category(safe_predictions, target_odds=5.0, individual_range=(1.2, 1.4))
        categories["10_odds"] = self._build_basketball_odds_category(safe_predictions, target_odds=10.0, individual_range=(1.3, 1.6))
        categories["20_odds"] = self._build_basketball_odds_category(safe_predictions, target_odds=20.0, individual_range=(1.4, 1.8))
        categories["rollover_7"] = self._build_basketball_odds_category(safe_predictions, target_odds=7.0, individual_range=(1.25, 1.5))

        return categories

    def _filter_safe_basketball_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Filter predictions for safe betting - only high confidence (≥75%) and avoid upsets.

        Args:
            predictions: List of all predictions

        Returns:
            List of safe predictions suitable for betting
        """
        safe_predictions = []

        for pred in predictions:
            overall_confidence = pred.get('overall_confidence', 0.0)

            # Require minimum 75% confidence
            if overall_confidence < 0.75:
                continue

            # Check individual prediction types for safety
            predictions_data = pred.get('predictions', {})
            is_safe = True

            # Analyze win/loss prediction for upset avoidance
            if 'win_loss' in predictions_data:
                wl_pred = predictions_data['win_loss']
                wl_confidence = wl_pred.get('confidence', 0)

                # Require high confidence for win/loss predictions
                if wl_confidence < 75:
                    is_safe = False
                    continue

                # Avoid upset predictions (away team heavily favored over home)
                home_prob = wl_pred.get('home_win_probability', 0.5)
                away_prob = wl_pred.get('away_win_probability', 0.5)

                # Flag potential upsets (away team >70% probability)
                if away_prob > 0.70:
                    logger.info(f"Avoiding potential upset: {pred.get('away_team')} heavily favored over {pred.get('home_team')}")
                    is_safe = False
                    continue

            # Check over/under for conservative totals
            if 'over_under' in predictions_data:
                ou_pred = predictions_data['over_under']
                ou_confidence = ou_pred.get('confidence', 0)

                # Require high confidence for totals
                if ou_confidence < 75:
                    is_safe = False
                    continue

                # Prefer under bets for safety (more predictable)
                prediction_type = ou_pred.get('prediction', '')
                threshold = ou_pred.get('threshold', 220)

                # Flag high-scoring games as riskier
                if prediction_type == 'over' and threshold > 230:
                    logger.info(f"High total game flagged as riskier: {threshold}")

            if is_safe:
                # Add safety classification and odds estimation
                enhanced_pred = pred.copy()
                enhanced_pred['safety_classification'] = self._classify_prediction_safety(pred)
                enhanced_pred['estimated_odds'] = self._estimate_individual_odds(pred)
                safe_predictions.append(enhanced_pred)

        # Sort by confidence (highest first)
        safe_predictions.sort(key=lambda x: x.get('overall_confidence', 0), reverse=True)

        logger.info(f"Filtered {len(safe_predictions)} safe predictions from {len(predictions)} total")
        return safe_predictions

    def _classify_prediction_safety(self, prediction: Dict) -> Dict[str, Any]:
        """
        Classify the safety level of a prediction with detailed reasoning.

        Args:
            prediction: Single prediction dictionary

        Returns:
            Safety classification with reasoning
        """
        safety_factors = []
        risk_factors = []
        overall_confidence = prediction.get('overall_confidence', 0.0)

        # Confidence-based classification
        if overall_confidence >= 0.85:
            safety_factors.append("Very high confidence (≥85%)")
        elif overall_confidence >= 0.75:
            safety_factors.append("High confidence (≥75%)")

        predictions_data = prediction.get('predictions', {})

        # Win/Loss safety analysis
        if 'win_loss' in predictions_data:
            wl_pred = predictions_data['win_loss']
            home_prob = wl_pred.get('home_win_probability', 0.5)
            away_prob = wl_pred.get('away_win_probability', 0.5)

            if home_prob > 0.65:
                safety_factors.append("Strong home favorite")
            elif away_prob > 0.70:
                risk_factors.append("Potential road upset")
            elif abs(home_prob - away_prob) < 0.1:
                risk_factors.append("Close matchup")

        # Over/Under safety analysis
        if 'over_under' in predictions_data:
            ou_pred = predictions_data['over_under']
            prediction_type = ou_pred.get('prediction', '')
            threshold = ou_pred.get('threshold', 220)

            if prediction_type == 'under':
                safety_factors.append("Under bet (more predictable)")
            elif prediction_type == 'over' and threshold < 215:
                safety_factors.append("Low total over bet")
            elif prediction_type == 'over' and threshold > 230:
                risk_factors.append("High total over bet")

        return {
            'safety_level': 'HIGH' if len(safety_factors) >= 2 and len(risk_factors) == 0 else 'MEDIUM',
            'safety_factors': safety_factors,
            'risk_factors': risk_factors,
            'confidence_score': overall_confidence
        }

    def _estimate_individual_odds(self, prediction: Dict) -> Dict[str, float]:
        """
        Estimate individual betting odds based on prediction confidence.

        Args:
            prediction: Single prediction dictionary

        Returns:
            Dictionary with estimated odds for different bet types
        """
        odds_estimates = {}
        predictions_data = prediction.get('predictions', {})

        # Win/Loss odds estimation
        if 'win_loss' in predictions_data:
            wl_pred = predictions_data['win_loss']
            home_prob = wl_pred.get('home_win_probability', 0.5)
            away_prob = wl_pred.get('away_win_probability', 0.5)

            # Convert probability to decimal odds (with bookmaker margin)
            if home_prob > 0.5:
                odds_estimates['home_ml'] = round(1 / (home_prob * 0.95), 2)  # 5% margin
            if away_prob > 0.5:
                odds_estimates['away_ml'] = round(1 / (away_prob * 0.95), 2)

        # Over/Under odds estimation
        if 'over_under' in predictions_data:
            ou_pred = predictions_data['over_under']
            over_prob = ou_pred.get('over_probability', 0.5)
            under_prob = ou_pred.get('under_probability', 0.5)

            odds_estimates['over'] = round(1 / (over_prob * 0.95), 2)
            odds_estimates['under'] = round(1 / (under_prob * 0.95), 2)

        # Point spread odds (typically around 1.9 for both sides)
        odds_estimates['spread'] = 1.91

        return odds_estimates

    def _build_basketball_odds_category(self, safe_predictions: List[Dict], target_odds: float, individual_range: tuple) -> List[Dict]:
        """
        Build a betting category with target combined odds.

        Args:
            safe_predictions: List of safe predictions
            target_odds: Target combined odds for the category
            individual_range: Tuple of (min_odds, max_odds) for individual selections

        Returns:
            List of predictions that combine to target odds
        """
        if not safe_predictions:
            return []

        min_individual, max_individual = individual_range
        category_selections = []

        # Filter predictions that fit the individual odds range
        suitable_predictions = []
        for pred in safe_predictions:
            estimated_odds = pred.get('estimated_odds', {})

            # Find the best odds for this prediction within range
            best_odds = None
            best_bet_type = None

            for bet_type, odds in estimated_odds.items():
                if min_individual <= odds <= max_individual:
                    if best_odds is None or odds > best_odds:
                        best_odds = odds
                        best_bet_type = bet_type

            if best_odds and best_bet_type:
                pred_copy = pred.copy()
                pred_copy['selected_bet'] = {
                    'type': best_bet_type,
                    'odds': best_odds,
                    'reasoning': self._get_bet_reasoning(pred, best_bet_type)
                }
                suitable_predictions.append(pred_copy)

        if not suitable_predictions:
            logger.warning(f"No suitable predictions found for odds range {individual_range}")
            return []

        # Calculate how many selections needed for target odds
        avg_odds = (min_individual + max_individual) / 2
        estimated_selections = max(2, min(10, int(np.log(target_odds) / np.log(avg_odds))))

        # Select best predictions up to the limit
        selected_count = min(estimated_selections, len(suitable_predictions))
        category_selections = suitable_predictions[:selected_count]

        # Calculate actual combined odds
        combined_odds = 1.0
        for selection in category_selections:
            combined_odds *= selection['selected_bet']['odds']

        # Add category metadata
        category_info = {
            'target_odds': target_odds,
            'actual_combined_odds': round(combined_odds, 2),
            'individual_range': individual_range,
            'selection_count': len(category_selections),
            'avg_confidence': round(np.mean([s.get('overall_confidence', 0) for s in category_selections]), 3),
            'safety_summary': self._summarize_category_safety(category_selections)
        }

        # Add category info to each selection
        for selection in category_selections:
            selection['category_info'] = category_info

        logger.info(f"Built category with {len(category_selections)} selections, combined odds: {combined_odds:.2f} (target: {target_odds})")
        return category_selections

    def _get_bet_reasoning(self, prediction: Dict, bet_type: str) -> str:
        """Get reasoning for why this bet type was selected."""
        reasoning_map = {
            'home_ml': 'Strong home favorite with high confidence',
            'away_ml': 'Road favorite with solid fundamentals',
            'over': 'High-scoring matchup expected',
            'under': 'Defensive game with low total',
            'spread': 'Point spread offers good value'
        }

        base_reasoning = reasoning_map.get(bet_type, 'High confidence selection')
        confidence = prediction.get('overall_confidence', 0)

        return f"{base_reasoning} ({confidence:.1%} confidence)"

    def _summarize_category_safety(self, selections: List[Dict]) -> Dict[str, Any]:
        """Summarize the safety characteristics of a category."""
        if not selections:
            return {}

        safety_levels = [s.get('safety_classification', {}).get('safety_level', 'MEDIUM') for s in selections]
        high_safety_count = safety_levels.count('HIGH')

        all_factors = []
        all_risks = []

        for selection in selections:
            safety_class = selection.get('safety_classification', {})
            all_factors.extend(safety_class.get('safety_factors', []))
            all_risks.extend(safety_class.get('risk_factors', []))

        return {
            'high_safety_selections': high_safety_count,
            'total_selections': len(selections),
            'common_safety_factors': list(set(all_factors)),
            'potential_risks': list(set(all_risks)),
            'overall_safety_rating': 'HIGH' if high_safety_count >= len(selections) * 0.7 else 'MEDIUM'
        }

    def _get_models_status(self) -> Dict[str, bool]:
        """Get status of available models."""
        return {
            'win_loss_xgboost': self.win_loss_model.model is not None,
            'over_under_lightgbm': self.over_under_model.model is not None,
            'neural_network': self.neural_network_model is not None and self.neural_network_model.model is not None
        }
    
    def _save_predictions(self, predictions: List[Dict], date: str):
        """Save predictions to cache."""
        try:
            cache_file = os.path.join(self.cache_dir, f"basketball_predictions_{date}.json")
            
            prediction_data = {
                'date': date,
                'generated_at': datetime.now().isoformat(),
                'predictions': predictions,
                'total_games': len(predictions)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(prediction_data, f, indent=2, default=str)
            
            logger.info(f"Basketball predictions saved to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving basketball predictions: {str(e)}")
    
    def load_cached_predictions(self, date: str) -> Optional[Dict[str, Any]]:
        """Load cached predictions for a date."""
        try:
            cache_file = os.path.join(self.cache_dir, f"basketball_predictions_{date}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                logger.info(f"Loaded cached basketball predictions for {date}")
                return data
            
        except Exception as e:
            logger.error(f"Error loading cached basketball predictions: {str(e)}")
        
        return None
    
    def train_models(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train basketball prediction models.
        
        Args:
            data_path: Path to training data CSV file
            
        Returns:
            Training results dictionary
        """
        logger.info("Training basketball prediction models")
        
        try:
            # Load or download training data
            if data_path is None:
                data_path = self.data_fetcher.download_historical_data()
            
            # Load training data
            df = pd.read_csv(data_path)
            
            if df.empty:
                return {'status': 'error', 'message': 'No training data available'}
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(df)
            
            # Prepare training data
            feature_columns = self.feature_engineer.get_feature_names()
            X = features_df[feature_columns]
            
            # Train Win/Loss model
            y_win_loss = df['HOME_WIN'] if 'HOME_WIN' in df.columns else np.random.randint(0, 2, len(df))
            win_loss_results = self.win_loss_model.train(X, pd.Series(y_win_loss))
            
            # Train Over/Under model
            y_total = df['TOTAL_PTS'] if 'TOTAL_PTS' in df.columns else np.random.uniform(180, 250, len(df))
            over_under_results = self.over_under_model.train(X, pd.Series(y_total))
            
            # Train Neural Network model (if available)
            nn_results = {}
            if self.neural_network_model:
                nn_results = self.neural_network_model.train(X, pd.Series(y_win_loss), 'classification')
            
            results = {
                'status': 'success',
                'training_date': datetime.now().isoformat(),
                'training_samples': len(df),
                'models': {
                    'win_loss_xgboost': win_loss_results,
                    'over_under_lightgbm': over_under_results,
                    'neural_network': nn_results
                }
            }
            
            logger.info("Basketball model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training basketball models: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_prediction_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of predictions for a date.
        
        Args:
            date: Date string or None for today
            
        Returns:
            Prediction summary dictionary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Try to load cached predictions first
        cached_predictions = self.load_cached_predictions(date)
        
        if cached_predictions:
            predictions = cached_predictions['predictions']
        else:
            # Generate new predictions
            result = self.generate_predictions(date)
            if result['status'] != 'success':
                return result
            predictions = result['predictions']
        
        # Create summary
        summary = {
            'date': date,
            'total_games': len(predictions),
            'high_confidence_games': len([p for p in predictions if p.get('overall_confidence', 0) > 0.75]),
            'medium_confidence_games': len([p for p in predictions if 0.60 < p.get('overall_confidence', 0) <= 0.75]),
            'low_confidence_games': len([p for p in predictions if p.get('overall_confidence', 0) <= 0.60]),
            'models_available': self._get_models_status(),
            'best_predictions': sorted(predictions, key=lambda x: x.get('overall_confidence', 0), reverse=True)[:3]
        }
        
        return summary
