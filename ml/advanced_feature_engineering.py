"""
Advanced Feature Engineering Module

This module provides enhanced feature engineering for football match prediction.
It includes more sophisticated features and preprocessing techniques to improve model accuracy.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

from utils.common import setup_logging, ensure_directory_exists
from utils.config import settings

# Set up logging
logger = setup_logging(__name__)

class AdvancedFootballFeatureEngineer:
    """
    Advanced feature engineering for football match prediction.

    Features:
    - Team form with exponential decay (recent matches weighted more heavily)
    - Advanced attack and defense metrics (xG, xGA, shot conversion)
    - Detailed head-to-head analysis with recency weighting
    - Player-based features (injuries, suspensions, key player availability)
    - Tactical matchup analysis
    - Weather and pitch conditions
    - Travel distance and fatigue metrics
    - Momentum and streak indicators
    - Betting market signals
    - Seasonal patterns and trends
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.historical_data = None
        self.player_data = None
        self.weather_data = None
        self.betting_data = None

        # Cache for feature calculations
        self.cache = {}
        self.cache_expiry = {}
        self.cache_dir = settings.ml.CACHE_DIR
        ensure_directory_exists(self.cache_dir)
        self.cache_file = os.path.join(self.cache_dir, "advanced_feature_cache.joblib")
        self.cache_expiry_hours = settings.ml.FEATURE_CACHE_EXPIRY

        # Feature scaling
        self.scaler = None
        self.scaler_file = os.path.join(self.cache_dir, "advanced_feature_scaler.joblib")

        # Load cache if it exists
        self._load_cache()
        self._load_scaler()

    def set_historical_data(self, historical_data: pd.DataFrame):
        """
        Set the historical match data.

        Args:
            historical_data: DataFrame with historical match data
        """
        self.historical_data = historical_data

    def set_player_data(self, player_data: pd.DataFrame):
        """
        Set player data including injuries, suspensions, etc.

        Args:
            player_data: DataFrame with player data
        """
        self.player_data = player_data

    def set_weather_data(self, weather_data: pd.DataFrame):
        """
        Set weather data for matches.

        Args:
            weather_data: DataFrame with weather data
        """
        self.weather_data = weather_data

    def set_betting_data(self, betting_data: pd.DataFrame):
        """
        Set betting market data.

        Args:
            betting_data: DataFrame with betting market data
        """
        self.betting_data = betting_data

    def engineer_features(self, fixture: Dict[str, Any]) -> pd.DataFrame:
        """
        Engineer features for a fixture.

        Args:
            fixture: Fixture data

        Returns:
            DataFrame with engineered features
        """
        try:
            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                logger.error("Historical data not set or empty")
                return pd.DataFrame()

            # Extract fixture data
            fixture_id = fixture.get("fixture", {}).get("id", "unknown")
            home_team = fixture.get("teams", {}).get("home", {}).get("name", "unknown")
            away_team = fixture.get("teams", {}).get("away", {}).get("name", "unknown")
            match_date_str = fixture.get("fixture", {}).get("date", datetime.now().isoformat())

            # Parse match date
            try:
                match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                match_date = datetime.now()

            # Check if features are in cache
            cache_key = f"{fixture_id}_{home_team}_{away_team}"
            cached_features = self._get_from_cache(cache_key)

            if cached_features is not None:
                return cached_features

            # Create features DataFrame
            features = pd.DataFrame({
                "fixture_id": [fixture_id],
                "home_team": [home_team],
                "away_team": [away_team],
                "match_date": [match_date]
            })

            # Add team form features with exponential decay
            features = self._add_team_form_features(features, home_team, away_team, match_date)

            # Add advanced attack and defense metrics
            features = self._add_advanced_team_metrics(features, home_team, away_team, match_date)

            # Add detailed head-to-head analysis
            features = self._add_detailed_h2h_features(features, home_team, away_team, match_date)

            # Add league position and momentum features
            features = self._add_league_position_features(features, home_team, away_team, match_date)

            # Add player-based features if player data is available
            if self.player_data is not None:
                features = self._add_player_features(features, home_team, away_team, match_date)

            # Add weather features if weather data is available
            if self.weather_data is not None:
                features = self._add_weather_features(features, fixture_id, match_date)

            # Add travel and fatigue metrics
            features = self._add_travel_fatigue_features(features, home_team, away_team, match_date)

            # Add momentum and streak indicators
            features = self._add_momentum_streak_features(features, home_team, away_team, match_date)

            # Add betting market signals if betting data is available
            if self.betting_data is not None:
                features = self._add_betting_market_features(features, fixture_id, home_team, away_team)

            # Add seasonal patterns and trends
            features = self._add_seasonal_pattern_features(features, match_date)

            # Add tactical matchup analysis
            features = self._add_tactical_matchup_features(features, home_team, away_team)

            # Drop non-feature columns
            feature_df = features.drop(columns=["fixture_id", "home_team", "away_team", "match_date"], errors="ignore")

            # Fill missing values with appropriate defaults
            feature_df = self._fill_missing_values(feature_df)

            # Scale features if scaler is available
            if self.scaler is not None:
                feature_df = pd.DataFrame(
                    self.scaler.transform(feature_df),
                    columns=feature_df.columns
                )

            # Add to cache
            self._add_to_cache(cache_key, feature_df)

            return feature_df

        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return pd.DataFrame()

    def fit_scaler(self, features_df: pd.DataFrame):
        """
        Fit the feature scaler.

        Args:
            features_df: DataFrame with features
        """
        try:
            # Create a new scaler
            self.scaler = StandardScaler()

            # Fit the scaler
            self.scaler.fit(features_df)

            # Save the scaler
            joblib.dump(self.scaler, self.scaler_file)

            logger.info(f"Feature scaler fitted and saved to {self.scaler_file}")

        except Exception as e:
            logger.error(f"Error fitting feature scaler: {str(e)}")

    def _load_scaler(self):
        """Load the feature scaler from disk."""
        try:
            if os.path.exists(self.scaler_file):
                self.scaler = joblib.load(self.scaler_file)
                logger.info(f"Feature scaler loaded from {self.scaler_file}")

        except Exception as e:
            logger.error(f"Error loading feature scaler: {str(e)}")
            self.scaler = None

    def _load_cache(self):
        """Load the feature cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                cache_data = joblib.load(self.cache_file)
                self.cache = cache_data.get("cache", {})
                self.cache_expiry = cache_data.get("expiry", {})
                logger.info(f"Feature cache loaded from {self.cache_file}")

        except Exception as e:
            logger.error(f"Error loading feature cache: {str(e)}")
            self.cache = {}
            self.cache_expiry = {}

    def _save_cache(self):
        """Save the feature cache to disk."""
        try:
            cache_data = {
                "cache": self.cache,
                "expiry": self.cache_expiry,
                "updated_at": datetime.now().isoformat()
            }
            joblib.dump(cache_data, self.cache_file)
            logger.info(f"Feature cache saved to {self.cache_file}")

        except Exception as e:
            logger.error(f"Error saving feature cache: {str(e)}")

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get features from cache.

        Args:
            key: Cache key

        Returns:
            DataFrame with features or None if not in cache or expired
        """
        try:
            if key in self.cache and key in self.cache_expiry:
                expiry_time = datetime.fromisoformat(self.cache_expiry[key])

                if datetime.now() < expiry_time:
                    logger.info(f"Features found in cache for key: {key}")
                    return self.cache[key]
                else:
                    logger.info(f"Cached features expired for key: {key}")
                    # Remove expired entry
                    del self.cache[key]
                    del self.cache_expiry[key]

            return None

        except Exception as e:
            logger.error(f"Error getting features from cache: {str(e)}")
            return None

    def _add_to_cache(self, key: str, features: pd.DataFrame):
        """
        Add features to cache.

        Args:
            key: Cache key
            features: DataFrame with features
        """
        try:
            self.cache[key] = features
            self.cache_expiry[key] = (datetime.now() + timedelta(hours=self.cache_expiry_hours)).isoformat()

            # Save cache periodically (every 10 additions)
            if len(self.cache) % 10 == 0:
                self._save_cache()

        except Exception as e:
            logger.error(f"Error adding features to cache: {str(e)}")

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with appropriate defaults.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with missing values filled
        """
        # Create a copy to avoid modifying the original
        filled_df = df.copy()

        # Fill missing values with appropriate defaults based on column type
        for col in filled_df.columns:
            # If column contains percentages or rates (0-1 range)
            if any(keyword in col.lower() for keyword in ['rate', 'percentage', 'pct', 'ratio', 'probability']):
                filled_df[col] = filled_df[col].fillna(0.5)

            # If column contains counts
            elif any(keyword in col.lower() for keyword in ['count', 'num', 'number']):
                filled_df[col] = filled_df[col].fillna(0)

            # If column contains differences
            elif 'diff' in col.lower():
                filled_df[col] = filled_df[col].fillna(0)

            # If column contains strengths or scores
            elif any(keyword in col.lower() for keyword in ['strength', 'score', 'rating']):
                filled_df[col] = filled_df[col].fillna(0.5)

            # Default case
            else:
                filled_df[col] = filled_df[col].fillna(0)

        return filled_df

    def _add_team_form_features(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add team form features with exponential decay (recent matches weighted more heavily).

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with team form features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Get historical matches for both teams
        home_matches = self._get_team_matches(home_team, match_date, max_days=365, max_matches=20)
        away_matches = self._get_team_matches(away_team, match_date, max_days=365, max_matches=20)

        # Calculate form features with exponential decay
        home_form_features = self._calculate_team_form_with_decay(home_team, home_matches, match_date)
        away_form_features = self._calculate_team_form_with_decay(away_team, away_matches, match_date)

        # Add features to DataFrame
        for feature, value in home_form_features.items():
            result_df[f"home_{feature}"] = value

        for feature, value in away_form_features.items():
            result_df[f"away_{feature}"] = value

        # Add form differential features
        for feature in home_form_features.keys():
            if feature in away_form_features:
                result_df[f"{feature}_diff"] = home_form_features[feature] - away_form_features[feature]

        return result_df

    def _get_team_matches(self, team: str, before_date: datetime, max_days: int = 365, max_matches: int = 20) -> pd.DataFrame:
        """
        Get historical matches for a team before a specific date.

        Args:
            team: Team name
            before_date: Only include matches before this date
            max_days: Maximum number of days to look back
            max_matches: Maximum number of matches to return

        Returns:
            DataFrame with team matches
        """
        try:
            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                return pd.DataFrame()

            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])

            # Calculate earliest date to consider
            earliest_date = before_date - timedelta(days=max_days)

            # Filter matches
            team_matches = self.historical_data[
                (
                    ((self.historical_data['home_team'] == team) |
                     (self.historical_data['away_team'] == team)) &
                    (self.historical_data['date'] < before_date) &
                    (self.historical_data['date'] >= earliest_date)
                )
            ].sort_values('date', ascending=False).head(max_matches)

            return team_matches

        except Exception as e:
            logger.error(f"Error getting team matches: {str(e)}")
            return pd.DataFrame()

    def _calculate_team_form_with_decay(self, team: str, matches: pd.DataFrame, match_date: datetime) -> Dict[str, float]:
        """
        Calculate team form with exponential decay (recent matches weighted more heavily).

        Args:
            team: Team name
            matches: DataFrame with team matches
            match_date: Match date for reference

        Returns:
            Dictionary with form features
        """
        try:
            # Initialize form features
            form_features = {
                'recent_win_rate': 0.5,
                'recent_draw_rate': 0.25,
                'recent_loss_rate': 0.25,
                'recent_goals_scored_per_game': 1.25,
                'recent_goals_conceded_per_game': 1.25,
                'recent_clean_sheet_rate': 0.3,
                'recent_failed_to_score_rate': 0.3,
                'recent_points_per_game': 1.5,
                'recent_xg_per_game': 1.25,
                'recent_xga_per_game': 1.25,
                'recent_shot_conversion_rate': 0.1,
                'recent_shot_on_target_rate': 0.3,
                'recent_possession_avg': 50.0,
                'recent_home_win_rate': 0.6,
                'recent_away_win_rate': 0.4,
                'form_momentum': 0.0,
                'days_since_last_match': 7.0
            }

            # If no matches, return default values
            if matches.empty:
                return form_features

            # Initialize counters with decay weights
            total_weight = 0
            weighted_wins = 0
            weighted_draws = 0
            weighted_losses = 0
            weighted_goals_scored = 0
            weighted_goals_conceded = 0
            weighted_clean_sheets = 0
            weighted_failed_to_score = 0
            weighted_points = 0
            weighted_xg = 0
            weighted_xga = 0
            weighted_shots = 0
            weighted_shots_on_target = 0
            weighted_possession = 0
            weighted_home_wins = 0
            weighted_home_matches = 0
            weighted_away_wins = 0
            weighted_away_matches = 0

            # Calculate days since last match
            if not matches.empty:
                last_match_date = matches.iloc[0]['date']
                days_since_last_match = (match_date - last_match_date).days
                form_features['days_since_last_match'] = max(1, days_since_last_match)

            # Calculate form momentum (trend in recent results)
            form_momentum = 0

            # Process each match with decay
            for i, match in matches.iterrows():
                # Calculate days since match
                days_since_match = (match_date - match['date']).days

                # Apply exponential decay (half-life of 30 days)
                decay_weight = np.exp(-0.023 * days_since_match)  # ln(2)/30 ≈ 0.023
                total_weight += decay_weight

                # Determine if team was home or away
                is_home = match['home_team'] == team

                # Get team and opponent goals
                team_goals = match['home_score'] if is_home else match['away_score']
                opponent_goals = match['away_score'] if is_home else match['home_score']

                # Update weighted counters
                if team_goals > opponent_goals:
                    weighted_wins += decay_weight
                    weighted_points += 3 * decay_weight
                    # For momentum: more recent wins have higher impact
                    form_momentum += decay_weight * 1.0
                elif team_goals == opponent_goals:
                    weighted_draws += decay_weight
                    weighted_points += 1 * decay_weight
                    # Draws have neutral impact on momentum
                else:
                    weighted_losses += decay_weight
                    # Losses have negative impact on momentum
                    form_momentum -= decay_weight * 1.0

                weighted_goals_scored += team_goals * decay_weight
                weighted_goals_conceded += opponent_goals * decay_weight

                if opponent_goals == 0:
                    weighted_clean_sheets += decay_weight

                if team_goals == 0:
                    weighted_failed_to_score += decay_weight

                # Track home/away performance
                if is_home:
                    weighted_home_matches += decay_weight
                    if team_goals > opponent_goals:
                        weighted_home_wins += decay_weight
                else:
                    weighted_away_matches += decay_weight
                    if team_goals > opponent_goals:
                        weighted_away_wins += decay_weight

                # Add expected goals if available
                if 'home_xg' in match and 'away_xg' in match:
                    team_xg = match['home_xg'] if is_home else match['away_xg']
                    opponent_xg = match['away_xg'] if is_home else match['home_xg']
                    weighted_xg += team_xg * decay_weight
                    weighted_xga += opponent_xg * decay_weight

                # Add shots if available
                if 'home_shots' in match and 'away_shots' in match:
                    team_shots = match['home_shots'] if is_home else match['away_shots']
                    weighted_shots += team_shots * decay_weight

                # Add shots on target if available
                if 'home_shots_on_target' in match and 'away_shots_on_target' in match:
                    team_shots_on_target = match['home_shots_on_target'] if is_home else match['away_shots_on_target']
                    weighted_shots_on_target += team_shots_on_target * decay_weight

                # Add possession if available
                if 'home_possession' in match and 'away_possession' in match:
                    team_possession = match['home_possession'] if is_home else match['away_possession']
                    weighted_possession += team_possession * decay_weight

            # Normalize by total weight
            if total_weight > 0:
                form_features['recent_win_rate'] = weighted_wins / total_weight
                form_features['recent_draw_rate'] = weighted_draws / total_weight
                form_features['recent_loss_rate'] = weighted_losses / total_weight
                form_features['recent_goals_scored_per_game'] = weighted_goals_scored / total_weight
                form_features['recent_goals_conceded_per_game'] = weighted_goals_conceded / total_weight
                form_features['recent_clean_sheet_rate'] = weighted_clean_sheets / total_weight
                form_features['recent_failed_to_score_rate'] = weighted_failed_to_score / total_weight
                form_features['recent_points_per_game'] = weighted_points / total_weight

                # Add home/away specific rates
                if weighted_home_matches > 0:
                    form_features['recent_home_win_rate'] = weighted_home_wins / weighted_home_matches

                if weighted_away_matches > 0:
                    form_features['recent_away_win_rate'] = weighted_away_wins / weighted_away_matches

                # Add expected goals if available
                if weighted_xg > 0:
                    form_features['recent_xg_per_game'] = weighted_xg / total_weight
                    form_features['recent_xga_per_game'] = weighted_xga / total_weight

                # Add shot conversion if available
                if weighted_shots > 0:
                    form_features['recent_shot_conversion_rate'] = weighted_goals_scored / weighted_shots

                    if weighted_shots_on_target > 0:
                        form_features['recent_shot_on_target_rate'] = weighted_shots_on_target / weighted_shots

                # Add possession if available
                if weighted_possession > 0:
                    form_features['recent_possession_avg'] = weighted_possession / total_weight

                # Normalize momentum to -1 to 1 range
                form_features['form_momentum'] = np.tanh(form_momentum / total_weight)

            return form_features

        except Exception as e:
            logger.error(f"Error calculating team form with decay: {str(e)}")
            return {
                'recent_win_rate': 0.5,
                'recent_draw_rate': 0.25,
                'recent_loss_rate': 0.25,
                'recent_goals_scored_per_game': 1.25,
                'recent_goals_conceded_per_game': 1.25,
                'recent_clean_sheet_rate': 0.3,
                'recent_failed_to_score_rate': 0.3,
                'recent_points_per_game': 1.5,
                'form_momentum': 0.0,
                'days_since_last_match': 7.0
            }

    def _add_advanced_team_metrics(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add advanced attack and defense metrics.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with advanced team metrics
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Get historical matches for both teams
        home_matches = self._get_team_matches(home_team, match_date, max_days=365, max_matches=30)
        away_matches = self._get_team_matches(away_team, match_date, max_days=365, max_matches=30)

        # Calculate advanced metrics
        home_metrics = self._calculate_advanced_metrics(home_team, home_matches)
        away_metrics = self._calculate_advanced_metrics(away_team, away_matches)

        # Add metrics to DataFrame
        for metric, value in home_metrics.items():
            result_df[f"home_{metric}"] = value

        for metric, value in away_metrics.items():
            result_df[f"away_{metric}"] = value

        # Add differential metrics
        for metric in home_metrics.keys():
            if metric in away_metrics:
                result_df[f"{metric}_diff"] = home_metrics[metric] - away_metrics[metric]

        return result_df

    def _calculate_advanced_metrics(self, team: str, matches: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate advanced team metrics.

        Args:
            team: Team name
            matches: DataFrame with team matches

        Returns:
            Dictionary with advanced metrics
        """
        try:
            # Initialize metrics
            metrics = {
                'attack_strength': 0.5,
                'defense_strength': 0.5,
                'home_advantage': 0.5,
                'away_disadvantage': 0.5,
                'xg_overperformance': 0.0,
                'xga_overperformance': 0.0,
                'shot_quality': 0.5,
                'defensive_pressure': 0.5,
                'set_piece_threat': 0.5,
                'counter_attack_threat': 0.5,
                'possession_effectiveness': 0.5,
                'defensive_organization': 0.5,
                'comeback_ability': 0.5,
                'game_management': 0.5
            }

            # If no matches, return default values
            if matches.empty:
                return metrics

            # Calculate basic stats
            total_matches = len(matches)
            home_matches = matches[matches['home_team'] == team]
            away_matches = matches[matches['away_team'] == team]

            total_goals_scored = 0
            total_goals_conceded = 0
            total_xg = 0
            total_xga = 0
            total_shots = 0
            total_shots_on_target = 0
            total_corners = 0
            total_possession = 0
            comebacks = 0
            leads_lost = 0

            # Process each match
            for _, match in matches.iterrows():
                is_home = match['home_team'] == team

                # Get team and opponent goals
                team_goals = match['home_score'] if is_home else match['away_score']
                opponent_goals = match['away_score'] if is_home else match['home_score']

                total_goals_scored += team_goals
                total_goals_conceded += opponent_goals

                # Add expected goals if available
                if 'home_xg' in match and 'away_xg' in match:
                    team_xg = match['home_xg'] if is_home else match['away_xg']
                    opponent_xg = match['away_xg'] if is_home else match['home_xg']
                    total_xg += team_xg
                    total_xga += opponent_xg

                # Add shots if available
                if 'home_shots' in match and 'away_shots' in match:
                    team_shots = match['home_shots'] if is_home else match['away_shots']
                    total_shots += team_shots

                # Add shots on target if available
                if 'home_shots_on_target' in match and 'away_shots_on_target' in match:
                    team_shots_on_target = match['home_shots_on_target'] if is_home else match['away_shots_on_target']
                    total_shots_on_target += team_shots_on_target

                # Add corners if available
                if 'home_corners' in match and 'away_corners' in match:
                    team_corners = match['home_corners'] if is_home else match['away_corners']
                    total_corners += team_corners

                # Add possession if available
                if 'home_possession' in match and 'away_possession' in match:
                    team_possession = match['home_possession'] if is_home else match['away_possession']
                    total_possession += team_possession

                # Check for comebacks and leads lost if half-time score is available
                if 'home_half_time_score' in match and 'away_half_time_score' in match:
                    ht_team_goals = match['home_half_time_score'] if is_home else match['away_half_time_score']
                    ht_opponent_goals = match['away_half_time_score'] if is_home else match['home_half_time_score']

                    # Comeback: behind at HT, won or drew
                    if ht_team_goals < ht_opponent_goals and team_goals >= opponent_goals:
                        comebacks += 1

                    # Lead lost: ahead at HT, lost or drew
                    if ht_team_goals > ht_opponent_goals and team_goals <= opponent_goals:
                        leads_lost += 1

            # Calculate metrics
            if total_matches > 0:
                # Attack and defense strength
                avg_goals_scored = total_goals_scored / total_matches
                avg_goals_conceded = total_goals_conceded / total_matches

                # League average goals (approximate)
                league_avg_goals = 1.3

                metrics['attack_strength'] = avg_goals_scored / league_avg_goals
                metrics['defense_strength'] = league_avg_goals / max(0.5, avg_goals_conceded)

                # Home/away performance
                if len(home_matches) > 0:
                    home_goals_scored = sum(home_matches['home_score'])
                    home_goals_conceded = sum(home_matches['away_score'])
                    home_avg_scored = home_goals_scored / len(home_matches)
                    metrics['home_advantage'] = home_avg_scored / max(0.5, avg_goals_scored)

                if len(away_matches) > 0:
                    away_goals_scored = sum(away_matches['away_score'])
                    away_goals_conceded = sum(away_matches['home_score'])
                    away_avg_scored = away_goals_scored / len(away_matches)
                    metrics['away_disadvantage'] = away_avg_scored / max(0.5, avg_goals_scored)

                # Expected goals overperformance
                if total_xg > 0 and total_xga > 0:
                    metrics['xg_overperformance'] = total_goals_scored / total_xg - 1.0
                    metrics['xga_overperformance'] = 1.0 - total_goals_conceded / total_xga

                # Shot quality and efficiency
                if total_shots > 0:
                    metrics['shot_quality'] = total_goals_scored / total_shots

                    if total_shots_on_target > 0:
                        shot_accuracy = total_shots_on_target / total_shots
                        shot_conversion = total_goals_scored / total_shots_on_target
                        metrics['shot_quality'] = (shot_accuracy + shot_conversion) / 2

                # Set piece threat
                if total_corners > 0:
                    metrics['set_piece_threat'] = min(1.0, total_corners / (total_matches * 5))

                # Possession effectiveness
                if total_possession > 0:
                    avg_possession = total_possession / total_matches
                    possession_goal_ratio = total_goals_scored / max(1, total_matches * avg_possession / 100)
                    metrics['possession_effectiveness'] = min(1.0, possession_goal_ratio)

                # Comeback ability and game management
                if total_matches > 0:
                    metrics['comeback_ability'] = min(1.0, comebacks / (total_matches * 0.2))
                    metrics['game_management'] = 1.0 - min(1.0, leads_lost / (total_matches * 0.2))

            return metrics

        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            return {
                'attack_strength': 0.5,
                'defense_strength': 0.5,
                'home_advantage': 0.5,
                'away_disadvantage': 0.5,
                'xg_overperformance': 0.0,
                'xga_overperformance': 0.0,
                'shot_quality': 0.5,
                'defensive_pressure': 0.5,
                'set_piece_threat': 0.5,
                'counter_attack_threat': 0.5,
                'possession_effectiveness': 0.5,
                'defensive_organization': 0.5,
                'comeback_ability': 0.5,
                'game_management': 0.5
            }

    def _add_detailed_h2h_features(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add detailed head-to-head analysis with recency weighting.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with head-to-head features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        try:
            # Get head-to-head matches
            h2h_matches = self._get_h2h_matches(home_team, away_team, match_date)

            # Calculate head-to-head features
            h2h_features = self._calculate_h2h_features(home_team, away_team, h2h_matches, match_date)

            # Add features to DataFrame
            for feature, value in h2h_features.items():
                result_df[feature] = value

            return result_df

        except Exception as e:
            logger.error(f"Error adding head-to-head features: {str(e)}")

            # Add default values
            result_df['h2h_home_win_rate'] = 0.4
            result_df['h2h_away_win_rate'] = 0.3
            result_df['h2h_draw_rate'] = 0.3
            result_df['h2h_home_goals_avg'] = 1.5
            result_df['h2h_away_goals_avg'] = 1.2
            result_df['h2h_total_goals_avg'] = 2.7
            result_df['h2h_btts_rate'] = 0.5
            result_df['h2h_home_dominance'] = 0.0
            result_df['h2h_matches_count'] = 0

            return result_df

    def _get_h2h_matches(self, home_team: str, away_team: str, before_date: datetime, max_matches: int = 10) -> pd.DataFrame:
        """
        Get head-to-head matches between two teams.

        Args:
            home_team: Home team name
            away_team: Away team name
            before_date: Only include matches before this date
            max_matches: Maximum number of matches to return

        Returns:
            DataFrame with head-to-head matches
        """
        try:
            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                return pd.DataFrame()

            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])

            # Filter head-to-head matches
            h2h_matches = self.historical_data[
                (
                    (
                        (self.historical_data['home_team'] == home_team) &
                        (self.historical_data['away_team'] == away_team)
                    ) |
                    (
                        (self.historical_data['home_team'] == away_team) &
                        (self.historical_data['away_team'] == home_team)
                    )
                ) &
                (self.historical_data['date'] < before_date)
            ].sort_values('date', ascending=False).head(max_matches)

            return h2h_matches

        except Exception as e:
            logger.error(f"Error getting head-to-head matches: {str(e)}")
            return pd.DataFrame()

    def _calculate_h2h_features(self, home_team: str, away_team: str, h2h_matches: pd.DataFrame, match_date: datetime) -> Dict[str, float]:
        """
        Calculate head-to-head features with recency weighting.

        Args:
            home_team: Home team name
            away_team: Away team name
            h2h_matches: DataFrame with head-to-head matches
            match_date: Match date

        Returns:
            Dictionary with head-to-head features
        """
        try:
            # Initialize features
            h2h_features = {
                'h2h_home_win_rate': 0.4,
                'h2h_away_win_rate': 0.3,
                'h2h_draw_rate': 0.3,
                'h2h_home_goals_avg': 1.5,
                'h2h_away_goals_avg': 1.2,
                'h2h_total_goals_avg': 2.7,
                'h2h_btts_rate': 0.5,
                'h2h_home_dominance': 0.0,
                'h2h_matches_count': 0
            }

            # If no matches, return default values
            if h2h_matches.empty:
                return h2h_features

            # Set matches count
            h2h_features['h2h_matches_count'] = len(h2h_matches)

            # Initialize counters with decay weights
            total_weight = 0
            weighted_home_wins = 0
            weighted_away_wins = 0
            weighted_draws = 0
            weighted_home_goals = 0
            weighted_away_goals = 0
            weighted_btts = 0
            weighted_home_dominance = 0

            # Process each match with decay
            for _, match in h2h_matches.iterrows():
                # Calculate days since match
                days_since_match = (match_date - match['date']).days

                # Apply exponential decay (half-life of 365 days for H2H)
                decay_weight = np.exp(-0.0019 * days_since_match)  # ln(2)/365 ≈ 0.0019
                total_weight += decay_weight

                # Check if current home team was home in this match
                current_home_was_home = match['home_team'] == home_team

                # Get goals
                if current_home_was_home:
                    home_goals = match['home_score']
                    away_goals = match['away_score']
                else:
                    home_goals = match['away_score']
                    away_goals = match['home_score']

                # Update weighted counters
                if current_home_was_home:
                    if home_goals > away_goals:
                        weighted_home_wins += decay_weight
                    elif home_goals < away_goals:
                        weighted_away_wins += decay_weight
                    else:
                        weighted_draws += decay_weight
                else:
                    if home_goals > away_goals:
                        weighted_away_wins += decay_weight
                    elif home_goals < away_goals:
                        weighted_home_wins += decay_weight
                    else:
                        weighted_draws += decay_weight

                weighted_home_goals += home_goals * decay_weight
                weighted_away_goals += away_goals * decay_weight

                # Both teams to score
                if home_goals > 0 and away_goals > 0:
                    weighted_btts += decay_weight

                # Home dominance (goal difference)
                weighted_home_dominance += (home_goals - away_goals) * decay_weight

            # Normalize by total weight
            if total_weight > 0:
                h2h_features['h2h_home_win_rate'] = weighted_home_wins / total_weight
                h2h_features['h2h_away_win_rate'] = weighted_away_wins / total_weight
                h2h_features['h2h_draw_rate'] = weighted_draws / total_weight
                h2h_features['h2h_home_goals_avg'] = weighted_home_goals / total_weight
                h2h_features['h2h_away_goals_avg'] = weighted_away_goals / total_weight
                h2h_features['h2h_total_goals_avg'] = (weighted_home_goals + weighted_away_goals) / total_weight
                h2h_features['h2h_btts_rate'] = weighted_btts / total_weight
                h2h_features['h2h_home_dominance'] = weighted_home_dominance / total_weight

            return h2h_features

        except Exception as e:
            logger.error(f"Error calculating head-to-head features: {str(e)}")
            return {
                'h2h_home_win_rate': 0.4,
                'h2h_away_win_rate': 0.3,
                'h2h_draw_rate': 0.3,
                'h2h_home_goals_avg': 1.5,
                'h2h_away_goals_avg': 1.2,
                'h2h_total_goals_avg': 2.7,
                'h2h_btts_rate': 0.5,
                'h2h_home_dominance': 0.0,
                'h2h_matches_count': 0
            }

    def _add_league_position_features(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add league position and momentum features.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with league position features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        try:
            # Calculate league positions
            home_position, home_momentum = self._calculate_league_position(home_team, match_date)
            away_position, away_momentum = self._calculate_league_position(away_team, match_date)

            # Add features to DataFrame
            result_df['home_league_position'] = home_position
            result_df['away_league_position'] = away_position
            result_df['league_position_diff'] = home_position - away_position
            result_df['home_league_momentum'] = home_momentum
            result_df['away_league_momentum'] = away_momentum
            result_df['league_momentum_diff'] = home_momentum - away_momentum

            return result_df

        except Exception as e:
            logger.error(f"Error adding league position features: {str(e)}")

            # Add default values
            result_df['home_league_position'] = 10
            result_df['away_league_position'] = 10
            result_df['league_position_diff'] = 0
            result_df['home_league_momentum'] = 0
            result_df['away_league_momentum'] = 0
            result_df['league_momentum_diff'] = 0

            return result_df

    def _calculate_league_position(self, team: str, match_date: datetime) -> Tuple[int, float]:
        """
        Calculate approximate league position and momentum.

        Args:
            team: Team name
            match_date: Match date

        Returns:
            Tuple of (league_position, momentum)
        """
        try:
            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                return 10, 0.0

            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])

            # Get recent matches (last 90 days)
            cutoff_date = match_date - timedelta(days=90)
            recent_matches = self.historical_data[
                (self.historical_data['date'] < match_date) &
                (self.historical_data['date'] >= cutoff_date)
            ]

            if recent_matches.empty:
                return 10, 0.0

            # Get unique leagues
            leagues = recent_matches['league'].unique()

            # Find which league the team plays in
            team_leagues = []
            for league in leagues:
                league_matches = recent_matches[recent_matches['league'] == league]
                league_teams = set(league_matches['home_team'].tolist() + league_matches['away_team'].tolist())

                if team in league_teams:
                    team_leagues.append(league)

            if not team_leagues:
                return 10, 0.0

            # Use the most common league
            team_league = max(team_leagues, key=team_leagues.count)

            # Get all matches in this league
            league_matches = recent_matches[recent_matches['league'] == team_league]

            # Get all teams in this league
            league_teams = set(league_matches['home_team'].tolist() + league_matches['away_team'].tolist())

            # Calculate points for each team
            team_points = {}
            team_matches = {}
            team_recent_form = {}

            for t in league_teams:
                team_points[t] = 0
                team_matches[t] = 0
                team_recent_form[t] = []

            # Calculate points
            for _, match in league_matches.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']

                if home_team not in team_points or away_team not in team_points:
                    continue

                home_score = match['home_score']
                away_score = match['away_score']

                # Update matches played
                team_matches[home_team] += 1
                team_matches[away_team] += 1

                # Calculate points
                if home_score > away_score:
                    team_points[home_team] += 3
                    # Add form (1 for win, 0 for draw, -1 for loss)
                    team_recent_form[home_team].append(1)
                    team_recent_form[away_team].append(-1)
                elif home_score == away_score:
                    team_points[home_team] += 1
                    team_points[away_team] += 1
                    # Add form
                    team_recent_form[home_team].append(0)
                    team_recent_form[away_team].append(0)
                else:
                    team_points[away_team] += 3
                    # Add form
                    team_recent_form[home_team].append(-1)
                    team_recent_form[away_team].append(1)

                # Keep only the last 5 matches for form
                team_recent_form[home_team] = team_recent_form[home_team][-5:]
                team_recent_form[away_team] = team_recent_form[away_team][-5:]

            # Calculate points per match
            team_ppg = {}
            for t in league_teams:
                if team_matches[t] > 0:
                    team_ppg[t] = team_points[t] / team_matches[t]
                else:
                    team_ppg[t] = 0

            # Sort teams by points per match
            sorted_teams = sorted(team_ppg.items(), key=lambda x: x[1], reverse=True)

            # Get team position (1-indexed)
            team_position = 1
            for i, (t, _) in enumerate(sorted_teams):
                if t == team:
                    team_position = i + 1
                    break

            # Calculate momentum from recent form
            momentum = 0.0
            if team in team_recent_form and team_recent_form[team]:
                # Weight recent matches more heavily
                weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(team_recent_form[team]):]
                momentum = sum(f * w for f, w in zip(team_recent_form[team], weights))

            return team_position, momentum

        except Exception as e:
            logger.error(f"Error calculating league position: {str(e)}")
            return 10, 0.0

    def _add_travel_fatigue_features(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add travel and fatigue metrics.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with travel and fatigue features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        try:
            # Calculate days since last match
            home_days_since_last_match = self._calculate_days_since_last_match(home_team, match_date)
            away_days_since_last_match = self._calculate_days_since_last_match(away_team, match_date)

            # Calculate match congestion
            home_match_congestion = self._calculate_match_congestion(home_team, match_date)
            away_match_congestion = self._calculate_match_congestion(away_team, match_date)

            # Add features to DataFrame
            result_df['home_days_since_last_match'] = home_days_since_last_match
            result_df['away_days_since_last_match'] = away_days_since_last_match
            result_df['home_match_congestion'] = home_match_congestion
            result_df['away_match_congestion'] = away_match_congestion
            result_df['rest_advantage'] = home_days_since_last_match - away_days_since_last_match

            return result_df

        except Exception as e:
            logger.error(f"Error adding travel fatigue features: {str(e)}")

            # Add default values
            result_df['home_days_since_last_match'] = 7
            result_df['away_days_since_last_match'] = 7
            result_df['home_match_congestion'] = 0.5
            result_df['away_match_congestion'] = 0.5
            result_df['rest_advantage'] = 0

            return result_df

    def _calculate_days_since_last_match(self, team: str, match_date: datetime) -> int:
        """
        Calculate days since last match.

        Args:
            team: Team name
            match_date: Match date

        Returns:
            Days since last match
        """
        try:
            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                return 7

            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])

            # Get team matches before the match date
            team_matches = self.historical_data[
                (
                    (self.historical_data['home_team'] == team) |
                    (self.historical_data['away_team'] == team)
                ) &
                (self.historical_data['date'] < match_date)
            ].sort_values('date', ascending=False)

            if team_matches.empty:
                return 7

            # Get date of last match
            last_match_date = team_matches.iloc[0]['date']

            # Calculate days since last match
            days_since_last_match = (match_date - last_match_date).days

            return max(1, days_since_last_match)

        except Exception as e:
            logger.error(f"Error calculating days since last match: {str(e)}")
            return 7

    def _calculate_match_congestion(self, team: str, match_date: datetime) -> float:
        """
        Calculate match congestion (number of matches in last 30 days).

        Args:
            team: Team name
            match_date: Match date

        Returns:
            Match congestion score (0-1)
        """
        try:
            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                return 0.5

            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])

            # Get matches in last 30 days
            cutoff_date = match_date - timedelta(days=30)
            recent_matches = self.historical_data[
                (
                    (self.historical_data['home_team'] == team) |
                    (self.historical_data['away_team'] == team)
                ) &
                (self.historical_data['date'] < match_date) &
                (self.historical_data['date'] >= cutoff_date)
            ]

            # Count matches
            match_count = len(recent_matches)

            # Calculate congestion score (0-1)
            # 0 = no congestion, 1 = high congestion
            # Assume 8+ matches in 30 days is high congestion
            congestion_score = min(1.0, match_count / 8.0)

            return congestion_score

        except Exception as e:
            logger.error(f"Error calculating match congestion: {str(e)}")
            return 0.5

    def _add_momentum_streak_features(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add momentum and streak indicators.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with momentum and streak features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        try:
            # Calculate streaks
            home_streaks = self._calculate_streaks(home_team, match_date)
            away_streaks = self._calculate_streaks(away_team, match_date)

            # Add features to DataFrame
            for streak_type, value in home_streaks.items():
                result_df[f"home_{streak_type}"] = value

            for streak_type, value in away_streaks.items():
                result_df[f"away_{streak_type}"] = value

            return result_df

        except Exception as e:
            logger.error(f"Error adding momentum streak features: {str(e)}")

            # Add default values
            result_df['home_win_streak'] = 0
            result_df['home_lose_streak'] = 0
            result_df['home_draw_streak'] = 0
            result_df['home_unbeaten_streak'] = 0
            result_df['home_scoring_streak'] = 0
            result_df['home_clean_sheet_streak'] = 0

            result_df['away_win_streak'] = 0
            result_df['away_lose_streak'] = 0
            result_df['away_draw_streak'] = 0
            result_df['away_unbeaten_streak'] = 0
            result_df['away_scoring_streak'] = 0
            result_df['away_clean_sheet_streak'] = 0

            return result_df

    def _calculate_streaks(self, team: str, match_date: datetime) -> Dict[str, int]:
        """
        Calculate various streaks for a team.

        Args:
            team: Team name
            match_date: Match date

        Returns:
            Dictionary with streak values
        """
        try:
            # Initialize streaks
            streaks = {
                'win_streak': 0,
                'lose_streak': 0,
                'draw_streak': 0,
                'unbeaten_streak': 0,
                'scoring_streak': 0,
                'clean_sheet_streak': 0
            }

            # Check if we have historical data
            if self.historical_data is None or self.historical_data.empty:
                return streaks

            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])

            # Get team matches before the match date
            team_matches = self.historical_data[
                (
                    (self.historical_data['home_team'] == team) |
                    (self.historical_data['away_team'] == team)
                ) &
                (self.historical_data['date'] < match_date)
            ].sort_values('date', ascending=False)

            if team_matches.empty:
                return streaks

            # Calculate streaks
            for i, match in team_matches.iterrows():
                is_home = match['home_team'] == team

                # Get team and opponent goals
                team_goals = match['home_score'] if is_home else match['away_score']
                opponent_goals = match['away_score'] if is_home else match['home_score']

                # Check result
                is_win = team_goals > opponent_goals
                is_loss = team_goals < opponent_goals
                is_draw = team_goals == opponent_goals

                # Update win streak
                if is_win:
                    if streaks['win_streak'] == i:  # Consecutive from first match
                        streaks['win_streak'] += 1
                    else:
                        break

                # Update lose streak
                if is_loss:
                    if streaks['lose_streak'] == i:  # Consecutive from first match
                        streaks['lose_streak'] += 1
                    else:
                        break

                # Update draw streak
                if is_draw:
                    if streaks['draw_streak'] == i:  # Consecutive from first match
                        streaks['draw_streak'] += 1
                    else:
                        break

                # Update unbeaten streak
                if is_win or is_draw:
                    if streaks['unbeaten_streak'] == i:  # Consecutive from first match
                        streaks['unbeaten_streak'] += 1
                    else:
                        break

                # Update scoring streak
                if team_goals > 0:
                    if streaks['scoring_streak'] == i:  # Consecutive from first match
                        streaks['scoring_streak'] += 1
                    else:
                        break

                # Update clean sheet streak
                if opponent_goals == 0:
                    if streaks['clean_sheet_streak'] == i:  # Consecutive from first match
                        streaks['clean_sheet_streak'] += 1
                    else:
                        break

            return streaks

        except Exception as e:
            logger.error(f"Error calculating streaks: {str(e)}")
            return {
                'win_streak': 0,
                'lose_streak': 0,
                'draw_streak': 0,
                'unbeaten_streak': 0,
                'scoring_streak': 0,
                'clean_sheet_streak': 0
            }

    def _add_seasonal_pattern_features(self, df: pd.DataFrame, match_date: datetime) -> pd.DataFrame:
        """
        Add seasonal patterns and trends.

        Args:
            df: DataFrame to add features to
            match_date: Match date

        Returns:
            DataFrame with seasonal pattern features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        try:
            # Extract date components
            month = match_date.month
            day_of_week = match_date.weekday()  # 0 = Monday, 6 = Sunday

            # Calculate season progress (0-1)
            # Assume season starts in August (month 8) and ends in May (month 5)
            if month >= 8:
                season_progress = (month - 8) / 9
            else:
                season_progress = (month + 4) / 9

            # Add features to DataFrame
            result_df['month'] = month
            result_df['day_of_week'] = day_of_week
            result_df['is_weekend'] = 1 if day_of_week >= 5 else 0  # 5 = Saturday, 6 = Sunday
            result_df['season_progress'] = season_progress

            # Add season phase
            if season_progress < 0.33:
                result_df['season_phase'] = 'early'
            elif season_progress < 0.67:
                result_df['season_phase'] = 'mid'
            else:
                result_df['season_phase'] = 'late'

            return result_df

        except Exception as e:
            logger.error(f"Error adding seasonal pattern features: {str(e)}")

            # Add default values
            result_df['month'] = match_date.month
            result_df['day_of_week'] = match_date.weekday()
            result_df['is_weekend'] = 1 if match_date.weekday() >= 5 else 0
            result_df['season_progress'] = 0.5
            result_df['season_phase'] = 'mid'

            return result_df

    def _add_tactical_matchup_features(self, df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
        """
        Add tactical matchup analysis.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name

        Returns:
            DataFrame with tactical matchup features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # For now, add placeholder features
        # In a real implementation, this would analyze team playing styles and tactical matchups
        result_df['style_matchup_advantage'] = 0.0
        result_df['tactical_advantage'] = 0.0

        return result_df

    def _add_player_features(self, df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime) -> pd.DataFrame:
        """
        Add player-based features.

        Args:
            df: DataFrame to add features to
            home_team: Home team name
            away_team: Away team name
            match_date: Match date

        Returns:
            DataFrame with player-based features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # For now, add placeholder features
        # In a real implementation, this would analyze player availability, injuries, etc.
        result_df['home_key_player_availability'] = 0.8
        result_df['away_key_player_availability'] = 0.8
        result_df['home_squad_strength'] = 0.7
        result_df['away_squad_strength'] = 0.7

        return result_df

    def _add_weather_features(self, df: pd.DataFrame, fixture_id: str, match_date: datetime) -> pd.DataFrame:
        """
        Add weather and pitch conditions.

        Args:
            df: DataFrame to add features to
            fixture_id: Fixture ID
            match_date: Match date

        Returns:
            DataFrame with weather features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # For now, add placeholder features
        # In a real implementation, this would use weather data
        result_df['temperature'] = 15.0  # Celsius
        result_df['precipitation'] = 0.0  # mm
        result_df['wind_speed'] = 5.0  # km/h
        result_df['is_rainy'] = 0

        return result_df

    def _add_betting_market_features(self, df: pd.DataFrame, fixture_id: str, home_team: str, away_team: str) -> pd.DataFrame:
        """
        Add betting market signals.

        Args:
            df: DataFrame to add features to
            fixture_id: Fixture ID
            home_team: Home team name
            away_team: Away team name

        Returns:
            DataFrame with betting market features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # For now, add placeholder features
        # In a real implementation, this would use betting market data
        result_df['market_home_win_prob'] = 0.45
        result_df['market_draw_prob'] = 0.25
        result_df['market_away_win_prob'] = 0.3
        result_df['market_over_2_5_prob'] = 0.55
        result_df['market_btts_prob'] = 0.6

        return result_df