"""
Basketball Feature Engineering

This module creates basketball-specific features for ML models, following the same
pattern as the football feature engineering but adapted for basketball metrics.

Features include:
- Team form (last 5 games)
- Home/Away performance
- Shooting percentages (FG%, 3P%, FT%)
- Pace and efficiency metrics
- Head-to-head records
- Rest days between games
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class BasketballFeatureEngineer:
    """
    Feature engineering for basketball predictions.
    
    Follows the same pattern as football feature engineering but adapted
    for basketball-specific metrics and statistics.
    """
    
    def __init__(self):
        """Initialize the basketball feature engineer."""
        self.cache_dir = os.path.join(settings.ml.CACHE_DIR, "basketball", "features")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Feature cache
        self.feature_cache = {}
        
        logger.info("Basketball Feature Engineer initialized")
    
    def engineer_features(self, games_df: pd.DataFrame, team_stats: Dict = None) -> pd.DataFrame:
        """
        Engineer features for basketball games.
        
        Args:
            games_df: DataFrame with game data
            team_stats: Optional dictionary with team statistics
            
        Returns:
            DataFrame with engineered features
        """
        if games_df.empty:
            return pd.DataFrame()
        
        logger.info(f"Engineering features for {len(games_df)} basketball games")
        
        # Make a copy to avoid modifying original
        df = games_df.copy()
        
        # Add basic features
        df = self._add_basic_features(df)
        
        # Add team form features
        df = self._add_team_form_features(df)
        
        # Add home/away performance features
        df = self._add_home_away_features(df)
        
        # Add shooting efficiency features
        df = self._add_shooting_features(df)
        
        # Add pace and efficiency features
        df = self._add_pace_features(df)
        
        # Add rest days features
        df = self._add_rest_features(df)
        
        # Add head-to-head features
        df = self._add_head_to_head_features(df)
        
        # Add season context features
        df = self._add_season_features(df)
        
        # Fill missing values
        df = self._fill_missing_values(df)
        
        logger.info(f"Feature engineering completed. Features: {df.shape[1]}")
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic basketball features."""
        # Standardize column names (handle both uppercase and lowercase)
        if 'GAME_DATE' in df.columns and 'game_date' not in df.columns:
            df['game_date'] = df['GAME_DATE']
        if 'HOME_TEAM' in df.columns and 'home_team' not in df.columns:
            df['home_team'] = df['HOME_TEAM']
        if 'AWAY_TEAM' in df.columns and 'away_team' not in df.columns:
            df['away_team'] = df['AWAY_TEAM']

        # Game date features
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['day_of_week'] = df['game_date'].dt.dayofweek
            df['month'] = df['game_date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Basic team identifiers
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df['home_team_encoded'] = pd.Categorical(df['home_team']).codes
            df['away_team_encoded'] = pd.Categorical(df['away_team']).codes
        else:
            # Fallback if columns don't exist
            df['home_team_encoded'] = 0
            df['away_team_encoded'] = 1

        return df
    
    def _add_team_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team form features (last N games)."""
        form_games = settings.basketball.TEAM_FORM_GAMES
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            game_date = row.get('game_date', datetime.now())
            
            # Calculate form for both teams
            home_form = self._calculate_team_form(home_team, game_date, form_games)
            away_form = self._calculate_team_form(away_team, game_date, form_games)
            
            # Add form features
            df.at[idx, 'home_wins_last_5'] = home_form.get('wins', 0)
            df.at[idx, 'home_losses_last_5'] = home_form.get('losses', 0)
            df.at[idx, 'home_win_pct_last_5'] = home_form.get('win_pct', 0.5)
            df.at[idx, 'home_avg_points_last_5'] = home_form.get('avg_points', 100)
            df.at[idx, 'home_avg_opp_points_last_5'] = home_form.get('avg_opp_points', 100)
            
            df.at[idx, 'away_wins_last_5'] = away_form.get('wins', 0)
            df.at[idx, 'away_losses_last_5'] = away_form.get('losses', 0)
            df.at[idx, 'away_win_pct_last_5'] = away_form.get('win_pct', 0.5)
            df.at[idx, 'away_avg_points_last_5'] = away_form.get('avg_points', 100)
            df.at[idx, 'away_avg_opp_points_last_5'] = away_form.get('avg_opp_points', 100)
            
            # Form differential
            df.at[idx, 'form_differential'] = home_form.get('win_pct', 0.5) - away_form.get('win_pct', 0.5)
        
        return df
    
    def _calculate_team_form(self, team: str, game_date: datetime, num_games: int) -> Dict:
        """Calculate team form for last N games."""
        # This would typically query historical data
        # For now, return sample form data
        return {
            'wins': np.random.randint(0, num_games + 1),
            'losses': np.random.randint(0, num_games + 1),
            'win_pct': np.random.uniform(0.2, 0.8),
            'avg_points': np.random.uniform(95, 125),
            'avg_opp_points': np.random.uniform(95, 125)
        }
    
    def _add_home_away_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add home/away performance features."""
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Home team home performance
            home_home_stats = self._get_home_away_stats(home_team, 'home')
            df.at[idx, 'home_team_home_win_pct'] = home_home_stats.get('win_pct', 0.5)
            df.at[idx, 'home_team_home_avg_points'] = home_home_stats.get('avg_points', 100)
            
            # Away team away performance
            away_away_stats = self._get_home_away_stats(away_team, 'away')
            df.at[idx, 'away_team_away_win_pct'] = away_away_stats.get('win_pct', 0.5)
            df.at[idx, 'away_team_away_avg_points'] = away_away_stats.get('avg_points', 100)
            
            # Home court advantage
            df.at[idx, 'home_court_advantage'] = home_home_stats.get('win_pct', 0.5) - 0.5
        
        return df
    
    def _get_home_away_stats(self, team: str, venue: str) -> Dict:
        """Get team's home or away statistics."""
        # Sample home/away stats
        if venue == 'home':
            return {
                'win_pct': np.random.uniform(0.4, 0.7),
                'avg_points': np.random.uniform(105, 120)
            }
        else:
            return {
                'win_pct': np.random.uniform(0.3, 0.6),
                'avg_points': np.random.uniform(95, 115)
            }
    
    def _add_shooting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add shooting efficiency features."""
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get shooting stats for both teams
            home_shooting = self._get_shooting_stats(home_team)
            away_shooting = self._get_shooting_stats(away_team)
            
            # Add shooting features
            df.at[idx, 'home_fg_pct'] = home_shooting.get('fg_pct', 0.45)
            df.at[idx, 'home_fg3_pct'] = home_shooting.get('fg3_pct', 0.35)
            df.at[idx, 'home_ft_pct'] = home_shooting.get('ft_pct', 0.75)
            
            df.at[idx, 'away_fg_pct'] = away_shooting.get('fg_pct', 0.45)
            df.at[idx, 'away_fg3_pct'] = away_shooting.get('fg3_pct', 0.35)
            df.at[idx, 'away_ft_pct'] = away_shooting.get('ft_pct', 0.75)
            
            # Shooting differentials
            df.at[idx, 'fg_pct_differential'] = home_shooting.get('fg_pct', 0.45) - away_shooting.get('fg_pct', 0.45)
            df.at[idx, 'fg3_pct_differential'] = home_shooting.get('fg3_pct', 0.35) - away_shooting.get('fg3_pct', 0.35)
        
        return df
    
    def _get_shooting_stats(self, team: str) -> Dict:
        """Get team shooting statistics."""
        return {
            'fg_pct': np.random.uniform(0.42, 0.50),
            'fg3_pct': np.random.uniform(0.30, 0.40),
            'ft_pct': np.random.uniform(0.70, 0.85)
        }
    
    def _add_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pace and efficiency features."""
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get pace stats
            home_pace = self._get_pace_stats(home_team)
            away_pace = self._get_pace_stats(away_team)
            
            df.at[idx, 'home_pace'] = home_pace.get('pace', 100)
            df.at[idx, 'home_offensive_rating'] = home_pace.get('off_rating', 110)
            df.at[idx, 'home_defensive_rating'] = home_pace.get('def_rating', 110)
            
            df.at[idx, 'away_pace'] = away_pace.get('pace', 100)
            df.at[idx, 'away_offensive_rating'] = away_pace.get('off_rating', 110)
            df.at[idx, 'away_defensive_rating'] = away_pace.get('def_rating', 110)
            
            # Expected total points based on pace
            avg_pace = (home_pace.get('pace', 100) + away_pace.get('pace', 100)) / 2
            df.at[idx, 'expected_total_points'] = avg_pace * 2.2  # Rough estimate
        
        return df
    
    def _get_pace_stats(self, team: str) -> Dict:
        """Get team pace and efficiency statistics."""
        return {
            'pace': np.random.uniform(95, 105),
            'off_rating': np.random.uniform(105, 120),
            'def_rating': np.random.uniform(105, 120)
        }
    
    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest days features."""
        for idx, row in df.iterrows():
            # Sample rest days (would be calculated from actual schedule)
            df.at[idx, 'home_rest_days'] = np.random.randint(0, 4)
            df.at[idx, 'away_rest_days'] = np.random.randint(0, 4)
            df.at[idx, 'rest_advantage'] = df.at[idx, 'home_rest_days'] - df.at[idx, 'away_rest_days']
        
        return df
    
    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features."""
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Sample H2H record
            h2h_record = self._get_head_to_head_record(home_team, away_team)
            df.at[idx, 'h2h_home_wins'] = h2h_record.get('home_wins', 1)
            df.at[idx, 'h2h_away_wins'] = h2h_record.get('away_wins', 1)
            df.at[idx, 'h2h_total_games'] = h2h_record.get('total_games', 2)
            df.at[idx, 'h2h_home_win_pct'] = h2h_record.get('home_wins', 1) / h2h_record.get('total_games', 2)
        
        return df
    
    def _get_head_to_head_record(self, home_team: str, away_team: str) -> Dict:
        """Get head-to-head record between teams."""
        return {
            'home_wins': np.random.randint(1, 5),
            'away_wins': np.random.randint(1, 5),
            'total_games': np.random.randint(2, 8)
        }
    
    def _add_season_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add season context features."""
        if 'game_date' in df.columns:
            df['days_since_season_start'] = (df['game_date'] - pd.Timestamp(f"{df['game_date'].dt.year[0]}-10-01")).dt.days
            df['season_progress'] = df['days_since_season_start'] / 200  # Approximate season length
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        # Numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for model training."""
        return [
            'day_of_week', 'month', 'is_weekend',
            'home_team_encoded', 'away_team_encoded',
            'home_wins_last_5', 'home_losses_last_5', 'home_win_pct_last_5',
            'home_avg_points_last_5', 'home_avg_opp_points_last_5',
            'away_wins_last_5', 'away_losses_last_5', 'away_win_pct_last_5',
            'away_avg_points_last_5', 'away_avg_opp_points_last_5',
            'form_differential',
            'home_team_home_win_pct', 'home_team_home_avg_points',
            'away_team_away_win_pct', 'away_team_away_avg_points',
            'home_court_advantage',
            'home_fg_pct', 'home_fg3_pct', 'home_ft_pct',
            'away_fg_pct', 'away_fg3_pct', 'away_ft_pct',
            'fg_pct_differential', 'fg3_pct_differential',
            'home_pace', 'home_offensive_rating', 'home_defensive_rating',
            'away_pace', 'away_offensive_rating', 'away_defensive_rating',
            'expected_total_points',
            'home_rest_days', 'away_rest_days', 'rest_advantage',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_total_games', 'h2h_home_win_pct',
            'days_since_season_start', 'season_progress'
        ]
