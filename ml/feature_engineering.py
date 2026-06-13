"""
Feature Engineering Module

This module provides advanced feature engineering for football match prediction.
It extracts and transforms features from historical match data to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import joblib

from utils.common import setup_logging, ensure_directory_exists, safe_divide
from utils.config import settings

# Set up logging
logger = setup_logging(__name__)

class FootballFeatureEngineer:
    """
    Feature engineering for football match prediction.

    Features:
    - Team form (recent performance)
    - Team attack and defense strength
    - Head-to-head statistics
    - League position and momentum
    - Home/away performance
    - Goal scoring patterns
    - Match importance
    - Rest days between matches
    """

    def __init__(self, historical_matches: Optional[pd.DataFrame] = None):
        """
        Initialize the feature engineer.

        Args:
            historical_matches: DataFrame containing historical match data
        """
        self.historical_matches = historical_matches

        # Cache for team statistics to avoid recalculation
        self.team_stats_cache = {}
        self.h2h_cache = {}

        # Cache directory for persistent caching
        self.cache_dir = os.path.join(settings.ml.CACHE_DIR, "features")
        ensure_directory_exists(self.cache_dir)

        # Cache expiry (in hours)
        self.cache_expiry = settings.ml.FEATURE_CACHE_EXPIRY

    def set_historical_data(self, historical_matches: pd.DataFrame) -> None:
        """
        Set historical match data.

        Args:
            historical_matches: DataFrame containing historical match data
        """
        self.historical_matches = historical_matches

        # Reset caches
        self.team_stats_cache = {}
        self.h2h_cache = {}

    def engineer_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for a set of matches.

        Args:
            matches_df: DataFrame containing matches to engineer features for

        Returns:
            DataFrame with engineered features
        """
        if self.historical_matches is None:
            raise ValueError("Historical match data not set. Call set_historical_data first.")

        # Create a copy to avoid modifying the original
        df = matches_df.copy()

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Add basic features
        logger.info("Engineering basic features...")
        df = self._add_basic_features(df)

        # Add team form features
        logger.info("Engineering team form features...")
        df = self._add_team_form_features(df)

        # Add team strength features
        logger.info("Engineering team strength features...")
        df = self._add_team_strength_features(df)

        # Add head-to-head features
        logger.info("Engineering head-to-head features...")
        df = self._add_head_to_head_features(df)

        # Add league position features
        logger.info("Engineering league position features...")
        df = self._add_league_position_features(df)

        # Add time-based features
        logger.info("Engineering time-based features...")
        df = self._add_time_based_features(df)

        # Add match importance features
        logger.info("Engineering match importance features...")
        df = self._add_match_importance_features(df)

        # Fill missing values
        df = df.fillna(0)

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with basic features
        """
        # Create a unique match ID if not present
        if 'match_id' not in df.columns:
            df['match_id'] = df.apply(
                lambda row: f"{row.get('competition_name', 'unknown')}_{row.get('season', 'unknown')}_{row.get('date', 'unknown')}_{row.get('home_team', 'unknown')}_{row.get('away_team', 'unknown')}",
                axis=1
            )

        return df

    def _add_team_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team form features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with team form features
        """
        # For each match, calculate form features
        for i, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']

            # Get previous matches for home team
            home_form = self._calculate_team_form(home_team, match_date, 10)

            # Get previous matches for away team
            away_form = self._calculate_team_form(away_team, match_date, 10)

            # Add form features
            df.at[i, 'home_form_points'] = home_form.get('points_per_game', 0.5)
            df.at[i, 'home_form_win_rate'] = home_form.get('win_rate', 0.5)
            df.at[i, 'home_form_loss_rate'] = home_form.get('loss_rate', 0.5)
            df.at[i, 'home_form_draw_rate'] = home_form.get('draw_rate', 0.5)
            df.at[i, 'home_form_goals_scored'] = home_form.get('goals_scored_per_game', 1.5)
            df.at[i, 'home_form_goals_conceded'] = home_form.get('goals_conceded_per_game', 1.5)
            df.at[i, 'home_form_clean_sheets'] = home_form.get('clean_sheet_rate', 0.3)

            df.at[i, 'away_form_points'] = away_form.get('points_per_game', 0.5)
            df.at[i, 'away_form_win_rate'] = away_form.get('win_rate', 0.5)
            df.at[i, 'away_form_loss_rate'] = away_form.get('loss_rate', 0.5)
            df.at[i, 'away_form_draw_rate'] = away_form.get('draw_rate', 0.5)
            df.at[i, 'away_form_goals_scored'] = away_form.get('goals_scored_per_game', 1.5)
            df.at[i, 'away_form_goals_conceded'] = away_form.get('goals_conceded_per_game', 1.5)
            df.at[i, 'away_form_clean_sheets'] = away_form.get('clean_sheet_rate', 0.3)

        return df

    def _calculate_team_form(self, team: str, date: datetime, num_matches: int = 10) -> Dict[str, float]:
        """
        Calculate team form based on previous matches.

        Args:
            team: Team name
            date: Date to calculate form up to
            num_matches: Number of previous matches to consider

        Returns:
            Dictionary with form metrics
        """
        # Create cache key
        cache_key = f"{team}_{date.strftime('%Y-%m-%d')}_{num_matches}"

        # Check in-memory cache
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"form_{cache_key}.joblib")
        if os.path.exists(cache_file):
            # Check if cache is still valid
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=self.cache_expiry):
                try:
                    return joblib.load(cache_file)
                except Exception as e:
                    logger.warning(f"Error loading form cache: {str(e)}")

        # Get previous matches for the team
        team_matches = self.historical_matches[
            ((self.historical_matches['home_team'] == team) |
             (self.historical_matches['away_team'] == team)) &
            (self.historical_matches['date'] < date)
        ].sort_values('date', ascending=False).head(num_matches)

        # If no previous matches, return default values
        if len(team_matches) == 0:
            default_form = {
                'points_per_game': 0.5,
                'win_rate': 0.5,
                'loss_rate': 0.5,
                'draw_rate': 0.5,
                'goals_scored_per_game': 1.5,
                'goals_conceded_per_game': 1.5,
                'clean_sheet_rate': 0.3
            }

            # Cache the result
            self.team_stats_cache[cache_key] = default_form

            return default_form

        # Calculate form metrics
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0

        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                # Team played at home
                team_goals = match['home_score']
                opponent_goals = match['away_score']

                if team_goals > opponent_goals:
                    wins += 1
                elif team_goals < opponent_goals:
                    losses += 1
                else:
                    draws += 1
            else:
                # Team played away
                team_goals = match['away_score']
                opponent_goals = match['home_score']

                if team_goals > opponent_goals:
                    wins += 1
                elif team_goals < opponent_goals:
                    losses += 1
                else:
                    draws += 1

            goals_scored += team_goals
            goals_conceded += opponent_goals

            if opponent_goals == 0:
                clean_sheets += 1

        # Calculate rates
        num_games = len(team_matches)
        win_rate = safe_divide(wins, num_games, 0.5)
        loss_rate = safe_divide(losses, num_games, 0.5)
        draw_rate = safe_divide(draws, num_games, 0.5)
        points_per_game = safe_divide(wins * 3 + draws, num_games, 0.5)
        goals_scored_per_game = safe_divide(goals_scored, num_games, 1.5)
        goals_conceded_per_game = safe_divide(goals_conceded, num_games, 1.5)
        clean_sheet_rate = safe_divide(clean_sheets, num_games, 0.3)

        # Create form dictionary
        form = {
            'points_per_game': points_per_game,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate,
            'goals_scored_per_game': goals_scored_per_game,
            'goals_conceded_per_game': goals_conceded_per_game,
            'clean_sheet_rate': clean_sheet_rate
        }

        # Cache the result in memory
        self.team_stats_cache[cache_key] = form

        # Cache the result on disk
        try:
            joblib.dump(form, cache_file)
        except Exception as e:
            logger.warning(f"Error saving form cache: {str(e)}")

        return form

    def _add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team strength features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with team strength features
        """
        # For each match, calculate strength features
        for i, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']

            # Calculate team strengths
            home_strength = self._calculate_team_strength(home_team, match_date)
            away_strength = self._calculate_team_strength(away_team, match_date)

            # Add strength features
            df.at[i, 'home_attack_strength'] = home_strength.get('attack_strength', 0.5)
            df.at[i, 'home_defense_strength'] = home_strength.get('defense_strength', 0.5)
            df.at[i, 'home_overall_strength'] = home_strength.get('overall_strength', 0.5)

            df.at[i, 'away_attack_strength'] = away_strength.get('attack_strength', 0.5)
            df.at[i, 'away_defense_strength'] = away_strength.get('defense_strength', 0.5)
            df.at[i, 'away_overall_strength'] = away_strength.get('overall_strength', 0.5)

            # Add relative strength features
            df.at[i, 'relative_attack_strength'] = home_strength.get('attack_strength', 0.5) - away_strength.get('defense_strength', 0.5)
            df.at[i, 'relative_defense_strength'] = home_strength.get('defense_strength', 0.5) - away_strength.get('attack_strength', 0.5)
            df.at[i, 'relative_overall_strength'] = home_strength.get('overall_strength', 0.5) - away_strength.get('overall_strength', 0.5)

        return df

    def _calculate_team_strength(self, team: str, date: datetime, num_matches: int = 20) -> Dict[str, float]:
        """
        Calculate team strength based on previous matches.

        Args:
            team: Team name
            date: Date to calculate strength up to
            num_matches: Number of previous matches to consider

        Returns:
            Dictionary with strength metrics
        """
        # Create cache key
        cache_key = f"{team}_{date.strftime('%Y-%m-%d')}_{num_matches}_strength"

        # Check cache
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]

        # Get previous matches for the team
        team_matches = self.historical_matches[
            ((self.historical_matches['home_team'] == team) |
             (self.historical_matches['away_team'] == team)) &
            (self.historical_matches['date'] < date)
        ].sort_values('date', ascending=False).head(num_matches)

        # If no previous matches, return default values
        if len(team_matches) == 0:
            return {
                'attack_strength': 0.5,
                'defense_strength': 0.5,
                'overall_strength': 0.5
            }

        # Calculate average goals scored and conceded
        goals_scored = []
        goals_conceded = []

        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                # Team played at home
                goals_scored.append(match['home_score'])
                goals_conceded.append(match['away_score'])
            else:
                # Team played away
                goals_scored.append(match['away_score'])
                goals_conceded.append(match['home_score'])

        avg_goals_scored = np.mean(goals_scored)
        avg_goals_conceded = np.mean(goals_conceded)

        # Calculate league average goals
        league_matches = self.historical_matches[
            (self.historical_matches['date'] < date)
        ].tail(1000)  # Use recent matches for league average

        league_avg_goals = (league_matches['home_score'].mean() + league_matches['away_score'].mean()) / 2

        # Calculate strength metrics
        attack_strength = avg_goals_scored / league_avg_goals if league_avg_goals > 0 else 0.5
        defense_strength = 1 - (avg_goals_conceded / league_avg_goals) if league_avg_goals > 0 else 0.5
        overall_strength = (attack_strength + defense_strength) / 2

        # Normalize to 0-1 range
        attack_strength = min(max(attack_strength, 0), 1)
        defense_strength = min(max(defense_strength, 0), 1)
        overall_strength = min(max(overall_strength, 0), 1)

        # Create strength dictionary
        strength = {
            'attack_strength': attack_strength,
            'defense_strength': defense_strength,
            'overall_strength': overall_strength
        }

        # Cache the result
        self.team_stats_cache[cache_key] = strength

        return strength

    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add head-to-head features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with head-to-head features
        """
        # For each match, calculate head-to-head features
        for i, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']

            # Get head-to-head statistics
            h2h_stats = self._calculate_head_to_head(home_team, away_team, match_date)

            # Add head-to-head features
            df.at[i, 'h2h_home_win_rate'] = h2h_stats.get('home_win_rate', 0.5)
            df.at[i, 'h2h_away_win_rate'] = h2h_stats.get('away_win_rate', 0.5)
            df.at[i, 'h2h_draw_rate'] = h2h_stats.get('draw_rate', 0.5)
            df.at[i, 'h2h_home_goals_avg'] = h2h_stats.get('home_goals_avg', 1.5)
            df.at[i, 'h2h_away_goals_avg'] = h2h_stats.get('away_goals_avg', 1.5)
            df.at[i, 'h2h_total_goals_avg'] = h2h_stats.get('total_goals_avg', 2.5)

        return df

    def _calculate_head_to_head(self, home_team: str, away_team: str, date: datetime, num_matches: int = 10) -> Dict[str, float]:
        """
        Calculate head-to-head statistics.

        Args:
            home_team: Home team name
            away_team: Away team name
            date: Date to calculate statistics up to
            num_matches: Number of previous matches to consider

        Returns:
            Dictionary with head-to-head statistics
        """
        # Create cache key
        cache_key = f"{home_team}_{away_team}_{date.strftime('%Y-%m-%d')}_{num_matches}"

        # Check cache
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]

        # Get previous matches between the teams
        h2h_matches = self.historical_matches[
            (((self.historical_matches['home_team'] == home_team) &
              (self.historical_matches['away_team'] == away_team)) |
             ((self.historical_matches['home_team'] == away_team) &
              (self.historical_matches['away_team'] == home_team))) &
            (self.historical_matches['date'] < date)
        ].sort_values('date', ascending=False).head(num_matches)

        # If no previous matches, return default values
        if len(h2h_matches) == 0:
            return {
                'home_win_rate': 0.5,
                'away_win_rate': 0.5,
                'draw_rate': 0.5,
                'home_goals_avg': 1.5,
                'away_goals_avg': 1.5,
                'total_goals_avg': 2.5
            }

        # Calculate head-to-head statistics
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals = []
        away_goals = []

        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team and match['away_team'] == away_team:
                # Same configuration as the current match
                home_team_goals = match['home_score']
                away_team_goals = match['away_score']
            else:
                # Reverse configuration
                home_team_goals = match['away_score']
                away_team_goals = match['home_score']

            if home_team_goals > away_team_goals:
                home_wins += 1
            elif home_team_goals < away_team_goals:
                away_wins += 1
            else:
                draws += 1

            home_goals.append(home_team_goals)
            away_goals.append(away_team_goals)

        # Calculate rates
        num_games = len(h2h_matches)
        home_win_rate = home_wins / num_games
        away_win_rate = away_wins / num_games
        draw_rate = draws / num_games
        home_goals_avg = np.mean(home_goals)
        away_goals_avg = np.mean(away_goals)
        total_goals_avg = home_goals_avg + away_goals_avg

        # Create head-to-head dictionary
        h2h_stats = {
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate,
            'draw_rate': draw_rate,
            'home_goals_avg': home_goals_avg,
            'away_goals_avg': away_goals_avg,
            'total_goals_avg': total_goals_avg
        }

        # Cache the result
        self.h2h_cache[cache_key] = h2h_stats

        return h2h_stats

    def _add_league_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add league position features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with league position features
        """
        # If league position is already in the data, use it
        if 'home_position' in df.columns and 'away_position' in df.columns:
            df['league_position_diff'] = df['home_position'] - df['away_position']
            return df

        # Otherwise, calculate approximate league positions
        for i, row in df.iterrows():
            if 'competition_name' not in df.columns or 'season' not in df.columns:
                # Skip if competition or season not available
                continue

            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']
            competition = row['competition_name']
            season = row['season']

            # Calculate approximate league positions
            home_position = self._calculate_league_position(home_team, competition, season, match_date)
            away_position = self._calculate_league_position(away_team, competition, season, match_date)

            # Add league position features
            df.at[i, 'home_position'] = home_position
            df.at[i, 'away_position'] = away_position
            df.at[i, 'league_position_diff'] = home_position - away_position

        return df

    def _calculate_league_position(self, team: str, competition: str, season: str, date: datetime) -> int:
        """
        Calculate approximate league position.

        Args:
            team: Team name
            competition: Competition name
            season: Season
            date: Date to calculate position up to

        Returns:
            Approximate league position
        """
        # Create cache key
        cache_key = f"{team}_{competition}_{season}_{date.strftime('%Y-%m-%d')}_position"

        # Check cache
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]

        # Get previous matches in the same competition and season
        competition_matches = self.historical_matches[
            (self.historical_matches['competition_name'] == competition) &
            (self.historical_matches['season'] == season) &
            (self.historical_matches['date'] < date)
        ]

        # If no previous matches, return default value
        if len(competition_matches) == 0:
            return 10  # Middle of the table

        # Calculate points for each team
        team_points = {}

        for _, match in competition_matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_score = match['home_score']
            away_score = match['away_score']

            # Initialize team points if not already in dictionary
            if home_team not in team_points:
                team_points[home_team] = 0
            if away_team not in team_points:
                team_points[away_team] = 0

            # Add points based on match result
            if home_score > away_score:
                team_points[home_team] += 3
            elif home_score < away_score:
                team_points[away_team] += 3
            else:
                team_points[home_team] += 1
                team_points[away_team] += 1

        # Sort teams by points
        sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)

        # Find position of the team
        position = 10  # Default to middle of the table

        for i, (t, _) in enumerate(sorted_teams):
            if t == team:
                position = i + 1
                break

        # Cache the result
        self.team_stats_cache[cache_key] = position

        return position

    def _add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with time-based features
        """
        # Add day of week
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek

            # Add month of year
            df['month'] = df['date'].dt.month

            # Add days since start of season
            if 'season' in df.columns:
                for i, row in df.iterrows():
                    season = row['season']
                    match_date = row['date']

                    # Estimate season start date (August 1st of the first year in the season)
                    if isinstance(season, str) and '-' in season:
                        start_year = int(season.split('-')[0])
                        season_start = datetime(start_year, 8, 1)
                    else:
                        # If season format is unknown, use a default value
                        season_start = match_date - timedelta(days=180)

                    # Calculate days since start of season
                    days_since_start = (match_date - season_start).days
                    df.at[i, 'days_since_season_start'] = max(0, days_since_start)

        return df

    def _add_match_importance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add match importance features.

        Args:
            df: DataFrame to add features to

        Returns:
            DataFrame with match importance features
        """
        # Add match importance based on league position and time of season
        if 'home_position' in df.columns and 'away_position' in df.columns and 'days_since_season_start' in df.columns:
            for i, row in df.iterrows():
                home_position = row['home_position']
                away_position = row['away_position']
                days_since_start = row['days_since_season_start']

                # Calculate season progress (0-1)
                season_progress = min(days_since_start / 300, 1)  # Assuming season is ~300 days

                # Calculate match importance based on league positions and season progress
                # Higher importance for matches between teams close in the table
                position_diff = abs(home_position - away_position)
                position_importance = 1 / (1 + position_diff)

                # Higher importance for matches later in the season
                time_importance = 0.5 + 0.5 * season_progress

                # Higher importance for matches involving top teams
                top_team_importance = 1 / (1 + min(home_position, away_position))

                # Combine factors
                match_importance = (position_importance + time_importance + top_team_importance) / 3

                df.at[i, 'match_importance'] = match_importance

        return df

class AdvancedFootballFeatureEngineer(FootballFeatureEngineer):
    """
    Advanced feature engineering for football match prediction.

    Extends the base FootballFeatureEngineer with additional advanced features
    and optimizations for production use.
    """

    def __init__(self, historical_matches: Optional[pd.DataFrame] = None):
        """Initialize the advanced feature engineer."""
        super().__init__(historical_matches)

        # Additional caches for advanced features
        self.momentum_cache = {}
        self.seasonal_cache = {}

    def engineer_features_for_match(self, home_team: str, away_team: str,
                                  league: str, match_date: datetime) -> pd.DataFrame:
        """
        Engineer features for a single match.

        Args:
            home_team: Home team name
            away_team: Away team name
            league: League name
            match_date: Match date

        Returns:
            DataFrame with engineered features for the match
        """
        # Create a single match DataFrame
        match_df = pd.DataFrame({
            'home_team': [home_team],
            'away_team': [away_team],
            'competition_name': [league],
            'date': [match_date]
        })

        # If no historical data, return basic features
        if self.historical_matches is None or len(self.historical_matches) == 0:
            return self._create_basic_features(match_df)

        # Engineer full features
        try:
            return self.engineer_features(match_df)
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return self._create_basic_features(match_df)

    def _create_basic_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features when historical data is not available."""
        df = match_df.copy()

        # Basic team strength mapping
        team_strengths = {
            "Arsenal": 0.8, "Chelsea": 0.8, "Manchester City": 0.9, "Liverpool": 0.85,
            "Manchester United": 0.75, "Tottenham": 0.7, "Real Madrid": 0.9,
            "Barcelona": 0.85, "Bayern Munich": 0.9, "PSG": 0.85, "Juventus": 0.8,
            "AC Milan": 0.75, "Inter Milan": 0.75, "Atletico Madrid": 0.8,
            "Borussia Dortmund": 0.75, "Ajax": 0.7
        }

        for i, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']

            home_strength = team_strengths.get(home_team, 0.5) + 0.1  # Home advantage
            away_strength = team_strengths.get(away_team, 0.5)

            # Basic features
            df.at[i, 'home_attack_strength'] = home_strength
            df.at[i, 'home_defense_strength'] = home_strength
            df.at[i, 'home_overall_strength'] = home_strength
            df.at[i, 'away_attack_strength'] = away_strength
            df.at[i, 'away_defense_strength'] = away_strength
            df.at[i, 'away_overall_strength'] = away_strength

            # Form features (defaults)
            df.at[i, 'home_form_points'] = home_strength * 2
            df.at[i, 'home_form_win_rate'] = home_strength * 0.8
            df.at[i, 'home_form_goals_scored'] = home_strength * 2
            df.at[i, 'away_form_points'] = away_strength * 2
            df.at[i, 'away_form_win_rate'] = away_strength * 0.8
            df.at[i, 'away_form_goals_scored'] = away_strength * 2

            # H2H features (defaults)
            df.at[i, 'h2h_home_win_rate'] = 0.4 if home_strength > away_strength else 0.3
            df.at[i, 'h2h_away_win_rate'] = 0.3 if away_strength > home_strength else 0.2
            df.at[i, 'h2h_draw_rate'] = 0.3
            df.at[i, 'h2h_total_goals_avg'] = 2.5

            # Relative strength
            df.at[i, 'relative_overall_strength'] = home_strength - away_strength

        # Fill any missing values
        df = df.fillna(0.5)

        return df


# Create singleton instances
feature_engineer = FootballFeatureEngineer()
advanced_feature_engineer = AdvancedFootballFeatureEngineer()
