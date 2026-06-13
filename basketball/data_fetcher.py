"""
NBA Data Fetcher for Basketball Predictions

This module fetches NBA data using free APIs and data sources:
1. nba_api package for live NBA data
2. Kaggle datasets for historical training data
3. Basketball Reference scraping (if needed)

All data sources are free and do not require API keys.
"""

import logging
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# NBA API imports (free package)
try:
    from nba_api.stats.endpoints import (
        leaguegamefinder, teamgamelogs,
        teamdashboardbygeneralsplits, leaguestandings
    )
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
    logger.info("NBA API imported successfully")
except ImportError as e:
    logger.warning(f"nba_api not available: {str(e)}. Install with: pip install nba_api")
    NBA_API_AVAILABLE = False
except Exception as e:
    logger.warning(f"nba_api import error: {str(e)}. Using fallback mode.")
    NBA_API_AVAILABLE = False

class NBADataFetcher:
    """
    Fetches NBA data from free sources for basketball predictions.
    
    Features:
    - Live NBA games and schedules
    - Team statistics and standings
    - Historical game data
    - Player statistics (basic)
    - No API keys required
    """
    
    def __init__(self):
        """Initialize the NBA data fetcher."""
        self.cache_dir = os.path.join(settings.ml.CACHE_DIR, "basketball")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # NBA teams mapping
        self.nba_teams = self._get_nba_teams() if NBA_API_AVAILABLE else {}
        
        logger.info("NBA Data Fetcher initialized")
    
    def _get_nba_teams(self) -> Dict[int, Dict]:
        """Get NBA teams mapping."""
        try:
            teams_data = teams.get_teams()
            return {team['id']: team for team in teams_data}
        except Exception as e:
            logger.error(f"Error fetching NBA teams: {str(e)}")
            return {}
    
    def fetch_today_games(self) -> pd.DataFrame:
        """
        Fetch today's NBA games.
        
        Returns:
            DataFrame with today's games
        """
        if not NBA_API_AVAILABLE:
            logger.error("NBA API not available")
            return pd.DataFrame()
        
        try:
            # Get today's games using leaguegamefinder
            today = datetime.now().strftime("%Y-%m-%d")

            # Use leaguegamefinder to get recent games
            game_finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=today,
                date_to_nullable=today
            )
            games_df = game_finder.get_data_frames()[0]

            if games_df.empty:
                logger.info("No NBA games today")
                return pd.DataFrame()
            
            # Process games data
            processed_games = []
            for _, game in games_df.iterrows():
                game_data = {
                    'game_id': game['GAME_ID'],
                    'game_date': game['GAME_DATE_EST'],
                    'home_team_id': game['HOME_TEAM_ID'],
                    'away_team_id': game['VISITOR_TEAM_ID'],
                    'home_team': self.nba_teams.get(game['HOME_TEAM_ID'], {}).get('full_name', 'Unknown'),
                    'away_team': self.nba_teams.get(game['VISITOR_TEAM_ID'], {}).get('full_name', 'Unknown'),
                    'game_status': game['GAME_STATUS_TEXT'],
                    'season': game['SEASON']
                }
                processed_games.append(game_data)
            
            result_df = pd.DataFrame(processed_games)
            
            # Cache the data
            cache_file = os.path.join(self.cache_dir, f"games_{today}.csv")
            result_df.to_csv(cache_file, index=False)
            
            logger.info(f"Fetched {len(result_df)} NBA games for {today}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error fetching today's games: {str(e)}")
            return pd.DataFrame()
    
    def fetch_team_stats(self, team_id: int, season: str = "2023-24") -> Dict:
        """
        Fetch team statistics for a given season.
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., "2023-24")
            
        Returns:
            Dictionary with team statistics
        """
        if not NBA_API_AVAILABLE:
            return {}
        
        try:
            # Get team dashboard stats
            team_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                team_id=team_id,
                season=season
            )
            
            # Get overall stats (first row of first dataframe)
            overall_stats = team_stats.get_data_frames()[0].iloc[0]
            
            stats_dict = {
                'team_id': team_id,
                'season': season,
                'games_played': overall_stats['GP'],
                'wins': overall_stats['W'],
                'losses': overall_stats['L'],
                'win_pct': overall_stats['W_PCT'],
                'points_per_game': overall_stats['PTS'] / overall_stats['GP'],
                'opp_points_per_game': overall_stats['OPP_PTS'] / overall_stats['GP'],
                'fg_pct': overall_stats['FG_PCT'],
                'fg3_pct': overall_stats['FG3_PCT'],
                'ft_pct': overall_stats['FT_PCT'],
                'rebounds_per_game': overall_stats['REB'] / overall_stats['GP'],
                'assists_per_game': overall_stats['AST'] / overall_stats['GP'],
                'turnovers_per_game': overall_stats['TOV'] / overall_stats['GP']
            }
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error fetching team stats for team {team_id}: {str(e)}")
            return {}
    
    def fetch_recent_games(self, team_id: int, num_games: int = 5) -> pd.DataFrame:
        """
        Fetch recent games for a team.
        
        Args:
            team_id: NBA team ID
            num_games: Number of recent games to fetch
            
        Returns:
            DataFrame with recent games
        """
        if not NBA_API_AVAILABLE:
            return pd.DataFrame()
        
        try:
            # Get team game logs
            game_logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable="2023-24"
            )
            
            games_df = game_logs.get_data_frames()[0]
            
            # Sort by game date and get recent games
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            recent_games = games_df.sort_values('GAME_DATE', ascending=False).head(num_games)
            
            return recent_games
            
        except Exception as e:
            logger.error(f"Error fetching recent games for team {team_id}: {str(e)}")
            return pd.DataFrame()
    
    def download_historical_data(self, save_path: str = None) -> str:
        """
        Download historical NBA data from Kaggle or other free sources.
        
        Args:
            save_path: Path to save the data
            
        Returns:
            Path to the downloaded data file
        """
        if save_path is None:
            save_path = os.path.join(settings.ml.DATA_DIR, "basketball", "nba_games.csv")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # For now, we'll create a sample dataset structure
        # In production, you would download from Kaggle or other sources
        logger.info("Creating sample NBA historical data structure...")
        
        # Sample data structure for NBA games
        sample_data = {
            'GAME_ID': ['0022300001', '0022300002', '0022300003'],
            'GAME_DATE': ['2023-10-18', '2023-10-18', '2023-10-19'],
            'HOME_TEAM_ID': [1610612737, 1610612738, 1610612739],
            'AWAY_TEAM_ID': [1610612740, 1610612741, 1610612742],
            'HOME_TEAM': ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets'],
            'AWAY_TEAM': ['Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers'],
            'HOME_PTS': [120, 115, 108],
            'AWAY_PTS': [110, 118, 105],
            'HOME_FG_PCT': [0.485, 0.462, 0.478],
            'AWAY_FG_PCT': [0.445, 0.489, 0.441],
            'TOTAL_PTS': [230, 233, 213],
            'HOME_WIN': [1, 0, 1]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(save_path, index=False)
        
        logger.info(f"Sample NBA data saved to {save_path}")
        return save_path

    def get_league_standings(self) -> pd.DataFrame:
        """
        Get current NBA league standings.

        Returns:
            DataFrame with league standings
        """
        if not NBA_API_AVAILABLE:
            return pd.DataFrame()

        try:
            standings = leaguestandings.LeagueStandings()
            standings_df = standings.get_data_frames()[0]

            return standings_df

        except Exception as e:
            logger.error(f"Error fetching league standings: {str(e)}")
            return pd.DataFrame()

    def rate_limit_delay(self, delay: float = 0.6):
        """
        Add delay to respect NBA API rate limits.

        Args:
            delay: Delay in seconds
        """
        time.sleep(delay)
