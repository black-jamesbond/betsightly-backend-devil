"""
Historical Data Service

Loads and normalizes the GitHub football dataset for use by the ML feature
engineering pipeline.

The dataset lives at data/github-football/Matches.csv and contains 228K
matches (2000-2025) across 1,214 teams with real odds, ELO ratings, and
pre-computed form values.

Normalization maps the raw CSV columns to the standard names expected by
FootballFeatureEngineer:
    home_team, away_team, date, home_score, away_score, competition_name

A name-normalization step handles common mismatches between API team names
(e.g. "Paris Saint-Germain") and dataset names (e.g. "Paris SG").
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: GitHub dataset → internal standard
# ---------------------------------------------------------------------------
_COLUMN_MAP = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "MatchDate": "date",
    "FTHome": "home_score",
    "FTAway": "away_score",
    "Division": "competition_name",
    # Keep useful extras under their original names
    "HomeElo": "home_elo",
    "AwayElo": "away_elo",
    "Form3Home": "form3_home",
    "Form5Home": "form5_home",
    "Form3Away": "form3_away",
    "Form5Away": "form5_away",
    "OddHome": "odd_home",
    "OddDraw": "odd_draw",
    "OddAway": "odd_away",
    "Over25": "odd_over25",
    "Under25": "odd_under25",
    "FTResult": "result",
}

# Common API name → dataset name aliases
# Add more as you discover mismatches in production logs
_TEAM_ALIASES: dict[str, str] = {
    "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG",
    "Atletico Madrid": "Atlético Madrid",
    "Atletico de Madrid": "Atlético Madrid",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "RB Leipzig": "Red Bull Leipzig",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Newcastle United": "Newcastle",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Sheffield United": "Sheffield Utd",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Köln": "FC Köln",
    "FC Bayern München": "Bayern Munich",
    "Bayer Leverkusen": "Leverkusen",
    "Sporting CP": "Sporting",
    "Sporting Lisbon": "Sporting",
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Real Sociedad": "Sociedad",
    "Athletic Bilbao": "Ath Bilbao",
    "Athletic Club": "Ath Bilbao",
    "Deportivo Alavés": "Alaves",
    "Rayo Vallecano": "Vallecano",
    "Celta Vigo": "Celta",
    "Girona FC": "Girona",
    "Getafe CF": "Getafe",
    "Cadiz CF": "Cadiz",
    "Almería": "Almeria",
    "Las Palmas": "Las Palmas",
    "UD Las Palmas": "Las Palmas",
}


class HistoricalDataService:
    """
    Loads, normalises, and caches the GitHub football dataset.

    Singleton pattern via module-level instance — the CSV is only read once
    per process lifetime, then held in memory.
    """

    _DATA_PATH = Path("data/github-football/Matches.csv")
    _REQUIRED_COLS = {"home_team", "away_team", "date", "home_score", "away_score"}

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Return the normalised historical DataFrame, loading it on first call.
        Returns None if the file cannot be read.
        """
        if self._df is not None:
            return self._df
        self._df = self._load()
        return self._df

    def normalize_team_name(self, name: str) -> str:
        """Map an API-returned team name to the dataset's naming convention."""
        return _TEAM_ALIASES.get(name, name)

    def get_team_coverage(self) -> dict:
        """Return stats on which teams have history in the dataset."""
        df = self.get_historical_data()
        if df is None:
            return {}
        teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
        return {
            "total_teams": len(teams),
            "total_matches": len(df),
            "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "sample_teams": sorted(list(teams))[:20],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> Optional[pd.DataFrame]:
        path = self._DATA_PATH
        if not path.exists():
            # Try relative to this file's location (handles different CWDs)
            alt = Path(__file__).parent.parent / self._DATA_PATH
            if alt.exists():
                path = alt
            else:
                logger.error(
                    f"Historical dataset not found at {self._DATA_PATH}. "
                    "Real feature engineering will fall back to heuristics."
                )
                return None

        logger.info(f"Loading historical dataset from {path} …")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read historical dataset: {e}")
            return None

        df = self._normalise(df)
        if df is None:
            return None

        logger.info(
            f"Historical dataset loaded: {len(df):,} matches, "
            f"{df['home_team'].nunique()} teams, "
            f"{df['date'].min().date()} – {df['date'].max().date()}"
        )
        return df

    def _normalise(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Rename columns, parse types, drop rows with missing essentials."""
        # Rename known columns
        df = df.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in df.columns})

        # Verify required columns exist
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            logger.error(f"Historical dataset missing required columns: {missing}")
            return None

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Parse scores to numeric
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df.dropna(subset=["home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # Strip whitespace from team names
        df["home_team"] = df["home_team"].str.strip()
        df["away_team"] = df["away_team"].str.strip()

        # Drop rows with missing team names
        df = df.dropna(subset=["home_team", "away_team"])

        # Derive season from date (football season straddles two years;
        # use the calendar year of August = season start)
        if "season" not in df.columns:
            df["season"] = df["date"].apply(
                lambda d: str(d.year) if d.month >= 8 else str(d.year - 1)
            )

        # Ensure competition_name exists
        if "competition_name" not in df.columns:
            df["competition_name"] = "Unknown"

        # Sort by date ascending (required for form lookups)
        df = df.sort_values("date").reset_index(drop=True)

        return df


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
historical_data_service = HistoricalDataService()
