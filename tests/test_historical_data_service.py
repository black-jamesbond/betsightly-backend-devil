"""
Tests for HistoricalDataService — the core fix that makes ML predictions meaningful.
"""

import pytest
from datetime import datetime


@pytest.fixture(scope="module")
def hist_df():
    """Load the real historical dataset once per test session."""
    from services.historical_data_service import HistoricalDataService
    svc = HistoricalDataService()
    df = svc.get_historical_data()
    if df is None:
        pytest.skip("Historical dataset not available (data/github-football/Matches.csv missing)")
    return df


def test_required_columns_present(hist_df):
    for col in ("home_team", "away_team", "date", "home_score", "away_score"):
        assert col in hist_df.columns, f"Missing column: {col}"


def test_dataset_size(hist_df):
    assert len(hist_df) > 100_000, "Expected >100K historical matches"


def test_dates_are_datetime(hist_df):
    import pandas as pd
    assert pd.api.types.is_datetime64_any_dtype(hist_df["date"])


def test_scores_are_integers(hist_df):
    assert hist_df["home_score"].dtype in ("int64", "int32")
    assert hist_df["away_score"].dtype in ("int64", "int32")


def test_big_clubs_have_history(hist_df):
    big_clubs = [
        "Arsenal", "Liverpool", "Barcelona", "Real Madrid",
        "Bayern Munich", "Paris SG", "Juventus",
    ]
    for club in big_clubs:
        count = ((hist_df["home_team"] == club) | (hist_df["away_team"] == club)).sum()
        assert count > 100, f"{club} has only {count} matches — expected >100"


def test_team_alias_normalization():
    from services.historical_data_service import HistoricalDataService
    svc = HistoricalDataService()
    assert svc.normalize_team_name("Paris Saint-Germain") == "Paris SG"
    assert svc.normalize_team_name("Inter Milan") == "Inter"
    assert svc.normalize_team_name("Unknown FC") == "Unknown FC"  # passthrough


def test_form_lookup_returns_real_data(hist_df):
    """Simulate _calculate_team_form to confirm real stats come back."""
    team = "Arsenal"
    cutoff = datetime(2025, 1, 1)
    recent = hist_df[
        ((hist_df["home_team"] == team) | (hist_df["away_team"] == team))
        & (hist_df["date"] < cutoff)
    ].sort_values("date", ascending=False).head(10)

    assert len(recent) == 10
    goals_scored = sum(
        row["home_score"] if row["home_team"] == team else row["away_score"]
        for _, row in recent.iterrows()
    )
    assert goals_scored > 0


def test_h2h_lookup(hist_df):
    h1, h2 = "Barcelona", "Real Madrid"
    cutoff = datetime(2025, 1, 1)
    h2h = hist_df[
        (
            ((hist_df["home_team"] == h1) & (hist_df["away_team"] == h2))
            | ((hist_df["home_team"] == h2) & (hist_df["away_team"] == h1))
        )
        & (hist_df["date"] < cutoff)
    ]
    assert len(h2h) >= 10, f"Expected ≥10 Clásico matches, got {len(h2h)}"


def test_season_column_derived(hist_df):
    assert "season" in hist_df.columns
    # All seasons should be 4-digit year strings
    sample = hist_df["season"].dropna().head(100)
    assert all(s.isdigit() and len(s) == 4 for s in sample)
