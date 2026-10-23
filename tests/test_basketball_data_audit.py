"""Tests for the no-network NBA historical data readiness audit."""
import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from scripts.audit_basketball_data import evaluate_csv

FIELDS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "MATCHUP",
          "WL", "PTS", "SEASON"]


class BasketballDataAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "nba.csv"
        self.today = date(2026, 10, 23)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, *, day=None, one_team=False, invalid_id=False):
        when = (day or self.today - timedelta(days=5)).isoformat()
        rows = [
            ["1", when, "A", "Team A", "A vs. B", "W", "101", "2026-27"],
            ["1", when, "B", "Team B", "B @ A", "L", "99", "2026-27"],
        ]
        if one_team:
            rows.pop()
        if invalid_id:
            rows[-1][2] = "A"
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(rows)

    def audit(self):
        return evaluate_csv(self.path, today=self.today,
                            min_games=1, min_seasons=1, max_lag_days=90)

    def test_valid_pair_is_structurally_ready(self):
        self.write()
        result = self.audit()
        self.assertEqual(result["status"], "READY_FOR_FEATURE_REVIEW")
        self.assertEqual(result["games"], 1)
        self.assertEqual(result["invalid_games"], 0)

    def test_stale_data_is_not_ready(self):
        self.write(day=date(2024, 4, 14))
        result = self.audit()
        self.assertEqual(result["status"], "NOT_READY")
        self.assertIn("STALE_SOURCE", result["findings"])

    def test_missing_second_team_rejected(self):
        self.write(one_team=True)
        self.assertEqual(self.audit()["invalid_games"], 1)

    def test_duplicate_team_id_rejected(self):
        self.write(invalid_id=True)
        self.assertEqual(self.audit()["invalid_games"], 1)

    def test_missing_file_rejected(self):
        self.assertEqual(self.audit()["findings"], ["MISSING_SOURCE_FILE"])

    def test_future_dated_history_rejected(self):
        self.write(day=self.today + timedelta(days=1))
        self.assertIn("FUTURE_GAMES_IN_HISTORY", self.audit()["findings"])


if __name__ == "__main__":
    unittest.main()
