"""Offline jackpot round freshness and validation regressions."""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import sportybet_jackpot as jp


NOW = datetime(2026, 10, 23, 14, 0, tzinfo=ZoneInfo("Africa/Lagos"))


def payload(now=NOW):
    return {
        "round": "20261025", "fetched": now.isoformat(),
        "games": [
            {"n": i, "home": "Home " + str(i), "away": "Away " + str(i),
             "date": "2026-10-25", "time": "14:00",
             "odds": [2.5, 3.1, 2.8]}
            for i in range(1, 13)
        ],
    }


class JackpotFreshnessTests(unittest.TestCase):
    def test_fresh_12_games_is_usable(self):
        self.assertTrue(jp.snapshot_freshness(payload(), NOW)["fresh"])

    def test_snapshot_older_than_week_is_rejected(self):
        data = payload(NOW - timedelta(days=8))
        self.assertFalse(jp.snapshot_freshness(data, NOW)["fresh"])

    def test_missing_fetched_at_is_not_assumed_current(self):
        data = payload()
        data.pop("fetched")
        self.assertFalse(jp.snapshot_freshness(data, NOW)["fresh"])

    def test_incomplete_round_is_rejected(self):
        data = payload()
        data["games"] = data["games"][:-1]
        self.assertFalse(jp.snapshot_freshness(data, NOW)["fresh"])

    def test_invalid_manual_snapshot_does_not_overwrite_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jackpot.json"
            with patch.object(jp, "CACHE", path):
                valid = payload()
                jp.save_snapshot(valid)
                original = path.read_bytes()
                invalid = payload()
                invalid["games"].pop()
                with self.assertRaises(ValueError):
                    jp.save_snapshot(invalid)
                self.assertEqual(path.read_bytes(), original)

    def test_duplicate_game_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(jp, "CACHE", Path(tmp) / "jackpot.json"):
                invalid = payload()
                invalid["games"][1]["home"] = invalid["games"][0]["home"]
                invalid["games"][1]["away"] = invalid["games"][0]["away"]
                with self.assertRaises(ValueError):
                    jp.save_snapshot(invalid)


if __name__ == "__main__":
    unittest.main()
