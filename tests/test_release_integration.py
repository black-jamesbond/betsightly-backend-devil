"""Cross-batch release contracts: fixtures -> model card -> 90m settlement.

No real network, credentials, model training, or bookmaker calls.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
import unittest

from services.weekly_fixture_collector import connect, collect_day, status_summary
from services.daily_prediction_publisher import load_fixtures, load_existing, publish
from services.settlement_service import init, sync_date


DAY = date(2026, 10, 25)
BEFORE = datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc)


def fixture(match_id, state="Not Started", *, home="Home FC", away="Away FC",
            kickoff="18:30", h_score="", a_score=""):
    return {
        "match_id": match_id, "match_date": DAY.isoformat(),
        "match_time": kickoff, "match_status": state,
        "match_hometeam_name": home, "match_awayteam_name": away,
        "match_hometeam_score": h_score, "match_awayteam_score": a_score,
        "match_hometeam_ft_score": h_score if state == "Finished" else "",
        "match_awayteam_ft_score": a_score if state == "Finished" else "",
    }


def prediction(match_id):
    return {
        "match_id": match_id, "home_team": "Home FC", "away_team": "Away FC",
        "prediction": "Home Win", "confidence": 72,
        "data_quality": "HIGH", "home_win": 72,
        "draw": 15, "away_win": 13,
        "over_1.5": 75, "over_2.5": 58, "btts": 54,
        "kick_off": "18:30", "match_date": DAY.isoformat(),
    }


class EndToEndOfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = connect(self.root / "data/fixtures/fixture_store.sqlite3")
        init(self.db)
        self.cache = self.root / "cache"

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_fixture_import_publish_settle_replay(self):
        source = lambda _: [fixture("fixture-123")]
        got = collect_day(self.db, DAY, fetcher=source,
                          cache_dir=self.cache, publish_cache=True)
        self.assertEqual(got["status"], "success")
        current = load_fixtures(self.db, DAY)
        self.assertEqual([f["match_id"] for f in current], ["fixture-123"])

        released = publish(DAY, self.cache, current,
                           [prediction("fixture-123")], BEFORE)
        self.assertEqual(released["appended_count"], 1)
        frozen = load_existing(self.cache / "predictions_2026-10-25.json")
        self.assertTrue(frozen[0]["published_at_utc"])
        before_bytes = (self.cache / "predictions_2026-10-25.json").read_bytes()

        finished = fixture("fixture-123", "Finished", h_score="2", a_score="1")
        updated = collect_day(
            self.db, DAY, fetcher=lambda _: [finished], cache_dir=self.cache,
            publish_cache=True)
        self.assertEqual(updated["status"], "success")
        outcome = sync_date(self.db, DAY, [finished], frozen)
        self.assertEqual(outcome["selections"]["prediction"]["WIN"], 1)
        self.assertEqual(outcome["fixtures"]["finished"], 1)
        second = sync_date(self.db, DAY, [finished], frozen)
        self.assertEqual(second["selections"]["prediction"]["WIN"], 1)
        self.assertEqual((self.cache / "predictions_2026-10-25.json").read_bytes(),
                         before_bytes)

    def test_withdrawal_before_first_card_never_publishes_removed_game(self):
        collect_day(self.db, DAY, fetcher=lambda d: [
            fixture("upcoming"), fixture("withdrawn")
        ], cache_dir=self.cache, publish_cache=True)
        collect_day(self.db, DAY, fetcher=lambda d: [fixture("upcoming")],
                    cache_dir=self.cache, publish_cache=True)
        self.assertEqual({p["match_id"] for p in load_fixtures(self.db, DAY)},
                         {"upcoming"})
        result = publish(DAY, self.cache, load_fixtures(self.db, DAY),
                         [prediction("upcoming")], BEFORE)
        self.assertEqual(result["published_count"], 1)
        self.assertEqual(len(self.db.execute(
            "SELECT * FROM collected_fixtures").fetchall()), 2)

    def test_provider_failure_does_not_erase_predictions_or_active_board(self):
        collect_day(self.db, DAY, fetcher=lambda _: [fixture("existing")],
                    cache_dir=self.cache, publish_cache=True)
        publish(DAY, self.cache, load_fixtures(self.db, DAY),
                [prediction("existing")], BEFORE)
        before = (self.cache / "predictions_2026-10-25.json").read_bytes()
        failed = collect_day(self.db, DAY, fetcher=lambda _: [],
                             attempts=1, publish_cache=True,
                             cache_dir=self.cache)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(status_summary(self.db, DAY)["counts"][DAY.isoformat()], 1)
        self.assertEqual((self.cache / "predictions_2026-10-25.json").read_bytes(),
                         before)


if __name__ == "__main__":
    unittest.main()
