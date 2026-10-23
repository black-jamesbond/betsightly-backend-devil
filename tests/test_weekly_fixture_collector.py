"""Offline regression coverage for the independent weekly fixture collector."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from services.weekly_fixture_collector import (
    collect_day, collect_week, connect, current_sunday, next_sunday, status_summary,
)


def fixture(day: str, match_id: str = "one", status: str = "Not Started") -> dict:
    return {
        "match_id": match_id,
        "match_date": day,
        "match_time": "18:30",
        "match_status": status,
        "match_hometeam_name": "Home FC",
        "match_awayteam_name": "Away FC",
        "league_name": "Test League",
        "country_name": "Testland",
    }


class FixtureCollectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.db = connect(root / "fixtures.sqlite3")
        self.cache = root / "cache"
        self.day = date(2026, 10, 25)

    def tearDown(self):
        self.db.close()
        self.directory.cleanup()

    def _stored(self):
        return self.db.execute(
            "SELECT source_id,status,payload_json FROM collected_fixtures"
        ).fetchall()

    def test_sunday_window(self):
        self.assertEqual(next_sunday(date(2026, 10, 23)), date(2026, 10, 25))
        self.assertEqual(next_sunday(date(2026, 10, 25)), date(2026, 10, 25))
        self.assertEqual(current_sunday(date(2026, 10, 26)), date(2026, 10, 25))

    def test_week_fetches_exactly_seven_dates(self):
        seen = []
        def provider(day):
            seen.append(day)
            return [fixture(day, match_id=day)]
        result = collect_week(self.db, self.day, fetcher=provider)
        self.assertEqual(len(result), 7)
        self.assertEqual(seen[0], "2026-10-25")
        self.assertEqual(seen[-1], "2026-10-31")
        self.assertTrue(all(day["status"] == "success" for day in result))
        self.assertEqual(len(self._stored()), 7)
        summary = status_summary(self.db, self.day)
        self.assertEqual(set(summary["counts"].values()), {1})

    def test_repeated_run_upserts_without_duplicates(self):
        source = lambda day: [fixture(day, match_id="123")]
        for _ in range(3):
            self.assertEqual(
                collect_day(self.db, self.day, fetcher=source)["status"],
                "success",
            )
        self.assertEqual(len(self._stored()), 1)

    def test_in_play_and_final_score_updates(self):
        def provider(status):
            record = fixture(self.day.isoformat(), "x", status)
            record["match_hometeam_score"] = "2" if status == "Finished" else ""
            return lambda day: [record]
        collect_day(self.db, self.day, fetcher=provider("Not Started"))
        collect_day(self.db, self.day, fetcher=provider("In Play"))
        collect_day(self.db, self.day, fetcher=provider("Finished"))
        self.assertEqual(self._stored()[0]["status"], "Finished")
        collect_day(self.db, self.day, fetcher=provider("Not Started"))
        saved = json.loads(self._stored()[0]["payload_json"])
        self.assertEqual(saved["match_status"], "Finished")
        self.assertEqual(saved["match_hometeam_score"], "2")

    def test_source_failure_keeps_old_snapshot(self):
        original = lambda day: [fixture(day)]
        collect_day(self.db, self.day, fetcher=original,
                    publish_cache=True, cache_dir=self.cache)
        target = self.cache / "events_2026-10-25.json"
        first = target.read_bytes()
        def broken(day):
            raise RuntimeError("feed offline")
        outcome = collect_day(self.db, self.day, fetcher=broken, attempts=1,
                              publish_cache=True, cache_dir=self.cache)
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(target.read_bytes(), first)
        self.assertEqual(len(self._stored()), 1)
        self.assertEqual(status_summary(self.db, self.day)["latest_run"]["status"], "error")

    def test_empty_feed_does_not_clear_saved_data(self):
        collect_day(self.db, self.day,
                    fetcher=lambda day: [fixture(day)],
                    publish_cache=True, cache_dir=self.cache)
        first = (self.cache / "events_2026-10-25.json").read_bytes()
        result = collect_day(self.db, self.day, fetcher=lambda _: [],
                             attempts=1, publish_cache=True,
                             cache_dir=self.cache)
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(self._stored()), 1)
        self.assertEqual((self.cache / "events_2026-10-25.json").read_bytes(), first)

    def test_invalid_and_duplicate_rows_are_rejected(self):
        def provider(day):
            invalid = fixture("2026-10-24", "off-day")
            missing_id = fixture(day, "")
            return [fixture(day, "valid"), fixture(day, "valid"),
                    invalid, missing_id]
        result = collect_day(self.db, self.day, fetcher=provider)
        self.assertEqual(result["fetched"], 4)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected"], 3)
        self.assertEqual(len(self._stored()), 1)

    def test_invalid_only_does_not_overwrite(self):
        result = collect_day(
            self.db, self.day,
            fetcher=lambda _: [{"match_id": "x", "match_date": "bad"}],
            attempts=1, publish_cache=True, cache_dir=self.cache,
        )
        self.assertEqual(result["status"], "error")
        self.assertFalse((self.cache / "events_2026-10-25.json").exists())

    def test_partial_week_records_failures_without_erasing_other_days(self):
        def provider(day):
            if day == "2026-10-27":
                raise ConnectionError("temporary upstream outage")
            return [fixture(day, match_id=day)]
        results = collect_week(self.db, self.day, fetcher=provider)
        self.assertEqual(len(results), 7)
        self.assertEqual(sum(r["status"] == "success" for r in results), 6)
        self.assertEqual(len(self._stored()), 6)
        stored_runs = self.db.execute("SELECT COUNT(*) FROM fixture_import_runs").fetchone()[0]
        self.assertEqual(stored_runs, 7)

    def test_atomic_cache_has_compatible_payload(self):
        result = collect_day(self.db, self.day,
                             fetcher=lambda day: [fixture(day, "abc")],
                             cache_dir=self.cache, publish_cache=True)
        self.assertEqual(result["status"], "success")
        data = json.loads((self.cache / "events_2026-10-25.json").read_text())
        self.assertEqual(data[0]["match_id"], "abc")
        self.assertEqual(data[0]["source"], "livescore")
        self.assertEqual(list(self.cache.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
