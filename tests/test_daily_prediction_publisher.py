"""Offline tests for protected prediction publication."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from services.daily_prediction_publisher import (
    load_existing, prepare_publication, publish,
)


DAY = date(2026, 10, 25)
NOW = datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc)


def fixture(match_id="A", kickoff="17:30", status="Not Started"):
    return {
        "match_id": match_id,
        "match_date": DAY.isoformat(),
        "match_time": kickoff,
        "match_status": status,
        "match_hometeam_name": "Home " + match_id,
        "match_awayteam_name": "Away " + match_id,
    }


def prediction(match_id="A", confidence=70, quality="HIGH"):
    return {
        "match_id": match_id,
        "home_team": "Home " + match_id,
        "away_team": "Away " + match_id,
        "prediction": "Home Win",
        "confidence": confidence,
        "data_quality": quality,
        "kick_off": "17:30",
        "home_win": 70,
        "draw": 20,
        "away_win": 10,
    }


class PredictionPublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_publication_preserves_ids_and_confidence(self):
        result = publish(DAY, self.cache, [fixture()], [prediction()], NOW)
        self.assertEqual(result["status"], "published")
        self.assertEqual(result["appended_count"], 1)
        saved = load_existing(self.cache / "predictions_2026-10-25.json")
        self.assertEqual(saved[0]["match_id"], "A")
        self.assertTrue(saved[0]["recommendation_eligible"])
        self.assertIn("published_at_utc", saved[0])

    def test_same_fixture_never_overwrites_published_pick(self):
        first = prediction()
        original, stats = prepare_publication([fixture()], [first], [], NOW)
        updated = prediction(confidence=99)
        same, stats2 = prepare_publication([fixture()], [updated], original, NOW)
        self.assertEqual(stats2["appended_count"], 0)
        self.assertEqual(same[0]["confidence"], 70)

    def test_new_fixture_is_added_and_old_preserved(self):
        first = publish(DAY, self.cache, [fixture()], [prediction()], NOW)
        updated = publish(
            DAY, self.cache, [fixture(), fixture("B")],
            [prediction(confidence=55), prediction("B", confidence=65)], NOW,
        )
        self.assertEqual(first["appended_count"], 1)
        self.assertEqual(updated["appended_count"], 1)
        existing = load_existing(self.cache / "predictions_2026-10-25.json")
        self.assertEqual(len(existing), 2)
        self.assertEqual(existing[0]["confidence"], 70)
        self.assertTrue(list((self.cache / "archives").glob("*.json")))

    def test_started_fixture_never_gets_retrospective_prediction(self):
        files, stats = prepare_publication(
            [fixture("A", kickoff="06:00"), fixture("B")],
            [prediction(), prediction("B")], [], NOW,
        )
        self.assertEqual([p["match_id"] for p in files], ["B"])
        self.assertEqual(stats["skipped_started"], 1)

    def test_inplay_status_prevents_new_pick_even_if_future_clock(self):
        with self.assertRaisesRegex(ValueError, "nothing valid"):
            prepare_publication(
                [fixture(status="In Play")], [prediction()], [], NOW
            )

    def test_started_existing_pick_is_immutable(self):
        existing = [prediction(confidence=61)]
        results, details = prepare_publication(
            [fixture(kickoff="06:00")],
            [prediction(confidence=99)],
            existing, NOW,
        )
        self.assertEqual(details["appended_count"], 0)
        self.assertEqual(results[0]["confidence"], 61)

    def test_invalid_confidence_blocks_publication(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            publish(DAY, self.cache, [fixture()], [prediction(confidence=125)], NOW)
        self.assertFalse((self.cache / "predictions_2026-10-25.json").exists())

    def test_team_mismatch_blocks_publication(self):
        bad = prediction()
        bad["away_team"] = "Other Team"
        with self.assertRaisesRegex(ValueError, "team identity mismatch"):
            publish(DAY, self.cache, [fixture()], [bad], NOW)

    def test_duplicate_model_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            publish(DAY, self.cache, [fixture()],
                    [prediction(), prediction()], NOW)

    def test_partial_upcoming_output_blocks_publication(self):
        with self.assertRaisesRegex(ValueError, "omitted upcoming"):
            publish(DAY, self.cache,
                    [fixture("A"), fixture("B")], [prediction()], NOW)

    def test_all_bad_data_still_tagged_as_ineligible(self):
        results, stats = prepare_publication(
            [fixture()], [prediction(confidence=90, quality="LOW")], [], NOW
        )
        self.assertFalse(results[0]["recommendation_eligible"])
        self.assertEqual(stats["eligible_count"], 0)

    def test_corrupted_existing_file_is_not_overwritten(self):
        path = self.cache / "predictions_2026-10-25.json"
        path.write_text('{"bad":"cache"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a list"):
            publish(DAY, self.cache, [fixture()], [prediction()], NOW)
        self.assertEqual(path.read_text(), '{"bad":"cache"}')

    def test_model_failure_shape_cannot_blank_last_good_card(self):
        path = self.cache / "predictions_2026-10-25.json"
        publish(DAY, self.cache, [fixture()], [prediction()], NOW)
        previous = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "zero predictions"):
            publish(DAY, self.cache, [fixture()], [], NOW)
        self.assertEqual(path.read_bytes(), previous)

    def test_unchanged_run_does_not_write_or_archive(self):
        path = self.cache / "predictions_2026-10-25.json"
        publish(DAY, self.cache, [fixture()], [prediction()], NOW)
        previous = path.read_bytes()
        outcome = publish(DAY, self.cache, [fixture()],
                          [prediction(confidence=97)], NOW)
        self.assertEqual(outcome["status"], "unchanged")
        self.assertEqual(previous, path.read_bytes())
        self.assertFalse((self.cache / "archives").exists())

    def test_prediction_without_fixture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model fixture"):
            publish(DAY, self.cache, [fixture()], [prediction("X")], NOW)


if __name__ == "__main__":
    unittest.main()
