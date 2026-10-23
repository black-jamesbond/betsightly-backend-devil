"""Regression tests for idempotent, conservative fixture and ticket settlement."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from services.settlement_service import (
    init, normalize_final, record_selection, settle_fixture, settle_market,
    settlement_report, sync_date,
)

DAY = "2026-10-23"


def fixture(match_id="abc", status="Finished", home="2", away="1"):
    return {
        "match_id": match_id, "match_date": DAY,
        "match_status": status, "match_hometeam_score": home,
        "match_awayteam_score": away,
        "match_hometeam_ft_score": home if status == "Finished" else "",
        "match_awayteam_ft_score": away if status == "Finished" else "",
    }


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = sqlite3.connect(Path(self.tmp.name) / "results.db")
        self.db.row_factory = sqlite3.Row
        init(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_finished_market_results(self):
        f = normalize_final(fixture())
        self.assertEqual(settle_market("Home Win", f), "WIN")
        self.assertEqual(settle_market("Draw", f), "LOSS")
        self.assertEqual(settle_market("Over 2.5", f), "WIN")
        self.assertEqual(settle_market("Under 2.5", f), "LOSS")
        self.assertEqual(settle_market("BTTS Yes", f), "WIN")

    def test_no_live_score_settlement(self):
        row = fixture(status="In Play", home="2", away="1")
        self.assertEqual(normalize_final(row)["status"], "pending")
        self.assertEqual(settle_market("Over 1.5", normalize_final(row)), "PENDING")

    def test_awarded_and_cancelled_are_void(self):
        for status in ("Cancelled", "Postponed", "Abandoned", "Awarded"):
            with self.subTest(status=status):
                result = normalize_final(fixture(status=status))
                self.assertEqual(result["status"], "void")
                self.assertEqual(settle_market("Over 2.5", result), "VOID")

    def test_extra_time_requires_explicit_ninety_minute_score(self):
        row = fixture(status="After ET")
        self.assertEqual(normalize_final(row)["status"], "pending")
        row["match_hometeam_ft_score"] = "0"
        row["match_awayteam_ft_score"] = "0"
        result = normalize_final(row)
        self.assertEqual(settle_market("Draw", result), "WIN")

    def test_raw_extra_time_phase_cannot_settle_as_ninety_minute(self):
        row = fixture(status="Finished", home="3", away="2")
        row["match_raw_status"] = "AET"
        row["match_hometeam_ft_score"] = ""
        row["match_awayteam_ft_score"] = ""
        self.assertEqual(normalize_final(row)["status"], "pending")

    def test_missing_or_invalid_score_remains_pending(self):
        for val in ("", None, "-1", "3.5", "unknown"):
            with self.subTest(val=val):
                row = fixture(home=val, away="0")
                row["match_hometeam_ft_score"] = val
                self.assertEqual(normalize_final(row)["status"], "pending")

    def test_replaying_same_result_keeps_one_row(self):
        settle_fixture(self.db, fixture())
        settle_fixture(self.db, fixture())
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) FROM settled_fixtures").fetchone()[0], 1
        )

    def test_stale_feed_does_not_revert_final(self):
        settle_fixture(self.db, fixture())
        settle_fixture(self.db, fixture(status="In Play"))
        row = self.db.execute("SELECT * FROM settled_fixtures").fetchone()
        self.assertEqual(row["result_status"], "finished")
        self.assertEqual(row["home_score"], 2)

    def test_conflicting_final_requires_manual_review(self):
        settle_fixture(self.db, fixture())
        with self.assertRaisesRegex(ValueError, "conflicting"):
            settle_fixture(self.db, fixture(home="0", away="1"))

    def test_void_selection_does_not_count_as_loss(self):
        settle_fixture(self.db, fixture(status="Cancelled"))
        self.assertEqual(
            record_selection(self.db, ticket_kind="prediction", ticket_id="t",
                             leg_id="1", match_date=DAY, match_id="abc",
                             market="Home Win"), "VOID"
        )
        summary = settlement_report(self.db, DAY)
        self.assertEqual(summary["selections"]["prediction"]["VOID"], 1)
        self.assertIsNone(summary["selections"]["prediction"]["win_rate_pct"])

    def test_empty_fixture_predictions_stay_pending(self):
        outcome = record_selection(
            self.db, ticket_kind="prediction", ticket_id="t",
            leg_id="1", match_date=DAY, match_id="unknown",
            market="Home Win",
        )
        self.assertEqual(outcome, "PENDING")

    def test_full_sync_predictions_and_bots(self):
        pred = [{"match_id": "abc", "prediction": "Home Win"}]
        bot = {"bots": [{"bot": "Over bot", "history": [
            {"match_id": "abc", "selection": "Over 1.5", "status": "PENDING"}
        ]}]}
        result = sync_date(self.db, DAY, [fixture()], pred, bot)
        self.assertEqual(result["fixtures"]["finished"], 1)
        self.assertEqual(result["selections"]["prediction"]["WIN"], 1)
        self.assertEqual(result["selections"]["bot"]["WIN"], 1)
        result2 = sync_date(self.db, DAY, [fixture()], pred, bot)
        self.assertEqual(result2["selections"]["prediction"]["WIN"], 1)
        self.assertEqual(result2["selections"]["bot"]["WIN"], 1)

    def test_wrong_score_is_not_zero(self):
        row = fixture(home="oops", away="1")
        self.assertEqual(normalize_final(row)["status"], "pending")


if __name__ == "__main__":
    unittest.main()
