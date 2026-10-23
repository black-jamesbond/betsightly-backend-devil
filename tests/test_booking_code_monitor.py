"""Offline verification that changed bookmaker prices trigger code renewals."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sportybet_betting as sbb
from services.booking_code_monitor import refresh_tracked_codes
from tests.test_bookmaker_verification import NOW, event, selection


def readback_payload(code):
    return {"bizCode": 10000, "data": {"shareCode": code,
        "unavailableOutcomes": [], "outcomes": [{
            "eventId": "sr:match:1",
            "markets": [{"id": "18", "specifier": "total=1.5",
                         "outcomes": [{"id": "12"}]}],
        }]}}


class BookingCodeMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "codes.json"
        self.override = patch.object(sbb, "CODE_CACHE", self.path)
        self.override.start()
        self.calls = []

    def tearDown(self):
        self.override.stop()
        self.tmp.cleanup()

    def _issue(self, rows):
        self.calls.append(rows)
        return {"share_code": "ABC" + str(len(self.calls)),
                "share_url": "https://www.sportybet.com/ng/"}

    def test_code_cache_invalidated_when_odds_change(self):
        q1 = __import__("services.bookmaker_verification",
                        fromlist=["verify_and_reprice"]).verify_and_reprice(
                            [selection()], [event()], now_ms=NOW
                        )
        q2 = __import__("services.bookmaker_verification",
                        fromlist=["verify_and_reprice"]).verify_and_reprice(
                            [selection()], [event(odds=1.44)], now_ms=NOW + 30_000
                        )
        with patch.object(sbb, "create_booking_code",
                          side_effect=self._issue), patch.object(
                              sbb, "read_booking_code",
                              side_effect=lambda code: readback_payload(code)
                          ):
            with patch.object(sbb.time, "time", return_value=NOW / 1000):
                old = sbb.booking_code_for(
                    q1["selections"], quote_signature=q1["quote_signature"],
                    deadline_ms=q1["deadline_ms"])
                repeated = sbb.booking_code_for(
                    q1["selections"], quote_signature=q1["quote_signature"],
                    deadline_ms=q1["deadline_ms"])
            with patch.object(sbb.time, "time", return_value=(NOW + 30_000) / 1000):
                new = sbb.booking_code_for(
                    q2["selections"], quote_signature=q2["quote_signature"],
                    deadline_ms=q2["deadline_ms"])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(repeated["cached"])
        self.assertNotEqual(old["share_code"], new["share_code"])
        self.assertEqual(new["tracking_id"], old["tracking_id"])

    def test_legacy_cached_code_is_not_reused(self):
        self.path.write_text(
            '{"sr:match:1|18|total=1.5|12":{"share_code":"OLD"}}'
        )
        from services.bookmaker_verification import verify_and_reprice
        q = verify_and_reprice([selection()], [event()], now_ms=NOW)
        with patch.object(sbb, "create_booking_code",
                          side_effect=self._issue), patch.object(
                              sbb, "read_booking_code",
                              side_effect=lambda code: readback_payload(code)
                          ), patch.object(
                              sbb.time, "time", return_value=NOW / 1000
                          ):
            new = sbb.booking_code_for(
                q["selections"], quote_signature=q["quote_signature"],
                deadline_ms=q["deadline_ms"],
            )
        self.assertFalse(new["cached"])
        self.assertEqual(new["share_code"], "ABC1")

    def test_watcher_renews_changed_quote(self):
        from services.bookmaker_verification import verify_and_reprice
        original = verify_and_reprice([selection()], [event()], now_ms=NOW)
        entry = {
            "selections": original["selections"],
            "quote_signature": original["quote_signature"],
            "share_code": "OLD", "deadline_ms": original["deadline_ms"],
        }
        def issue(quote):
            self.calls.append(quote)
            return {"share_code": "NEW"}
        result = refresh_tracked_codes(
            {"ticket": entry}, [event(odds=1.51)], issue_code=issue,
            now_ms=NOW + 10_000,
        )
        self.assertEqual(result["renewed"], 1)
        self.assertEqual(result["tickets"]["ticket"]["share_code"], "NEW")
        self.assertEqual(len(self.calls), 1)

    def test_watcher_does_not_issue_if_quote_unchanged(self):
        from services.bookmaker_verification import verify_and_reprice
        original = verify_and_reprice([selection()], [event()], now_ms=NOW)
        entry = {
            "selections": original["selections"],
            "quote_signature": original["quote_signature"],
            "share_code": "OLD", "deadline_ms": original["deadline_ms"],
        }
        result = refresh_tracked_codes(
            {"ticket": entry}, [event()], issue_code=lambda _: 1 / 0,
            now_ms=NOW + 1_000,
        )
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(result["renewed"], 0)

    def test_watcher_cannot_renew_started_match(self):
        from services.bookmaker_verification import verify_and_reprice
        old = verify_and_reprice([selection()], [event()], now_ms=NOW)
        entry = {
            "selections": old["selections"],
            "quote_signature": old["quote_signature"],
            "share_code": "OLD", "deadline_ms": old["deadline_ms"],
        }
        result = refresh_tracked_codes(
            {"ticket": entry}, [event(kickoff=NOW)], issue_code=lambda _: 1 / 0,
            now_ms=NOW + 100,
        )
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual(result["renewed"], 0)

    def test_concurrent_requests_share_one_code_and_preserve_cache(self):
        from concurrent.futures import ThreadPoolExecutor
        from services.bookmaker_verification import verify_and_reprice
        quote = verify_and_reprice([selection()], [event()], now_ms=NOW)
        with patch.object(sbb, "create_booking_code",
                          side_effect=self._issue), patch.object(
                              sbb, "read_booking_code",
                              side_effect=lambda code: readback_payload(code)
                          ), patch.object(sbb.time, "time", return_value=NOW / 1000):
            def request_code(_):
                return sbb.booking_code_for(
                    quote["selections"],
                    quote_signature=quote["quote_signature"],
                    deadline_ms=quote["deadline_ms"],
                )
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(request_code, range(10)))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual({r["share_code"] for r in results}, {"ABC1"})
        import json
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 1)

    def test_find_event_reversed_is_not_accepted(self):
        idx = sbb.build_index([event()])
        actual, kind = sbb.find_event(idx, "Away FC", "Home FC")
        self.assertIsNone(actual)
        self.assertIsNone(kind)

    def test_duplicate_pair_different_event_ids_is_ambiguous(self):
        idx = sbb.build_index([event("sr:match:1"), event("sr:match:2")])
        found, kind = sbb.find_event(idx, "Home FC", "Away FC")
        self.assertIsNone(found)
        self.assertIsNone(kind)

    def test_missing_board_kickoff_rejected(self):
        from build_accas import event_matches_leg
        leg = {"kick_off_utc": "15:00"}
        self.assertFalse(event_matches_leg(event(kickoff=None), leg))

    def test_under_price_validated(self):
        from services.bookmaker_verification import verify_and_reprice
        s = selection()
        s["outcome_id"] = "13"
        q = verify_and_reprice([s], [event()], now_ms=NOW)
        self.assertEqual(q["odds"], 3.2)


if __name__ == "__main__":
    unittest.main()
