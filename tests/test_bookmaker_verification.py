"""No-network regression tests for exact SportyBet booking verification."""
import unittest
from datetime import datetime, timezone

from services.bookmaker_verification import (
    BookingValidationError, ensure_not_expired, quote_is_recent, verify_and_reprice,
)


NOW = int(datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
KICKOFF = int(datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)


def event(event_id="sr:match:1", *, odds=1.35, kickoff=KICKOFF,
          home="Home FC", away="Away FC", status="Not start", live=False):
    return {
        "event_id": event_id, "home": home, "away": away, "start_ms": kickoff,
        "status": status, "booking": "", "live": live,
        "prices": {
            "over_1.5": {
                "market_id": "18", "specifier": "total=1.5",
                "outcome_id": "12", "odds": odds,
                "under_outcome_id": "13", "under_odds": 3.2,
            },
        },
    }


def selection(event_id="sr:match:1"):
    return {"event_id": event_id, "market_id": "18",
            "specifier": "total=1.5", "outcome_id": "12"}


def leg():
    return {"home": "Home FC", "away": "Away FC",
            "match_date": "2026-10-23", "kick_off_utc": "15:00"}


class BookingVerificationTests(unittest.TestCase):
    def verify(self, events=None, selections=None, legs=None):
        return verify_and_reprice(
            selections or [selection()], events if events is not None else [event()],
            expected_legs=legs, now_ms=NOW,
        )

    def test_exact_event_market_outcome_and_odds(self):
        quote = self.verify(legs=[leg()])
        self.assertEqual(quote["odds"], 1.35)
        self.assertEqual(quote["selections"][0]["event_id"], "sr:match:1")
        self.assertTrue(ensure_not_expired(quote, NOW))

    def test_price_change_changes_quote_signature(self):
        old = self.verify()
        new = self.verify([event(odds=1.42)])
        self.assertNotEqual(old["quote_signature"], new["quote_signature"])

    def test_rescheduled_fixture_changes_quote_signature(self):
        original = self.verify()
        delayed = self.verify([event(kickoff=KICKOFF + 3600000)])
        self.assertNotEqual(original["quote_signature"], delayed["quote_signature"])

    def test_started_event_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "started"):
            self.verify([event(kickoff=NOW)])

    def test_live_flag_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "started"):
            self.verify([event(live=True)])

    def test_closed_status_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "started"):
            self.verify([event(status="H1")])

    def test_no_board_record_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "no longer"):
            self.verify([])

    def test_wrong_event_id_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "no longer"):
            self.verify([event("sr:match:other")])

    def test_market_outcome_mismatch_refused(self):
        wrong = selection()
        wrong["outcome_id"] = "99"
        with self.assertRaisesRegex(BookingValidationError, "outcome"):
            self.verify(selections=[wrong])

    def test_duplicate_event_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "repeated"):
            self.verify(selections=[selection(), selection()])

    def test_reversed_teams_refused(self):
        flipped = event(home="Away FC", away="Home FC")
        with self.assertRaisesRegex(BookingValidationError, "Home/away"):
            self.verify([flipped], legs=[leg()])

    def test_second_fixture_different_kickoff_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "Kickoff mismatch"):
            self.verify([event(kickoff=KICKOFF + 3600000)], legs=[leg()])

    def test_invalid_price_refused(self):
        for price in (None, 0, 1.0, float("nan"), -5):
            with self.subTest(price=price):
                with self.assertRaises(BookingValidationError):
                    self.verify([event(odds=price)])

    def test_missing_event_start_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "started"):
            self.verify([event(kickoff=None)])

    def test_code_expired_at_first_kickoff(self):
        quote = self.verify()
        self.assertFalse(ensure_not_expired(quote, KICKOFF + 1))

    def test_share_code_is_not_treated_as_forever_live(self):
        issued = {"issued_at_ms": NOW, "deadline_ms": KICKOFF}
        self.assertTrue(quote_is_recent(issued, now_ms=NOW + 120000))
        self.assertFalse(quote_is_recent(issued, now_ms=NOW + 16 * 60000))
        self.assertTrue(quote_is_recent(
            issued, quote_checked_at_ms=NOW + 15 * 60000,
            now_ms=NOW + 20 * 60000))

    def test_invalid_monitor_timestamp_cannot_prove_freshness(self):
        issued = {"issued_at_ms": 0, "deadline_ms": KICKOFF}
        self.assertFalse(quote_is_recent(
            issued, quote_checked_at_ms="bogus", now_ms=NOW))

    def test_conflicting_records_refused(self):
        with self.assertRaisesRegex(BookingValidationError, "Conflicting"):
            self.verify([event(), event(odds=1.38)])


if __name__ == "__main__":
    unittest.main()
