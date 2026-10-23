"""Verify that SportyBet booking code readback contains the exact requested legs."""
import unittest

from services.bookmaker_verification import BookingValidationError
from services.booking_code_readback import verify_code_readback


def response(code="ABCD12", selections=None):
    if selections is None:
        selections = [
            ("sr:match:1", "18", "total=1.5", "12"),
            ("sr:match:2", "18", "total=2.5", "13"),
        ]
    return {
        "bizCode": 10000,
        "data": {
            "shareCode": code, "unavailableOutcomes": [],
            "outcomes": [
                {
                    "eventId": e,
                    "markets": [{"id": market, "specifier": specifier,
                                 "outcomes": [{"id": outcome}]}],
                } for e, market, specifier, outcome in selections
            ],
        },
    }


def request():
    return [
        {"event_id": "sr:match:1", "market_id": "18",
         "specifier": "total=1.5", "outcome_id": "12"},
        {"event_id": "sr:match:2", "market_id": "18",
         "specifier": "total=2.5", "outcome_id": "13"},
    ]


class CodeReadbackTests(unittest.TestCase):
    def test_exact_2_leg_readback_verifies(self):
        got = verify_code_readback("ABCD12", request(), response())
        self.assertEqual(got["verification"], "readback_verified")
        self.assertEqual(got["verified_legs"], 2)

    def test_missing_leg_fails(self):
        short = response(selections=[("sr:match:1", "18", "total=1.5", "12")])
        with self.assertRaisesRegex(BookingValidationError, "different legs"):
            verify_code_readback("ABCD12", request(), short)

    def test_extra_leg_fails(self):
        extra = response(selections=[
            ("sr:match:1", "18", "total=1.5", "12"),
            ("sr:match:2", "18", "total=2.5", "13"),
            ("sr:match:3", "18", "total=1.5", "12"),
        ])
        with self.assertRaisesRegex(BookingValidationError, "different legs"):
            verify_code_readback("ABCD12", request(), extra)

    def test_wrong_side_of_market_fails(self):
        changed = response(selections=[
            ("sr:match:1", "18", "total=1.5", "13"),
            ("sr:match:2", "18", "total=2.5", "13"),
        ])
        with self.assertRaisesRegex(BookingValidationError, "different legs"):
            verify_code_readback("ABCD12", request(), changed)

    def test_different_share_code_fails(self):
        with self.assertRaisesRegex(BookingValidationError, "different share code"):
            verify_code_readback("ABCD12", request(), response(code="EFGH89"))

    def test_unavailable_outcomes_fails(self):
        data = response()
        data["data"]["unavailableOutcomes"] = [{"id": 999}]
        with self.assertRaisesRegex(BookingValidationError, "unavailable outcomes"):
            verify_code_readback("ABCD12", request(), data)

    def test_undocumented_response_shape_fails(self):
        with self.assertRaises(BookingValidationError):
            verify_code_readback("ABCD12", request(), {"bizCode": 10000, "data": {}})

    def test_duplicate_readback_fails(self):
        data = response()
        data["data"]["outcomes"].append(data["data"]["outcomes"][0])
        with self.assertRaisesRegex(BookingValidationError, "duplicate"):
            verify_code_readback("ABCD12", request(), data)

    def test_different_bookmaker_market_fails(self):
        data = response()
        data["data"]["outcomes"][1]["markets"][0]["id"] = "1"
        with self.assertRaisesRegex(BookingValidationError, "different legs"):
            verify_code_readback("ABCD12", request(), data)


if __name__ == "__main__":
    unittest.main()
