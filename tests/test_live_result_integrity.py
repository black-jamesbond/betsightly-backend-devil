"""Legacy live-score dashboard must share the conservative 90-minute policy."""
import importlib
import unittest
from unittest.mock import patch

from fixtures_livescore import _event_live

dashboard = importlib.import_module("dashboard.app")


def raw_event(phase, home="3", away="2"):
    return {"Eid": "sample", "Eps": phase, "Esd": "20261025150000",
            "Tr1": home, "Tr2": away}


class LiveResultIntegrityTests(unittest.TestCase):
    def test_live_adapter_keeps_provider_result_phase(self):
        overtime = _event_live(raw_event("AET"))
        self.assertEqual(overtime["status"], "Finished")
        self.assertEqual(overtime["raw_status"], "AET")
        regular = _event_live(raw_event("FT", "0", "0"))
        self.assertEqual(regular["raw_status"], "FT")
        self.assertEqual(regular["score"], "0-0")

    def test_extra_time_cannot_be_used_for_90_minute_dashboard_settlement(self):
        with patch.object(dashboard, "live_map", return_value={
            "match": _event_live(raw_event("AET")),
        }):
            result = dashboard._with_live_finals("2026-10-25", {})
        self.assertEqual(result, {})

    def test_plain_full_time_merges_even_goalless_match(self):
        with patch.object(dashboard, "live_map", return_value={
            "match": _event_live(raw_event("FT", "0", "0")),
        }):
            result = dashboard._with_live_finals("2026-10-25", {})
        self.assertEqual(result["match"], {"hs": "0", "as": "0"})

    def test_penalties_without_90min_score_not_silent_win(self):
        with patch.object(dashboard, "live_map", return_value={
            "match": _event_live(raw_event("FT Pen.")),
        }):
            result = dashboard._with_live_finals("2026-10-25", {})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
