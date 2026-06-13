"""
Unit tests for the prediction engine facade.
"""

from unittest.mock import MagicMock, patch
from services import prediction_engine


def _make_success_response(date="2026-05-01"):
    return {
        "status": "success",
        "date": date,
        "predictions": [],
        "categories": {"2_odds": [], "5_odds": [], "10_odds": [], "rollover": []},
    }


class TestGetPredictions:
    def test_returns_dict(self):
        result = prediction_engine.get_predictions("2026-05-01", mode="basic")
        assert isinstance(result, dict)

    def test_basic_mode_always_returns_something(self):
        result = prediction_engine.get_predictions("2026-05-01", mode="basic")
        # Even with no API keys the basic service should return a structured payload
        assert "categories" in result or "status" in result

    def test_empty_result_shape(self):
        result = prediction_engine._empty_result("2026-05-01", "test error")
        assert result["status"] == "error"
        assert "categories" in result
        assert set(result["categories"].keys()) == {"2_odds", "5_odds", "10_odds", "rollover"}

    def test_advanced_fallback_to_basic(self):
        with patch.object(prediction_engine, "_get_advanced", return_value=None):
            with patch.object(prediction_engine, "_get_quick", return_value=None):
                with patch.object(
                    prediction_engine,
                    "_get_basic",
                    return_value=MagicMock(
                        get_predictions_for_date=MagicMock(return_value=_make_success_response())
                    ),
                ):
                    result = prediction_engine.get_predictions("2026-05-01", mode="advanced")
                    assert result["status"] == "success"
