"""
Shared types and response helpers for prediction endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class PredictionCategory(str, Enum):
    SAFE_BETS = "2_odds"
    BALANCED_RISK = "5_odds"
    HIGH_REWARD = "10_odds"
    ROLLOVER = "rollover"


class ResponseFormat(str, Enum):
    SIMPLE = "simple"
    DETAILED = "detailed"
    COMBINATIONS = "combinations"


_CATEGORY_META: Dict[str, Dict[str, Any]] = {
    "2_odds": {
        "name": "Safe Bets",
        "description": "Lower odds, higher confidence predictions",
        "target_odds": 2.0,
        "risk_level": "low",
    },
    "5_odds": {
        "name": "Balanced Risk",
        "description": "Medium odds, balanced risk-reward",
        "target_odds": 5.0,
        "risk_level": "medium",
    },
    "10_odds": {
        "name": "High Reward",
        "description": "Higher odds, higher potential returns",
        "target_odds": 10.0,
        "risk_level": "high",
    },
    "rollover": {
        "name": "10-Day Rollover",
        "description": "Daily predictions for a 10-day rollover strategy",
        "target_odds": 3.0,
        "risk_level": "medium",
    },
}


def get_category_metadata(category: str) -> Dict[str, Any]:
    return _CATEGORY_META.get(category, {})


def standardize_prediction_response(
    predictions: List[Any],
    category: Optional[str] = None,
    format_type: ResponseFormat = ResponseFormat.SIMPLE,
) -> Dict[str, Any]:
    """Convert a list of predictions into the standard envelope format."""
    if not predictions:
        return {
            "count": 0,
            "predictions": [],
            "metadata": get_category_metadata(category) if category else {},
        }

    serialized = [p.to_dict() if hasattr(p, "to_dict") else p for p in predictions]
    response: Dict[str, Any] = {"count": len(serialized), "predictions": serialized}

    if category:
        response["metadata"] = get_category_metadata(category)

    if format_type == ResponseFormat.DETAILED:
        confidences = [p.get("confidence", 0) for p in serialized if isinstance(p, dict)]
        odds = [p.get("odds", 0) for p in serialized if isinstance(p, dict)]
        response["statistics"] = {
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "avg_odds": sum(odds) / len(odds) if odds else 0,
        }

    return response
