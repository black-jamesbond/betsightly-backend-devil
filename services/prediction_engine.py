"""
Unified Prediction Engine

Single entry-point for all prediction operations. Consolidates:
- advanced_prediction_service  (full 22-model ensemble)
- quick_prediction_service     (fast 3-model path)
- basic_prediction_service     (heuristic fallback)
- cached_prediction_service    (in-memory caching)
- daily_prediction_cache       (DB-backed nightly batch)

Callers choose a mode; this module routes accordingly and falls back
gracefully when a heavier service is unavailable.

Modes
-----
"advanced"  Full XGBoost/LightGBM/NN ensemble. Slowest, most accurate.
"quick"     3-model subset with in-memory cache. Fast.
"cached"    Same as "quick" but served from a persistent cache layer.
"basic"     Heuristic-only, no ML. Used when models aren't loaded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy service references — imported on first use to avoid startup cost
# ---------------------------------------------------------------------------

_advanced = None
_quick = None
_cached = None
_basic = None
_daily = None


def _get_advanced():
    global _advanced
    if _advanced is None:
        try:
            from services.advanced_prediction_service import advanced_prediction_service
            _advanced = advanced_prediction_service
        except Exception as e:
            logger.warning(f"Advanced prediction service unavailable: {e}")
    return _advanced


def _get_quick():
    global _quick
    if _quick is None:
        try:
            from services.quick_prediction_service import quick_prediction_service
            _quick = quick_prediction_service
        except Exception as e:
            logger.warning(f"Quick prediction service unavailable: {e}")
    return _quick


def _get_cached():
    global _cached
    if _cached is None:
        try:
            from services.cached_prediction_service import cached_prediction_service
            _cached = cached_prediction_service
        except Exception as e:
            logger.warning(f"Cached prediction service unavailable: {e}")
    return _cached


def _get_basic():
    global _basic
    if _basic is None:
        try:
            from services.basic_prediction_service import basic_prediction_service
            _basic = basic_prediction_service
        except Exception as e:
            logger.warning(f"Basic prediction service unavailable: {e}")
    return _basic


def _get_daily():
    global _daily
    if _daily is None:
        try:
            from services.daily_prediction_cache import daily_prediction_cache
            _daily = daily_prediction_cache
        except Exception as e:
            logger.warning(f"Daily prediction cache unavailable: {e}")
    return _daily


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_predictions(
    date_str: Optional[str] = None,
    mode: str = "advanced",
    include_explanations: bool = False,
) -> Dict[str, Any]:
    """
    Get predictions for a date.

    Parameters
    ----------
    date_str : str, optional
        Date in YYYY-MM-DD format. Defaults to today.
    mode : str
        One of "advanced", "quick", "cached", "basic".
    include_explanations : bool
        When True (and mode=="advanced"), attach SHAP/LIME explanations.

    Returns
    -------
    dict  Standard prediction result payload.
    """
    if mode == "advanced":
        svc = _get_advanced()
        if svc:
            if include_explanations:
                return svc.get_enhanced_predictions_with_explanations(date_str)
            return svc.get_predictions_for_date(date_str)
        logger.warning("Advanced service unavailable, falling back to quick")
        mode = "quick"

    if mode == "quick":
        svc = _get_quick()
        if svc:
            return svc.get_predictions_for_date(date_str)
        logger.warning("Quick service unavailable, falling back to basic")
        mode = "basic"

    if mode == "cached":
        svc = _get_cached()
        if svc:
            return svc.get_predictions_for_date(date_str)
        logger.warning("Cached service unavailable, falling back to quick")
        return get_predictions(date_str, mode="quick")

    # basic — always available
    svc = _get_basic()
    if svc:
        return svc.get_predictions_for_date(date_str)

    return _empty_result(date_str, "No prediction service available")


def get_daily_cached_predictions(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve predictions from the DB-backed nightly cache."""
    svc = _get_daily()
    if svc:
        return svc.get_cached_predictions(date_str)
    return get_predictions(date_str, mode="cached")


def regenerate_daily_cache(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Force-regenerate the nightly prediction cache."""
    svc = _get_daily()
    if svc:
        return svc.generate_daily_predictions(date_str)
    return {"status": "error", "message": "Daily cache service unavailable"}


def get_model_info() -> Dict[str, Any]:
    """Return metadata about currently loaded models."""
    svc = _get_advanced()
    if svc:
        return svc.get_model_info()
    return {"status": "unavailable", "message": "Advanced service not loaded"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_result(date_str: Optional[str], message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "date": date_str,
        "predictions": [],
        "categories": {"2_odds": [], "5_odds": [], "10_odds": [], "rollover": []},
        "message": message,
    }
