"""
Services package for BetSightly backend.
"""

# Lazy imports — avoid pulling in heavy dependencies (requests, fastapi, etc.)
# at import time so that lightweight modules (e.g. HistoricalDataService) can
# be imported in test environments without the full dependency tree installed.

def __getattr__(name):
    if name == "basic_prediction_service":
        from .basic_prediction_service import basic_prediction_service
        return basic_prediction_service
    raise AttributeError(f"module 'services' has no attribute {name!r}")

__all__ = [
    "basic_prediction_service",
]