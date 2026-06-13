"""
API router.
"""

from fastapi import APIRouter, Depends

from api.endpoints import (
    betting_codes, predictions, fixtures, punters,
    bookmakers, dashboard, health, ml_predictions,
    daily_predictions, accumulators,
)
from utils.security import require_api_key

# Basketball re-enable when NBA data fetcher is production-ready:
# from api.endpoints import basketball_predictions

api_router = APIRouter()

# Health endpoints are intentionally unauthenticated (load-balancer probes need them)
api_router.include_router(health.router, prefix="/health", tags=["health"])

# All other endpoints require a valid API key
_protected = {"dependencies": [Depends(require_api_key)]}
api_router.include_router(betting_codes.router, prefix="/betting-codes", tags=["betting-codes"], **_protected)
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"], **_protected)
api_router.include_router(ml_predictions.router, prefix="/ml-predictions", tags=["ml-predictions"], **_protected)
api_router.include_router(daily_predictions.router, prefix="/daily-predictions", tags=["daily-predictions"], **_protected)
api_router.include_router(accumulators.router, prefix="/accumulators", tags=["accumulators"], **_protected)
api_router.include_router(fixtures.router, prefix="/fixtures", tags=["fixtures"], **_protected)
api_router.include_router(punters.router, prefix="/punters", tags=["punters"], **_protected)
api_router.include_router(bookmakers.router, prefix="/bookmakers", tags=["bookmakers"], **_protected)
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"], **_protected)
