"""
Main application.
"""

import logging
import os
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.orm import Session

import threading

from api.api import api_router
from database import init_db, get_db
from utils.config import settings
from utils.error_handling import setup_exception_handlers
from utils.security import SecurityMiddleware, RateLimitMiddleware

# ---------------------------------------------------------------------------
# Sentry — error tracking (no-op when SENTRY_DSN is not set)
# ---------------------------------------------------------------------------
_SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            environment=os.getenv("ENVIRONMENT", "production"),
            traces_sample_rate=0.1,   # capture 10% of transactions for performance
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        )
        logging.getLogger(__name__).info("Sentry error tracking enabled")
    except ImportError:
        logging.getLogger(__name__).warning(
            "SENTRY_DSN is set but sentry-sdk is not installed — "
            "add sentry-sdk to requirements.txt"
        )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize database (skip for Railway deployment if no DATABASE_URL)
try:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        init_db()
        logger.info("Database initialized successfully")
    else:
        logger.warning("No DATABASE_URL found - skipping database initialization for Railway deployment")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    # Don't raise in Railway deployment - let app start without DB for now
    if os.getenv("ENVIRONMENT") != "production":
        raise

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="BetSightly Football Predictions API",
    description="Advanced ML-powered football match predictions with confidence scoring",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None
)

# Phase 5: Re-enable production middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.betsightly.com", "*.onrender.com", "*.railway.app", "testserver"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityMiddleware)

# Add CORS middleware with enhanced security - allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Phase 5: Re-enable exception handlers
setup_exception_handlers(app)

# Include API router
app.include_router(api_router, prefix="/api")

# World Cup 2026 endpoints
try:
    from worldcup.api import router as worldcup_router
    app.include_router(worldcup_router)
    logger.info("World Cup 2026 API endpoints registered")
except ImportError as e:
    logger.warning(f"World Cup module not available: {e}")

def _auto_generate_predictions():
    """Auto-generate today's predictions on startup (runs in background thread)."""
    import time
    time.sleep(5)  # Let the server finish booting first
    try:
        from services.daily_predictions_service import DailyPredictionsService
        from database import SessionLocal
        from services.daily_predictions_service import DailyPredictionSummary

        today_str = datetime.now().strftime("%Y-%m-%d")
        today_date = datetime.now().date()

        db = SessionLocal()
        try:
            existing = db.query(DailyPredictionSummary).filter(
                DailyPredictionSummary.prediction_date == today_date
            ).first()

            if existing and existing.generation_status == "completed":
                logger.info(f"Predictions for {today_str} already exist — skipping auto-generate")
                return

            logger.info(f"Auto-generating predictions for {today_str}...")
            service = DailyPredictionsService()
            result = service.generate_daily_predictions(today_str)
            status = result.get("status", "unknown")
            count = result.get("summary", {}).get("predictions_generated", 0)
            logger.info(f"Auto-generate complete: status={status}, predictions={count}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Auto-generate predictions failed: {e}")

# Start auto-generation in background thread on app startup
threading.Thread(target=_auto_generate_predictions, daemon=True).start()
logger.info("Background prediction auto-generation scheduled")


@app.get("/")
def root():
    """Root endpoint with enhanced information."""
    return {
        "service": "BetSightly Football Predictions API",
        "version": "1.0.0",
        "status": "operational",
        "message": "Advanced ML-powered football predictions",
        "docs_url": "/docs" if settings.DEBUG else "Contact admin for API documentation",
        "health_check": "/api/health/",
        "predictions": "/api/predictions/",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
def health_check():
    """Basic health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "BetSightly API",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "production")
        }
    except Exception as e:
        # Fallback response if anything fails
        return {
            "status": "healthy",
            "service": "BetSightly API",
            "version": "1.0.0",
            "error": str(e)
        }

@app.get("/api/debug/predictions")
def debug_predictions():
    """Debug endpoint to check predictions data."""
    try:
        # Simple debug endpoint for Railway deployment
        return {
            "status": "operational",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "message": "Minimal deployment - ML services loading",
            "endpoints": {
                "health": "/api/health",
                "predictions": "/api/predictions/",
                "betting_codes": "/api/betting-codes/",
                "punters": "/api/punters/"
            },
            "deployment": "railway-minimal"
        }
    except Exception as e:
        logger.error(f"Error in debug predictions endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting debug predictions: {str(e)}")
