"""
Health Check API Endpoints

This module provides health check endpoints to monitor the application status.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BetSightly Backend API"
    }


@router.get("/detailed")
def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check that verifies all critical components.
    
    Returns:
        Comprehensive health status including database, API keys, and models
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BetSightly Backend API",
        "version": "1.0.0",
        "checks": {}
    }
    
    overall_healthy = True
    
    # Check database connectivity
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        overall_healthy = False
    
    # Check API keys configuration
    api_keys_status = check_api_keys()
    health_status["checks"]["api_keys"] = api_keys_status
    if api_keys_status["status"] != "healthy":
        overall_healthy = False
    
    # Check model availability
    models_status = check_models()
    health_status["checks"]["models"] = models_status
    if models_status["status"] != "healthy":
        overall_healthy = False
    
    # Check external API connectivity
    external_apis_status = check_external_apis()
    health_status["checks"]["external_apis"] = external_apis_status
    if external_apis_status["status"] != "healthy":
        overall_healthy = False
    
    # Check file system permissions
    filesystem_status = check_filesystem()
    health_status["checks"]["filesystem"] = filesystem_status
    if filesystem_status["status"] != "healthy":
        overall_healthy = False
    
    # Set overall status
    health_status["status"] = "healthy" if overall_healthy else "unhealthy"
    
    # Return appropriate HTTP status
    if not overall_healthy:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


def check_api_keys() -> Dict[str, Any]:
    """Check if required API keys are configured."""
    try:
        missing_keys = []
        
        # Check Football Data API key
        if not settings.football_data.API_KEY:
            missing_keys.append("FOOTBALL_DATA_API_KEY")

        # Check API Football key
        if not settings.api_football.API_KEY:
            missing_keys.append("API_FOOTBALL_API_KEY")
        
        if missing_keys:
            return {
                "status": "unhealthy",
                "message": f"Missing API keys: {', '.join(missing_keys)}",
                "missing_keys": missing_keys
            }
        
        return {
            "status": "healthy",
            "message": "All required API keys are configured"
        }
        
    except Exception as e:
        logger.error(f"API keys check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Error checking API keys: {str(e)}"
        }


def check_models() -> Dict[str, Any]:
    """Check if ML models are available."""
    try:
        # Check if model factory is available
        try:
            from ml.model_factory import model_factory
            available_models = model_factory.get_available_models() if hasattr(model_factory, 'get_available_models') else []
            
            return {
                "status": "healthy",
                "message": f"Model factory available with {len(available_models)} models",
                "available_models": available_models
            }
        except ImportError:
            return {
                "status": "degraded",
                "message": "Model factory not available - using fallback predictions"
            }
        
    except Exception as e:
        logger.error(f"Models check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Error checking models: {str(e)}"
        }


def check_external_apis() -> Dict[str, Any]:
    """Check connectivity to external APIs."""
    try:
        # This is a basic check - in production you might want to make actual API calls
        # but be careful about rate limits
        
        apis_status = {
            "football_data": "not_tested",
            "api_football": "not_tested"
        }
        
        # For now, just check if the base URLs are configured
        if settings.football_data.BASE_URL:
            apis_status["football_data"] = "configured"
        
        if settings.api_football.BASE_URL:
            apis_status["api_football"] = "configured"
        
        return {
            "status": "healthy",
            "message": "External APIs configured",
            "apis": apis_status
        }
        
    except Exception as e:
        logger.error(f"External APIs check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Error checking external APIs: {str(e)}"
        }


def check_filesystem() -> Dict[str, Any]:
    """Check file system permissions and required directories."""
    try:
        required_dirs = [
            settings.ml.DATA_DIR,
            settings.ml.CACHE_DIR,
            settings.ml.MODEL_DIR
        ]
        
        missing_dirs = []
        permission_errors = []
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    missing_dirs.append(f"{dir_path}: {str(e)}")
            
            # Test write permissions
            try:
                test_file = os.path.join(dir_path, ".health_check_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                permission_errors.append(f"{dir_path}: {str(e)}")
        
        if missing_dirs or permission_errors:
            return {
                "status": "unhealthy",
                "message": "File system issues detected",
                "missing_directories": missing_dirs,
                "permission_errors": permission_errors
            }
        
        return {
            "status": "healthy",
            "message": "All required directories accessible"
        }
        
    except Exception as e:
        logger.error(f"Filesystem check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Error checking filesystem: {str(e)}"
        }


@router.get("/ready")
def readiness_check(db: Session = Depends(get_db)):
    """
    Kubernetes-style readiness check.
    
    Returns 200 if the service is ready to accept traffic.
    """
    try:
        # Check database
        db.execute(text("SELECT 1"))
        
        # Check critical API keys
        if not settings.football_data.API_KEY or not settings.api_football.API_KEY:
            raise HTTPException(status_code=503, detail="API keys not configured")
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/live")
def liveness_check():
    """
    Kubernetes-style liveness check.
    
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}
