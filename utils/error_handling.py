"""
Error Handling Utilities

This module provides centralized error handling for the application.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException, Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logger = logging.getLogger(__name__)


class BetSightlyError(Exception):
    """Base exception class for BetSightly application."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class APIError(BetSightlyError):
    """Exception for API-related errors."""
    pass


class DatabaseError(BetSightlyError):
    """Exception for database-related errors."""
    pass


class ModelError(BetSightlyError):
    """Exception for ML model-related errors."""
    pass


class ValidationError(BetSightlyError):
    """Exception for validation errors."""
    pass


def handle_database_error(error: Exception, operation: str = "database operation") -> HTTPException:
    """
    Handle database errors and convert them to HTTP exceptions.
    
    Args:
        error: The database error
        operation: Description of the operation that failed
        
    Returns:
        HTTPException with appropriate status code and message
    """
    logger.error(f"Database error during {operation}: {str(error)}")
    
    if isinstance(error, SQLAlchemyError):
        return HTTPException(
            status_code=500,
            detail=f"Database error during {operation}. Please try again later."
        )
    
    return HTTPException(
        status_code=500,
        detail=f"Unexpected error during {operation}: {str(error)}"
    )


def handle_api_error(error: Exception, operation: str = "API operation") -> HTTPException:
    """
    Handle API errors and convert them to HTTP exceptions.
    
    Args:
        error: The API error
        operation: Description of the operation that failed
        
    Returns:
        HTTPException with appropriate status code and message
    """
    logger.error(f"API error during {operation}: {str(error)}")
    
    if isinstance(error, APIError):
        return HTTPException(
            status_code=502,
            detail=f"External API error during {operation}: {error.message}"
        )
    
    return HTTPException(
        status_code=500,
        detail=f"Unexpected error during {operation}: {str(error)}"
    )


def handle_model_error(error: Exception, operation: str = "model operation") -> HTTPException:
    """
    Handle ML model errors and convert them to HTTP exceptions.
    
    Args:
        error: The model error
        operation: Description of the operation that failed
        
    Returns:
        HTTPException with appropriate status code and message
    """
    logger.error(f"Model error during {operation}: {str(error)}")
    
    if isinstance(error, ModelError):
        return HTTPException(
            status_code=503,
            detail=f"Model unavailable during {operation}: {error.message}"
        )
    
    return HTTPException(
        status_code=500,
        detail=f"Unexpected error during {operation}: {str(error)}"
    )


def create_error_response(
    status_code: int,
    message: str,
    error_code: str = None,
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        status_code: HTTP status code
        message: Error message
        error_code: Application-specific error code
        details: Additional error details
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "status": "error",
        "message": message,
        "status_code": status_code
    }
    
    if error_code:
        response["error_code"] = error_code
    
    if details:
        response["details"] = details
    
    return response


def log_and_raise_http_exception(
    status_code: int,
    message: str,
    operation: str = None,
    error_code: str = None,
    details: Dict[str, Any] = None
) -> None:
    """
    Log an error and raise an HTTP exception.
    
    Args:
        status_code: HTTP status code
        message: Error message
        operation: Description of the operation that failed
        error_code: Application-specific error code
        details: Additional error details
    """
    log_message = f"Error during {operation}: {message}" if operation else message
    logger.error(log_message)
    
    if details:
        logger.error(f"Error details: {details}")
    
    raise HTTPException(
        status_code=status_code,
        detail=create_error_response(status_code, message, error_code, details)
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Set up global exception handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(BetSightlyError)
    async def betsightly_exception_handler(request: Request, exc: BetSightlyError):
        """Handle custom BetSightly exceptions."""
        logger.error(f"BetSightly error: {exc.message} (Code: {exc.error_code})")

        return JSONResponse(
            status_code=getattr(exc, 'status_code', 500),
            content={
                "status": "error",
                "message": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")

        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "Request validation failed",
                "error_code": "VALIDATION_ERROR",
                "details": {"validation_errors": exc.errors()},
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "message": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        """Handle database errors."""
        logger.error(f"Database error: {str(exc)}")

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Database operation failed",
                "error_code": "DATABASE_ERROR",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred",
                "error_code": "INTERNAL_SERVER_ERROR",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
