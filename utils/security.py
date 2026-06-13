"""
Security utilities for BetSightly backend.

This module provides security enhancements including rate limiting,
API key validation, input sanitization, and security headers.
"""

import logging
import time
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Set up logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to prevent API abuse."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_id: [(timestamp, count), ...]}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is allowed to make a request.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if timestamp > window_start
            ]
        else:
            self.requests[client_id] = []
        
        # Count requests in current window
        total_requests = sum(count for _, count in self.requests[client_id])
        
        if total_requests >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append((now, 1))
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        window_start = now - self.window_seconds
        
        if client_id not in self.requests:
            return self.max_requests
        
        # Count requests in current window
        total_requests = sum(
            count for timestamp, count in self.requests[client_id]
            if timestamp > window_start
        )
        
        return max(0, self.max_requests - total_requests)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global rate-limiting middleware applied to all requests."""

    # Paths that are exempt from rate limiting (health/liveness probes)
    _EXEMPT = {"/api/health", "/api/health/ready", "/api/health/live", "/"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in self._EXEMPT:
            client_id = get_client_id(request)
            if not rate_limiter.is_allowed(client_id):
                from starlette.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "remaining_requests": rate_limiter.get_remaining_requests(client_id),
                    },
                )
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        try:
            del response.headers["Server"]
        except (KeyError, AttributeError):
            pass

        return response


class APIKeyValidator:
    """Validator for API key authentication."""
    
    def __init__(self):
        self.valid_keys = set()
        self.key_metadata = {}  # {key: {created_at, last_used, permissions}}
    
    def generate_api_key(self, permissions: List[str] = None) -> str:
        """
        Generate a new API key.
        
        Args:
            permissions: List of permissions for this key
            
        Returns:
            Generated API key
        """
        api_key = f"betsightly_{secrets.token_urlsafe(32)}"
        
        self.valid_keys.add(api_key)
        self.key_metadata[api_key] = {
            "created_at": datetime.now(),
            "last_used": None,
            "permissions": permissions or ["read"],
            "usage_count": 0
        }
        
        return api_key
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if api_key in self.valid_keys:
            # Update usage metadata
            self.key_metadata[api_key]["last_used"] = datetime.now()
            self.key_metadata[api_key]["usage_count"] += 1
            return True
        
        return False
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked, False if key didn't exist
        """
        if api_key in self.valid_keys:
            self.valid_keys.remove(api_key)
            del self.key_metadata[api_key]
            return True
        
        return False
    
    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key."""
        return self.key_metadata.get(api_key)


# Global instances
rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)  # 1000 requests per hour
api_key_validator = APIKeyValidator()
security_bearer = HTTPBearer(auto_error=False)


def require_api_key(request: Request) -> None:
    """
    Dependency that enforces API key auth when API_KEY env var is set.

    Pass the key via the X-API-Key header. In development (API_KEY unset),
    all requests are allowed through so local iteration stays frictionless.
    """
    import os
    expected_key = os.getenv("API_KEY", "")
    if not expected_key:
        # Dev / unprotected mode — skip auth
        return

    provided_key = request.headers.get("X-API-Key", "")
    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail={"error": "Missing API key", "message": "Provide your key in the X-API-Key header."}
        )

    # Constant-time comparison to prevent timing attacks
    import hmac
    if not hmac.compare_digest(provided_key, expected_key):
        logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid API key", "message": "The provided API key is not valid."}
        )


def get_client_id(request: Request) -> str:
    """
    Get unique client identifier from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Unique client identifier
    """
    # Use IP address as client ID (in production, consider using API keys)
    client_ip = request.client.host if request.client else "unknown"
    
    # Include user agent for better uniqueness
    user_agent = request.headers.get("user-agent", "")
    
    # Create hash of IP + User Agent
    client_data = f"{client_ip}:{user_agent}"
    return hashlib.sha256(client_data.encode()).hexdigest()[:16]


def check_rate_limit(request: Request) -> None:
    """
    Check rate limit for request.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_id = get_client_id(request)
    
    if not rate_limiter.is_allowed(client_id):
        remaining = rate_limiter.get_remaining_requests(client_id)
        logger.warning(f"Rate limit exceeded for client {client_id}")
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "remaining_requests": remaining,
                "reset_time": int(time.time() + rate_limiter.window_seconds)
            }
        )


def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)) -> Optional[str]:
    """
    Validate API key from Authorization header.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        API key if valid, None otherwise
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not credentials:
        return None
    
    api_key = credentials.credentials
    
    if not api_key_validator.validate_key(api_key):
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is invalid or has been revoked."
            }
        )
    
    return api_key


def sanitize_input(data: Any) -> Any:
    """
    Sanitize input data to prevent injection attacks.
    
    Args:
        data: Input data to sanitize
        
    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", "&", "\"", "'", ";", "(", ")", "{", "}", "[", "]"]
        for char in dangerous_chars:
            data = data.replace(char, "")
        
        # Limit string length
        if len(data) > 1000:
            data = data[:1000]
    
    elif isinstance(data, dict):
        return {key: sanitize_input(value) for key, value in data.items()}
    
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    
    return data


def require_permissions(required_permissions: List[str]):
    """
    Decorator to require specific permissions for an endpoint.
    
    Args:
        required_permissions: List of required permissions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented with proper authentication
            # For now, just log the requirement
            logger.info(f"Endpoint {func.__name__} requires permissions: {required_permissions}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_security_event(event_type: str, details: Dict[str, Any], request: Request = None):
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event
        details: Event details
        request: FastAPI request object
    """
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "details": details
    }
    
    if request:
        log_data.update({
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
            "path": str(request.url.path),
            "method": request.method
        })
    
    logger.warning(f"Security event: {log_data}")
