"""
API Client Module

This module provides a base API client class for making HTTP requests.
It handles authentication, rate limiting, and error handling.
"""

import json
import requests
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from utils.common import setup_logging, retry_operation
from utils.config import settings

# Set up logging
logger = setup_logging(__name__)

class APIClient:
    """
    Base API client for making HTTP requests.

    Features:
    - Authentication
    - Rate limiting
    - Error handling
    - Retry logic
    - Response caching
    """

    def __init__(self, base_url: str, headers: Dict[str, str] = None, timeout: int = 30):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for API requests
            headers: HTTP headers to include in requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

        # Rate limiting
        self.rate_limit = {
            "limit": 100,  # Default limit
            "remaining": 100,
            "reset": datetime.now() + timedelta(days=1)
        }

        # Request history
        self.request_history = []

        # Response cache
        self.cache = {}
        self.cache_expiry = {}

    def _update_rate_limit(self, headers: Dict[str, str]) -> None:
        """
        Update rate limit information from response headers.

        Args:
            headers: Response headers
        """
        # Default implementation - override in subclasses
        pass

    def _check_rate_limit(self) -> bool:
        """
        Check if rate limit allows a request.

        Returns:
            True if request is allowed, False otherwise
        """
        if self.rate_limit["remaining"] <= 0:
            # Check if rate limit has reset
            if datetime.now() >= self.rate_limit["reset"]:
                # Reset rate limit
                self.rate_limit["remaining"] = self.rate_limit["limit"]
                return True

            # Rate limit exceeded
            reset_in = (self.rate_limit["reset"] - datetime.now()).total_seconds()
            logger.warning(f"Rate limit exceeded. Resets in {reset_in:.1f} seconds.")
            return False

        return True

    def _wait_for_rate_limit(self) -> None:
        """
        Wait for rate limit to reset if needed.
        """
        if not self._check_rate_limit():
            # Calculate wait time
            wait_time = (self.rate_limit["reset"] - datetime.now()).total_seconds()

            # Add a small buffer
            wait_time += 1

            logger.info(f"Waiting {wait_time:.1f} seconds for rate limit to reset...")
            time.sleep(wait_time)

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any] = None) -> str:
        """
        Generate a cache key for a request.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Cache key
        """
        params_str = json.dumps(params or {}, sort_keys=True)
        return f"{endpoint}_{params_str}"

    def _get_from_cache(self, endpoint: str, params: Dict[str, Any] = None, max_age: int = 3600) -> Optional[Dict[str, Any]]:
        """
        Get a response from cache if available and not expired.

        Args:
            endpoint: API endpoint
            params: Request parameters
            max_age: Maximum age of cached response in seconds

        Returns:
            Cached response or None
        """
        cache_key = self._get_cache_key(endpoint, params)

        if cache_key in self.cache:
            # Check if cache has expired
            if cache_key in self.cache_expiry:
                if datetime.now() < self.cache_expiry[cache_key]:
                    logger.debug(f"Cache hit for {endpoint}")
                    return self.cache[cache_key]

            # Cache expired or no expiry set
            logger.debug(f"Cache expired for {endpoint}")
            del self.cache[cache_key]
            if cache_key in self.cache_expiry:
                del self.cache_expiry[cache_key]

        return None

    def _add_to_cache(self, endpoint: str, params: Dict[str, Any], response: Dict[str, Any], max_age: int = 3600) -> None:
        """
        Add a response to cache.

        Args:
            endpoint: API endpoint
            params: Request parameters
            response: Response to cache
            max_age: Cache expiry in seconds
        """
        cache_key = self._get_cache_key(endpoint, params)

        self.cache[cache_key] = response
        self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=max_age)

        logger.debug(f"Cached response for {endpoint} (expires in {max_age} seconds)")

    def request(self, method: str, endpoint: str, params: Dict[str, Any] = None,
                data: Dict[str, Any] = None, use_cache: bool = True,
                max_cache_age: int = 3600, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            use_cache: Whether to use cache
            max_cache_age: Maximum age of cached response in seconds
            max_retries: Maximum number of retries

        Returns:
            Response data
        """
        # Check cache if GET request and caching is enabled
        if method.upper() == "GET" and use_cache:
            cached_response = self._get_from_cache(endpoint, params, max_cache_age)
            if cached_response is not None:
                return cached_response

        # Wait for rate limit if needed
        self._wait_for_rate_limit()

        # Build URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Define request operation
        def make_request():
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=self.timeout
            )

            # Update rate limit
            self._update_rate_limit(response.headers)

            # Decrement remaining requests
            self.rate_limit["remaining"] = max(0, self.rate_limit["remaining"] - 1)

            # Add to request history
            self.request_history.append({
                "method": method.upper(),
                "url": url,
                "params": params,
                "data": data,
                "status_code": response.status_code,
                "timestamp": datetime.now().isoformat()
            })

            # Trim history if too long
            if len(self.request_history) > 100:
                self.request_history = self.request_history[-100:]

            # Check for errors
            response.raise_for_status()

            # Parse response
            try:
                response_data = response.json()
            except ValueError:
                response_data = {"text": response.text}

            return response_data

        try:
            # Make request with retries
            response_data = retry_operation(
                make_request,
                max_retries=max_retries,
                retry_delay=2,
                exceptions=(requests.RequestException,),
                logger=logger
            )

            # Cache response if GET request and caching is enabled
            if method.upper() == "GET" and use_cache:
                self._add_to_cache(endpoint, params, response_data, max_cache_age)

            return response_data

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "endpoint": endpoint,
                "params": params
            }

    def get(self, endpoint: str, params: Dict[str, Any] = None, use_cache: bool = True,
            max_cache_age: int = 3600, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use cache
            max_cache_age: Maximum age of cached response in seconds
            max_retries: Maximum number of retries

        Returns:
            Response data
        """
        return self.request("GET", endpoint, params=params, use_cache=use_cache,
                           max_cache_age=max_cache_age, max_retries=max_retries)

    def post(self, endpoint: str, data: Dict[str, Any] = None, params: Dict[str, Any] = None,
             max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            max_retries: Maximum number of retries

        Returns:
            Response data
        """
        return self.request("POST", endpoint, params=params, data=data, use_cache=False,
                           max_retries=max_retries)

    def put(self, endpoint: str, data: Dict[str, Any] = None, params: Dict[str, Any] = None,
            max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            max_retries: Maximum number of retries

        Returns:
            Response data
        """
        return self.request("PUT", endpoint, params=params, data=data, use_cache=False,
                           max_retries=max_retries)

    def delete(self, endpoint: str, params: Dict[str, Any] = None, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            max_retries: Maximum number of retries

        Returns:
            Response data
        """
        return self.request("DELETE", endpoint, params=params, use_cache=False,
                           max_retries=max_retries)

    def get_fixtures_improved(self, date: str = None, league: int = None, last: int = None) -> Dict[str, Any]:
        """
        Get fixtures with improved parameters for API Football.

        Args:
            date: Date string in YYYY-MM-DD format
            league: League ID
            last: Number of last fixtures to get

        Returns:
            Response data
        """
        endpoint = "fixtures"
        params = {}

        if date:
            params["date"] = date
        if league:
            params["league"] = league
        if last:
            params["last"] = last

        return self.get(endpoint, params=params)


class FootballDataClient(APIClient):
    """
    Football-Data.org API client.

    This client is specifically designed to work with the Football-Data.org API.
    It handles authentication, rate limiting, and provides methods for common endpoints.
    """

    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30):
        """
        Initialize the Football-Data.org API client.

        Args:
            api_key: Football-Data.org API key
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
        """
        # Use settings if not provided
        api_key = api_key or settings.football_data.API_KEY
        base_url = base_url or settings.football_data.BASE_URL

        # Set up headers with API key
        headers = {
            "X-Auth-Token": api_key
        }

        super().__init__(base_url=base_url, headers=headers, timeout=timeout)

        # Set rate limit for Football-Data.org
        self.rate_limit = {
            "limit": 10,  # 10 calls per minute
            "remaining": 10,
            "reset": datetime.now() + timedelta(minutes=1)
        }

    def _update_rate_limit(self, headers: Dict[str, str]) -> None:
        """
        Update rate limit information from response headers.

        Args:
            headers: Response headers
        """
        # Football-Data.org uses X-Ratelimit-* headers
        if "X-Ratelimit-Remaining" in headers:
            self.rate_limit["remaining"] = int(headers["X-Ratelimit-Remaining"])

        if "X-Ratelimit-Reset" in headers:
            # X-Ratelimit-Reset is in seconds
            reset_seconds = int(headers["X-Ratelimit-Reset"])
            self.rate_limit["reset"] = datetime.now() + timedelta(seconds=reset_seconds)

    def get_matches(self, date_str: str = None, competition: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get matches for a specific date and/or competition.

        Args:
            date_str: Date string in YYYY-MM-DD format
            competition: Competition code (e.g., PL, PD, SA)
            use_cache: Whether to use cache

        Returns:
            Dictionary with matches data
        """
        if competition:
            # Get matches for a specific competition
            endpoint = f"competitions/{competition}/matches"
            params = {}

            if date_str:
                params["dateFrom"] = date_str
                params["dateTo"] = date_str

            return self.get(endpoint, params=params, use_cache=use_cache)
        elif date_str:
            # Get matches for a specific date across all competitions
            endpoint = "matches"
            params = {
                "dateFrom": date_str,
                "dateTo": date_str
            }

            return self.get(endpoint, params=params, use_cache=use_cache)
        else:
            # Get today's matches
            today = datetime.now().strftime("%Y-%m-%d")
            endpoint = "matches"
            params = {
                "dateFrom": today,
                "dateTo": today
            }

            return self.get(endpoint, params=params, use_cache=use_cache)

    def get_competition_matches(self, competition: str, date_str: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get matches for a specific competition.

        Args:
            competition: Competition code (e.g., PL, PD, SA)
            date_str: Date string in YYYY-MM-DD format
            use_cache: Whether to use cache

        Returns:
            Dictionary with matches data
        """
        return self.get_matches(date_str=date_str, competition=competition, use_cache=use_cache)

    def get_daily_matches(self, date_str: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get matches for a specific date across all competitions.

        Args:
            date_str: Date string in YYYY-MM-DD format (default: today)
            use_cache: Whether to use cache

        Returns:
            Dictionary with matches data
        """
        return self.get_matches(date_str=date_str, use_cache=use_cache)
