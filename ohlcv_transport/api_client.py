"""
API client for Twelve Data.

This module provides utilities for making API requests to Twelve Data,
handling rate limits, implementing caching, and managing error responses.
"""
import os
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import requests
import pandas as pd
from requests.exceptions import RequestException

# Import the config manager
from ohlcv_transport.config import config_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_client")


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, calls_per_minute: int = 8, calls_per_day: int = 800):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
            calls_per_day: Maximum number of calls allowed per day
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_timestamps: List[float] = []
        self.day_timestamps: List[float] = []
    
    def wait_if_needed(self) -> None:
        """
        Check rate limits and wait if necessary.
        """
        current_time = time.time()
        
        # Remove timestamps older than one minute
        self.minute_timestamps = [ts for ts in self.minute_timestamps 
                                 if current_time - ts < 60]
        
        # Remove timestamps older than one day
        self.day_timestamps = [ts for ts in self.day_timestamps 
                              if current_time - ts < 86400]  # 86400 seconds = 1 day
        
        # Check if we're at the minute limit
        if len(self.minute_timestamps) >= self.calls_per_minute:
            oldest_timestamp = min(self.minute_timestamps)
            wait_time = 60 - (current_time - oldest_timestamp) + 0.1  # Add 0.1s buffer
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                # Update current time after waiting
                current_time = time.time()
        
        # Check if we're at the daily limit
        if len(self.day_timestamps) >= self.calls_per_day:
            oldest_timestamp = min(self.day_timestamps)
            wait_time = 86400 - (current_time - oldest_timestamp) + 0.1  # Add 0.1s buffer
            if wait_time > 0:
                logger.warning(f"Daily rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                # Update current time after waiting
                current_time = time.time()
        
        # Add current timestamp to both lists
        self.minute_timestamps.append(current_time)
        self.day_timestamps.append(current_time)


class APICache:
    """Cache for API responses."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, 
                 expiry_hours: int = 24):
        """
        Initialize the API response cache.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_hours: Number of hours before cached responses expire
        """
        if cache_dir is None:
            cache_dir = config_manager.get_cache_dir()
        else:
            cache_dir = Path(cache_dir)
            
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Path to the cache file
        """
        # Create a filename from the cache key using a hash
        hash_object = hashlib.md5(cache_key.encode())
        filename = f"{hash_object.hexdigest()}.json"
        return self.cache_dir / filename
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response.
        
        Args:
            cache_key: The cache key
            
        Returns:
            The cached response or None if not found or expired
        """
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            timestamp = cache_data.get('timestamp', 0)
            current_time = time.time()
            if current_time - timestamp > self.expiry_hours * 3600:
                logger.debug(f"Cache expired for key {cache_key}")
                return None
            
            return cache_data.get('data')
        except Exception as e:
            logger.warning(f"Error reading cache file: {e}")
            return None
    
    def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        Cache a response.
        
        Args:
            cache_key: The cache key
            data: The data to cache
        """
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'data': data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            logger.debug(f"Cached response for key {cache_key}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")


class TwelveDataClient:
    """Client for the Twelve Data API."""
    
    def __init__(self):
        """Initialize the Twelve Data API client."""
        self.base_url = config_manager.get('api.twelve_data.base_url')
        self.api_key = config_manager.get('api.twelve_data.api_key')
        
        # Initialize rate limiter with configured values
        rate_limit_config = config_manager.get('api.twelve_data.rate_limit', {})
        self.rate_limiter = RateLimiter(
            calls_per_minute=rate_limit_config.get('calls_per_minute', 8),
            calls_per_day=rate_limit_config.get('calls_per_day', 800)
        )
        
        # Initialize cache with configured values
        self.cache = APICache(
            expiry_hours=config_manager.get('system.data_cache_expiry_hours', 24)
        )
        
        # Initialize session
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any], 
                      use_cache: bool = True) -> Dict[str, Any]:
        """
        Make a request to the Twelve Data API.
        
        Args:
            endpoint: API endpoint (e.g., 'time_series')
            params: Query parameters
            use_cache: Whether to use and update the cache
            
        Returns:
            API response as a dictionary
            
        Raises:
            RequestException: If the request fails after retries
        """
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        # Build the full URL
        url = f"{self.base_url}/{endpoint}"
        
        # Generate a cache key from the URL and parameters
        cache_key = f"{url}?{json.dumps(params, sort_keys=True)}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_response
        
        # Wait if needed to respect rate limits
        self.rate_limiter.wait_if_needed()
        
        # Make the request with exponential backoff for retries
        max_retries = 5
        base_backoff = 1  # second
        
        for retry_count in range(max_retries):
            try:
                logger.debug(f"Making request to {endpoint} (attempt {retry_count + 1}/{max_retries})")
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                
                response_data = response.json()
                
                # Check for API-level errors
                if 'status' in response_data and response_data['status'] == 'error':
                    error_message = response_data.get('message', 'Unknown API error')
                    logger.warning(f"API error: {error_message}")
                    
                    # Handle rate limit errors specially
                    if 'Rate limit' in error_message:
                        wait_time = base_backoff * (2 ** retry_count)
                        logger.info(f"Rate limit exceeded. Waiting {wait_time} seconds")
                        time.sleep(wait_time)
                        continue
                        
                    raise RequestException(f"API error: {error_message}")
                
                # Cache the successful response if caching is enabled
                if use_cache:
                    self.cache.set(cache_key, response_data)
                
                return response_data
                
            except RequestException as e:
                logger.warning(f"Request failed: {e}")
                
                # If this was the last retry, raise the exception
                if retry_count == max_retries - 1:
                    raise
                
                # Calculate backoff time with exponential increase and jitter
                wait_time = base_backoff * (2 ** retry_count) * (0.5 + 0.5 * (time.time() % 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds")
                time.sleep(wait_time)
    
    def get_time_series(self, symbol: str, interval: str = '1min', 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         outputsize: int = 5000,
                         use_cache: bool = True) -> pd.DataFrame:
        """
        Get time series data for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., 'AAPL')
            interval: Data interval ('1min', '5min', '1h', '1day', etc.)
            start_date: Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            end_date: End date in same format (defaults to current time)
            outputsize: Number of data points to return
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RequestException: If the API request fails
            ValueError: If the response format is invalid
        """
        # Prepare parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'format': 'JSON'
        }
        
        # Add date parameters if provided
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        try:
            response = self._make_request('time_series', params, use_cache=use_cache)
            
            # Check if we have the expected data in the response
            if 'values' not in response:
                if 'status' in response and response['status'] == 'error':
                    error_message = response.get('message', 'Unknown error')
                    raise ValueError(f"API returned error: {error_message}")
                else:
                    raise ValueError("Unexpected API response format")
            
            # Convert to DataFrame
            df = pd.DataFrame(response['values'])
            
            # Convert types
            if not df.empty:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df = df.set_index('datetime')
                    # Sort by datetime
                    df = df.sort_index()
            
            logger.info(f"Retrieved {len(df)} {interval} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving time series data for {symbol}: {e}")
            raise
            
    def get_quote(self, symbol: str, use_cache: bool = False) -> Dict[str, Any]:
        """
        Get the latest quote for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., 'AAPL')
            use_cache: Whether to use cached data (usually set to False for real-time quotes)
            
        Returns:
            Dictionary with quote data
            
        Raises:
            RequestException: If the API request fails
            ValueError: If the response format is invalid
        """
        params = {
            'symbol': symbol,
            'format': 'JSON'
        }
        
        try:
            response = self._make_request('quote', params, use_cache=use_cache)
            
            logger.debug(f"Retrieved quote for {symbol}")
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving quote for {symbol}: {e}")
            raise
            
    def get_price(self, symbol: str, use_cache: bool = False) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., 'AAPL')
            use_cache: Whether to use cached data
            
        Returns:
            Latest price as float
            
        Raises:
            RequestException: If the API request fails
            ValueError: If the response format is invalid
        """
        params = {
            'symbol': symbol,
            'format': 'JSON'
        }
        
        try:
            response = self._make_request('price', params, use_cache=use_cache)
            
            if 'price' not in response:
                raise ValueError("Unexpected API response format")
            
            price = float(response['price'])
            logger.debug(f"Retrieved price for {symbol}: {price}")
            return price
            
        except Exception as e:
            logger.error(f"Error retrieving price for {symbol}: {e}")
            raise
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the API connection and key validity.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.api_key:
            return False, "API key not configured"
        
        try:
            # Use a simple endpoint to test the connection
            response = self._make_request('stocks', {'exchange': 'NASDAQ', 'format': 'JSON'}, use_cache=False)
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"


# Create a singleton instance for easy import
twelve_data_client = TwelveDataClient()


if __name__ == "__main__":
    # Example usage
    print("API client test:")
    api_key_valid = config_manager.validate_api_key()
    
    if not api_key_valid:
        print("API key not configured. Please set it with:")
        print("config_manager.set_api_key('YOUR_API_KEY')")
    else:
        success, message = twelve_data_client.test_connection()
        print(f"Connection test: {message}")
        
        if success:
            try:
                # Get a symbol price
                symbol = "AAPL"
                price = twelve_data_client.get_price(symbol)
                print(f"Current price of {symbol}: ${price}")
                
                # Get recent time series data
                df = twelve_data_client.get_time_series(
                    symbol=symbol,
                    interval="1min",
                    outputsize=10
                )
                print("\nRecent OHLCV data:")
                print(df)
                
            except Exception as e:
                print(f"Error during test: {e}")