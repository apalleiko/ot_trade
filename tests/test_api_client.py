"""
Tests for the Twelve Data API client.
"""
import os
import time
import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
from datetime import datetime, timedelta

# Import modules to test
from ohlcv_transport.api_client import TwelveDataClient, RateLimiter, APICache
from ohlcv_transport.config import ConfigManager


class TestRateLimiter(unittest.TestCase):
    """Tests for the RateLimiter class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a rate limiter with strict limits for testing
        self.rate_limiter = RateLimiter(calls_per_minute=3, calls_per_day=10)
    
    def test_initialization(self):
        """Test initialization of RateLimiter."""
        self.assertEqual(self.rate_limiter.calls_per_minute, 3)
        self.assertEqual(self.rate_limiter.calls_per_day, 10)
        self.assertEqual(len(self.rate_limiter.minute_timestamps), 0)
        self.assertEqual(len(self.rate_limiter.day_timestamps), 0)
    
    def test_wait_if_needed_under_limit(self):
        """Test that no wait is needed when under the rate limit."""
        start_time = time.time()
        
        # Make a few requests that are under the limit
        for _ in range(2):
            self.rate_limiter.wait_if_needed()
        
        elapsed_time = time.time() - start_time
        
        # The elapsed time should be very small since no waiting is needed
        self.assertLess(elapsed_time, 0.1)
        self.assertEqual(len(self.rate_limiter.minute_timestamps), 2)
    
    @patch('time.sleep', return_value=None)
    def test_wait_if_needed_at_limit(self, mock_sleep):
        """Test that waiting is triggered when at the rate limit."""
        # Manually add timestamps to simulate being at the rate limit
        current_time = time.time()
        self.rate_limiter.minute_timestamps = [
            current_time - 30,  # 30 seconds ago
            current_time - 15,  # 15 seconds ago
            current_time - 5    # 5 seconds ago
        ]
        
        # This should trigger a wait
        self.rate_limiter.wait_if_needed()
        
        # Verify that time.sleep was called with a value around 55 seconds
        # (60 - 5 seconds since the last request)
        mock_sleep.assert_called_once()
        wait_time = mock_sleep.call_args[0][0]
        self.assertGreater(wait_time, 50)  # Should wait at least 50 seconds
        self.assertLess(wait_time, 60)     # But less than 60 seconds
    
    def test_cleanup_old_timestamps(self):
        """Test that old timestamps are cleaned up."""
        current_time = time.time()
        
        # Add some old timestamps
        self.rate_limiter.minute_timestamps = [
            current_time - 120,  # 2 minutes ago (should be removed)
            current_time - 30,   # 30 seconds ago (should be kept)
            current_time - 10    # 10 seconds ago (should be kept)
        ]
        
        self.rate_limiter.wait_if_needed()
        
        # Only the recent timestamps plus the new one should remain
        self.assertEqual(len(self.rate_limiter.minute_timestamps), 3)
        
        # All timestamps should be within the last minute
        for ts in self.rate_limiter.minute_timestamps:
            self.assertGreater(ts, current_time - 60)


class TestAPICache(unittest.TestCase):
    """Tests for the APICache class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a temporary directory for the cache
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'test_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create the cache with a short expiry time for testing
        self.cache = APICache(cache_dir=self.cache_dir, expiry_hours=1)
        
        # Test data
        self.test_data = {
            'values': [
                {'datetime': '2024-04-01 09:30:00', 'open': '100.0', 'high': '101.0', 'low': '99.0', 'close': '100.5', 'volume': '1000'},
                {'datetime': '2024-04-01 09:31:00', 'open': '100.5', 'high': '102.0', 'low': '100.0', 'close': '101.0', 'volume': '1500'}
            ]
        }
    
    def tearDown(self):
        """Clean up after the test case."""
        # Remove the test cache directory
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            os.rmdir(self.cache_dir)
    
    def test_cache_operations(self):
        """Test setting and getting cached data."""
        cache_key = 'test_key'
        
        # Initially, the cache should be empty
        self.assertIsNone(self.cache.get(cache_key))
        
        # Set some data in the cache
        self.cache.set(cache_key, self.test_data)
        
        # Now we should be able to retrieve it
        cached_data = self.cache.get(cache_key)
        self.assertEqual(cached_data, self.test_data)
    
    def test_cache_expiry(self):
        """Test that cached data expires."""
        cache_key = 'test_expiry'
        
        # Set some data in the cache with a timestamp in the past
        cache_file = self.cache._get_cache_file_path(cache_key)
        cache_data = {
            'timestamp': time.time() - 3600 * 2,  # 2 hours ago (expired)
            'data': self.test_data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # The data should be considered expired
        self.assertIsNone(self.cache.get(cache_key))
        
        # Set fresh data
        self.cache.set(cache_key, self.test_data)
        
        # Now we should be able to retrieve it
        self.assertIsNotNone(self.cache.get(cache_key))


class TestTwelveDataClient(unittest.TestCase):
    """Tests for the TwelveDataClient class."""
    
    @patch('ohlcv_transport.config.ConfigManager.get')
    def setUp(self, mock_get):
        """Set up the test case with mocked dependencies."""
        # Mock config values
        mock_get.side_effect = lambda key, default=None: {
            'api.twelve_data.base_url': 'https://api.twelvedata.com',
            'api.twelve_data.api_key': 'test_api_key',
            'api.twelve_data.rate_limit': {'calls_per_minute': 8, 'calls_per_day': 800},
            'system.data_cache_expiry_hours': 24
        }.get(key, default)
        
        # Create the client with mocked dependencies
        self.client = TwelveDataClient()
        
        # Mock the session and cache
        self.client.session = MagicMock()
        self.client.cache = MagicMock()
        
        # Test data
        self.time_series_data = {
            'meta': {
                'symbol': 'AAPL',
                'interval': '1min',
                'currency': 'USD',
                'exchange_timezone': 'America/New_York',
                'exchange': 'NASDAQ',
                'type': 'Common Stock'
            },
            'values': [
                {'datetime': '2024-04-01 09:30:00', 'open': '100.0', 'high': '101.0', 'low': '99.0', 'close': '100.5', 'volume': '1000'},
                {'datetime': '2024-04-01 09:31:00', 'open': '100.5', 'high': '102.0', 'low': '100.0', 'close': '101.0', 'volume': '1500'}
            ],
            'status': 'ok'
        }
        
        self.quote_data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc',
            'exchange': 'NASDAQ',
            'datetime': '2024-04-01 09:35:00',
            'open': '100.0',
            'high': '102.0',
            'low': '99.0',
            'close': '101.5',
            'volume': '5000',
            'previous_close': '100.0',
            'change': '1.5',
            'percent_change': '1.5',
            'status': 'ok'
        }
        
        self.price_data = {
            'price': '101.5',
            'status': 'ok'
        }
    
    @patch('ohlcv_transport.api_client.RateLimiter.wait_if_needed')
    def test_make_request(self, mock_wait):
        """Test making a request to the API."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = self.time_series_data
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        # Mock the cache to return None (cache miss)
        self.client.cache.get.return_value = None
        
        # Make the request
        endpoint = 'time_series'
        params = {'symbol': 'AAPL', 'interval': '1min'}
        
        result = self.client._make_request(endpoint, params)
        
        # Verify the result
        self.assertEqual(result, self.time_series_data)
        
        # Verify the cache was checked and updated
        self.client.cache.get.assert_called_once()
        self.client.cache.set.assert_called_once()
        
        # Verify rate limiting was respected
        mock_wait.assert_called_once()
        
        # Verify the correct URL and parameters were used
        self.client.session.get.assert_called_once()
        call_args = self.client.session.get.call_args
        self.assertEqual(call_args[0][0], 'https://api.twelvedata.com/time_series')
        self.assertEqual(call_args[1]['params']['symbol'], 'AAPL')
        self.assertEqual(call_args[1]['params']['interval'], '1min')
        self.assertEqual(call_args[1]['params']['apikey'], 'test_api_key')
    
    @patch('ohlcv_transport.api_client.TwelveDataClient._make_request')
    def test_get_time_series(self, mock_make_request):
        """Test retrieving time series data."""
        # Mock the _make_request method
        mock_make_request.return_value = self.time_series_data
        
        # Get time series data
        df = self.client.get_time_series(
            symbol='AAPL',
            interval='1min',
            outputsize=10
        )
        
        # Verify the result is a DataFrame with the correct data
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        
        # Verify the _make_request method was called with the correct parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[0][0], 'time_series')
        self.assertEqual(call_args[0][1]['symbol'], 'AAPL')
        self.assertEqual(call_args[0][1]['interval'], '1min')
        self.assertEqual(call_args[0][1]['outputsize'], 10)
    
    @patch('ohlcv_transport.api_client.TwelveDataClient._make_request')
    def test_get_quote(self, mock_make_request):
        """Test retrieving a quote."""
        # Mock the _make_request method
        mock_make_request.return_value = self.quote_data
        
        # Get a quote
        quote = self.client.get_quote(symbol='AAPL')
        
        # Verify the result
        self.assertEqual(quote, self.quote_data)
        
        # Verify the _make_request method was called with the correct parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[0][0], 'quote')
        self.assertEqual(call_args[0][1]['symbol'], 'AAPL')
    
    @patch('ohlcv_transport.api_client.TwelveDataClient._make_request')
    def test_get_price(self, mock_make_request):
        """Test retrieving a price."""
        # Mock the _make_request method
        mock_make_request.return_value = self.price_data
        
        # Get a price
        price = self.client.get_price(symbol='AAPL')
        
        # Verify the result
        self.assertEqual(price, 101.5)
        
        # Verify the _make_request method was called with the correct parameters
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[0][0], 'price')
        self.assertEqual(call_args[0][1]['symbol'], 'AAPL')


if __name__ == '__main__':
    unittest.main()