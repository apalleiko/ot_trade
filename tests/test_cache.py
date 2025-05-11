"""
Tests for the OHLCV data caching module.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import json
from pathlib import Path
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ohlcv_transport.data.cache import OHLCVCache, DatasetManager


class TestOHLCVCache(unittest.TestCase):
    """Tests for the OHLCVCache class."""
    
    def setUp(self):
        """Set up test data and temporary cache directory."""
        # Create a temporary directory for the cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache = OHLCVCache(cache_dir=self.temp_dir)
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # Base price series with some randomness
        np.random.seed(42)  # For reproducibility
        close = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        
        # Create OHLC data with realistic relationships
        high = close + np.random.uniform(0, 0.5, 100)
        low = close - np.random.uniform(0, 0.5, 100)
        open_price = low + np.random.uniform(0, 1, 100) * (high - low)
        
        # Volume with some randomness
        volume = 1000 + np.random.uniform(0, 500, 100)
        
        # Create DataFrame
        self.sample_df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    def tearDown(self):
        """Clean up temporary files after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_generate_cache_key(self):
        """Test generating a cache key."""
        # Test with minimal parameters
        key1 = self.cache._generate_cache_key('AAPL', '1min')
        self.assertTrue(key1.startswith('AAPL_1min_'))
        
        # Test with date parameters
        key2 = self.cache._generate_cache_key('AAPL', '1min', '2024-04-01', '2024-04-02')
        self.assertTrue(key2.startswith('AAPL_1min_'))
        
        # Different parameters should generate different keys
        self.assertNotEqual(key1, key2)
        
        # Same parameters should generate the same key
        key3 = self.cache._generate_cache_key('AAPL', '1min')
        self.assertEqual(key1, key3)
        
        # Case should not matter
        key4 = self.cache._generate_cache_key('aapl', '1MIN')
        self.assertEqual(key1, key4)
    
    def test_save_and_load(self):
        """Test saving and loading data from cache."""
        # Save data to cache
        symbol = 'AAPL'
        interval = '1min'
        cache_key = self.cache.save(self.sample_df, symbol, interval)
        
        # Verify the cache key is not empty
        self.assertTrue(cache_key)
        
        # Check that the files were created
        cache_path = self.cache._get_cache_path(cache_key)
        metadata_path = self.cache._get_metadata_path(cache_key)
        self.assertTrue(cache_path.exists())
        self.assertTrue(metadata_path.exists())
        
        # Load the data back from cache
        loaded_df = self.cache.load(symbol, interval)
        
        # Verify the data is the same
        pd.testing.assert_frame_equal(self.sample_df, loaded_df)
        
        # Check metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['symbol'], symbol)
        self.assertEqual(metadata['interval'], interval)
        self.assertEqual(metadata['num_bars'], len(self.sample_df))
    
    def test_cache_expiry(self):
        """Test cache expiry functionality."""
        # Save data with a very short expiry time
        symbol = 'AAPL'
        interval = '1min'
        expiry_hours = 0.001  # 3.6 seconds
        cache_key = self.cache.save(self.sample_df, symbol, interval, expiry_hours=expiry_hours)
        
        # Wait for the cache to expire
        time.sleep(4)
        
        # Try to load the expired data
        loaded_df = self.cache.load(symbol, interval)
        
        # Should return None because the cache has expired
        self.assertIsNone(loaded_df)
        
        # Try loading without checking expiry
        loaded_df = self.cache.load(symbol, interval, check_expiry=False)
        
        # Should return the data even though it's expired
        pd.testing.assert_frame_equal(self.sample_df, loaded_df)
    
    def test_delete(self):
        """Test deleting cache entries."""
        # Save data to cache
        symbol = 'AAPL'
        interval = '1min'
        cache_key = self.cache.save(self.sample_df, symbol, interval)
        
        # Verify the cache entry exists
        self.assertTrue(self.cache._get_cache_path(cache_key).exists())
        self.assertTrue(self.cache._get_metadata_path(cache_key).exists())
        
        # Delete the cache entry
        success = self.cache.delete(symbol, interval)
        
        # Verify the deletion was successful
        self.assertTrue(success)
        self.assertFalse(self.cache._get_cache_path(cache_key).exists())
        self.assertFalse(self.cache._get_metadata_path(cache_key).exists())
        
        # Try to load the deleted data
        loaded_df = self.cache.load(symbol, interval)
        self.assertIsNone(loaded_df)
    
    def test_clean_expired(self):
        """Test cleaning expired cache entries."""
        # Save data with different expiry times
        symbol = 'AAPL'
        
        # Entry 1: Already expired
        self.cache.save(self.sample_df, symbol, '1min', expiry_hours=0.001)
        time.sleep(2)  # Wait for it to expire
        
        # Entry 2: Not expired
        self.cache.save(self.sample_df, symbol, '5min', expiry_hours=1)
        
        # Entry 3: Another expired one
        self.cache.save(self.sample_df, symbol, '15min', expiry_hours=0.001)
        time.sleep(2)  # Wait for it to expire
        
        # Clean expired entries
        deleted_count = self.cache.clean_expired()
        
        # Should have deleted 2 entries
        self.assertEqual(deleted_count, 2)
        
        # Verify the expired entries are gone
        self.assertIsNone(self.cache.load(symbol, '1min'))
        self.assertIsNone(self.cache.load(symbol, '15min'))
        
        # But the non-expired entry should still be there
        loaded_df = self.cache.load(symbol, '5min')
        self.assertIsNotNone(loaded_df)
        pd.testing.assert_frame_equal(self.sample_df, loaded_df)
    
    def test_list_cache_entries(self):
        """Test listing cache entries."""
        # Save multiple entries
        self.cache.save(self.sample_df, 'AAPL', '1min')
        self.cache.save(self.sample_df, 'MSFT', '5min')
        self.cache.save(self.sample_df, 'GOOG', '15min')
        
        # List the entries
        entries = self.cache.list_cache_entries()
        
        # Should have 3 entries
        self.assertEqual(len(entries), 3)
        
        # Check entry details
        symbols = [entry['symbol'] for entry in entries]
        intervals = [entry['interval'] for entry in entries]
        
        self.assertIn('AAPL', symbols)
        self.assertIn('MSFT', symbols)
        self.assertIn('GOOG', symbols)
        
        self.assertIn('1min', intervals)
        self.assertIn('5min', intervals)
        self.assertIn('15min', intervals)
        
        # Entries should have age and expired status
        for entry in entries:
            self.assertIn('age_hours', entry)
            self.assertIn('expired', entry)
            self.assertIn('cache_key', entry)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        # Try to save an empty DataFrame
        empty_df = pd.DataFrame()
        cache_key = self.cache.save(empty_df, 'AAPL', '1min')
        
        # Should return an empty string
        self.assertEqual(cache_key, "")


class TestDatasetManager(unittest.TestCase):
    """Tests for the DatasetManager class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # Base price series with some randomness
        np.random.seed(42)  # For reproducibility
        close = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        
        # Create OHLC data with realistic relationships
        high = close + np.random.uniform(0, 0.5, 100)
        low = close - np.random.uniform(0, 0.5, 100)
        open_price = low + np.random.uniform(0, 1, 100) * (high - low)
        
        # Volume with some randomness
        volume = 1000 + np.random.uniform(0, 500, 100)
        
        # Create DataFrame
        self.sample_df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    @patch('ohlcv_transport.data.cache.OHLCVCache')
    def test_get_dataset(self, mock_cache_class):
        """Test getting a dataset."""
        # Setup mock
        mock_cache = mock_cache_class.return_value
        mock_cache.load.return_value = self.sample_df
        
        # Create dataset manager with mocked cache
        with patch('ohlcv_transport.data.cache.OHLCVCache', return_value=mock_cache):
            manager = DatasetManager()
        
        # Get a dataset
        result = manager.get_dataset('AAPL', '1min')
        
        # Verify the cache was used
        mock_cache.load.assert_called_once()
        
        # Check the result
        pd.testing.assert_frame_equal(result, self.sample_df)
        
        # Verify the dataset was tracked
        self.assertEqual(len(manager.datasets), 1)
        self.assertIn('AAPL_1min', manager.datasets)
        self.assertEqual(manager.datasets['AAPL_1min']['symbol'], 'AAPL')
        self.assertEqual(manager.datasets['AAPL_1min']['interval'], '1min')
        self.assertEqual(manager.datasets['AAPL_1min']['num_bars'], 100)
    
    @patch('ohlcv_transport.data.cache.OHLCVCache')
    @patch('ohlcv_transport.data.retrieval.get_symbol_ohlcv')
    def test_get_dataset_with_cache_miss(self, mock_get_symbol, mock_cache_class):
        """Test getting a dataset with a cache miss."""
        # Setup mocks
        mock_cache = mock_cache_class.return_value
        mock_cache.load.return_value = None  # Cache miss
        mock_get_symbol.return_value = self.sample_df
        
        # Create dataset manager with mocked dependencies
        with patch('ohlcv_transport.data.cache.OHLCVCache', return_value=mock_cache):
            manager = DatasetManager()
        
        # Get a dataset
        result = manager.get_dataset('AAPL', '1min')
        
        # Verify the cache was checked
        mock_cache.load.assert_called_once()
        
        # Verify the data was retrieved from the API
        mock_get_symbol.assert_called_once()
        
        # Verify the data was saved to cache
        mock_cache.save.assert_called_once()
        
        # Check the result
        pd.testing.assert_frame_equal(result, self.sample_df)
    
    @patch('ohlcv_transport.data.cache.OHLCVCache')
    @patch('ohlcv_transport.data.retrieval.get_symbol_ohlcv')
    def test_get_dataset_with_force_refresh(self, mock_get_symbol, mock_cache_class):
        """Test getting a dataset with force refresh."""
        # Setup mocks
        mock_cache = mock_cache_class.return_value
        mock_get_symbol.return_value = self.sample_df
        
        # Create dataset manager with mocked dependencies
        with patch('ohlcv_transport.data.cache.OHLCVCache', return_value=mock_cache):
            manager = DatasetManager()
        
        # Get a dataset with force refresh
        result = manager.get_dataset('AAPL', '1min', force_refresh=True)
        
        # Verify the cache was not checked
        mock_cache.load.assert_not_called()
        
        # Verify the data was retrieved from the API
        mock_get_symbol.assert_called_once()
        
        # Verify the data was saved to cache
        mock_cache.save.assert_called_once()
        
        # Check the result
        pd.testing.assert_frame_equal(result, self.sample_df)
    
    @patch('ohlcv_transport.data.cache.OHLCVCache')
    @patch('ohlcv_transport.data.retrieval.get_symbol_ohlcv')
    def test_refresh_dataset(self, mock_get_symbol, mock_cache_class):
        """Test refreshing a dataset."""
        # Setup mocks
        mock_cache = mock_cache_class.return_value
        mock_cache.load.return_value = self.sample_df
        mock_get_symbol.return_value = self.sample_df.iloc[-50:]  # Return half the data to simulate new data
        
        # Create dataset manager with mocked dependencies
        with patch('ohlcv_transport.data.cache.OHLCVCache', return_value=mock_cache):
            manager = DatasetManager()
        
        # First get a dataset to track it
        manager.get_dataset('AAPL', '1min')
        
        # Reset the mocks
        mock_cache.load.reset_mock()
        mock_get_symbol.reset_mock()
        mock_cache.save.reset_mock()
        
        # Then refresh it
        refreshed_df = manager.refresh_dataset('AAPL', '1min')
        
        # Verify the API was called with the correct parameters
        mock_get_symbol.assert_called_once()
        call_args = mock_get_symbol.call_args[1]
        self.assertEqual(call_args['symbol'], 'AAPL')
        self.assertEqual(call_args['interval'], '1min')
        self.assertTrue(call_args['use_cache'] is False)
        
        # Check that start_date is based on the last end date
        self.assertIsNotNone(call_args['start_date'])
        
        # Verify the data was saved to cache
        mock_cache.save.assert_called_once()
        
        # Check the result
        self.assertIsNotNone(refreshed_df)
        self.assertEqual(len(refreshed_df), 50)  # Should be the new data length
    
    @patch('ohlcv_transport.data.cache.OHLCVCache')
    def test_refresh_dataset_not_tracked(self, mock_cache_class):
        """Test refreshing a dataset that is not tracked."""
        # Create dataset manager with mocked dependencies
        with patch('ohlcv_transport.data.cache.OHLCVCache', return_value=mock_cache_class.return_value):
            manager = DatasetManager()
        
        # Try to refresh a dataset that is not tracked
        result = manager.refresh_dataset('AAPL', '1min')
        
        # Should return None
        self.assertIsNone(result)
    
    @patch('ohlcv_transport.data.cache.OHLCVCache')
    def test_list_datasets(self, mock_cache_class):
        """Test listing datasets."""
        # Create dataset manager with mocked dependencies
        with patch('ohlcv_transport.data.cache.OHLCVCache', return_value=mock_cache_class.return_value):
            manager = DatasetManager()
        
        # Add some datasets
        manager.datasets = {
            'AAPL_1min': {'symbol': 'AAPL', 'interval': '1min', 'num_bars': 100},
            'MSFT_5min': {'symbol': 'MSFT', 'interval': '5min', 'num_bars': 50},
            'GOOG_15min': {'symbol': 'GOOG', 'interval': '15min', 'num_bars': 20}
        }
        
        # List datasets
        datasets = manager.list_datasets()
        
        # Check the result
        self.assertEqual(len(datasets), 3)
        symbols = [ds['symbol'] for ds in datasets]
        intervals = [ds['interval'] for ds in datasets]
        
        self.assertIn('AAPL', symbols)
        self.assertIn('MSFT', symbols)
        self.assertIn('GOOG', symbols)
        
        self.assertIn('1min', intervals)
        self.assertIn('5min', intervals)
        self.assertIn('15min', intervals)


if __name__ == '__main__':
    unittest.main()