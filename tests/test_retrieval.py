"""
Tests for the OHLCV data retrieval module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ohlcv_transport.data.retrieval import (
    get_symbol_ohlcv,
    get_multi_timeframe_data,
    get_batch_symbols_data,
    download_historical_data,
    calculate_appropriate_limit,
    interval_to_seconds,
    estimate_days_per_request,
    resample_ohlcv,
    interval_to_pandas_freq,
    get_complete_dataset
)


class TestRetrieval(unittest.TestCase):
    """Tests for the data retrieval utilities."""
    
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
    
    @patch('ohlcv_transport.data.retrieval.twelve_data_client')
    def test_get_symbol_ohlcv(self, mock_client):
        """Test retrieving OHLCV data for a symbol."""
        # Setup mock
        mock_client.get_time_series.return_value = self.sample_df
        
        # Call the function
        result = get_symbol_ohlcv('AAPL', interval='1min', limit=100)
        
        # Check the result
        self.assertEqual(len(result), 100)
        self.assertEqual(list(result.columns), ['open', 'high', 'low', 'close', 'volume'])
        
        # Verify the mock was called correctly
        mock_client.get_time_series.assert_called_once_with(
            symbol='AAPL',
            interval='1min',
            start_date=None,
            end_date=None,
            outputsize=100,
            use_cache=True
        )
        
        # Test with date parameters
        start_date = '2024-04-01'
        end_date = '2024-04-02'
        mock_client.get_time_series.reset_mock()
        get_symbol_ohlcv('AAPL', interval='1min', start_date=start_date, end_date=end_date)
        
        mock_client.get_time_series.assert_called_once_with(
            symbol='AAPL',
            interval='1min',
            start_date=start_date,
            end_date=end_date,
            outputsize=5000,  # Default value when limit not specified
            use_cache=True
        )
        
        # Test with datetime objects
        start_dt = datetime(2024, 4, 1)
        end_dt = datetime(2024, 4, 2)
        mock_client.get_time_series.reset_mock()
        get_symbol_ohlcv('AAPL', interval='1min', start_date=start_dt, end_date=end_dt)
        
        mock_client.get_time_series.assert_called_once_with(
            symbol='AAPL',
            interval='1min',
            start_date='2024-04-01 00:00:00',
            end_date='2024-04-02 00:00:00',
            outputsize=5000,
            use_cache=True
        )
        
        # Test error handling
        mock_client.get_time_series.side_effect = Exception("API error")
        result = get_symbol_ohlcv('AAPL', interval='1min')
        self.assertTrue(result.empty)
    
    @patch('ohlcv_transport.data.retrieval.get_symbol_ohlcv')
    def test_get_multi_timeframe_data(self, mock_get_symbol):
        """Test retrieving data for multiple timeframes."""
        # Setup mock
        mock_get_symbol.return_value = self.sample_df
        
        # Call the function
        result = get_multi_timeframe_data('AAPL', timeframes=['1min', '5min', '15min'])
        
        # Check the result
        self.assertEqual(len(result), 3)
        self.assertIn('1min', result)
        self.assertIn('5min', result)
        self.assertIn('15min', result)
        
        # Each dataframe should have data
        for tf in result:
            self.assertEqual(len(result[tf]), 100)
        
        # Verify the mock was called correctly
        self.assertEqual(mock_get_symbol.call_count, 3)
    
    @patch('ohlcv_transport.data.retrieval.get_symbol_ohlcv')
    def test_get_batch_symbols_data(self, mock_get_symbol):
        """Test retrieving data for multiple symbols."""
        # Setup mock
        mock_get_symbol.return_value = self.sample_df
        
        # Call the function
        symbols = ['AAPL', 'MSFT', 'GOOG']
        result = get_batch_symbols_data(symbols, interval='1min', delay_between_requests=0)
        
        # Check the result
        self.assertEqual(len(result), 3)
        for symbol in symbols:
            self.assertIn(symbol, result)
            self.assertEqual(len(result[symbol]), 100)
        
        # Verify the mock was called correctly
        self.assertEqual(mock_get_symbol.call_count, 3)
        calls = mock_get_symbol.call_args_list
        for i, symbol in enumerate(symbols):
            self.assertEqual(calls[i][0][0], symbol)
    
    @patch('ohlcv_transport.data.retrieval.get_symbol_ohlcv')
    def test_download_historical_data(self, mock_get_symbol):
        """Test downloading historical data with multiple requests."""
        # Setup mock
        mock_get_symbol.return_value = self.sample_df
        
        # Call the function
        result = download_historical_data('AAPL', interval='1min', days_back=5, max_requests=2)
        
        # Check the result
        self.assertEqual(len(result), 100)  # Only one unique dataframe in this test
        
        # Verify the mock was called correctly
        self.assertEqual(mock_get_symbol.call_count, 2)
        
        # Test error handling in one of the requests
        mock_get_symbol.reset_mock()
        mock_get_symbol.side_effect = [self.sample_df, pd.DataFrame()]
        
        result = download_historical_data('AAPL', interval='1min', days_back=5, max_requests=2)
        
        # Should still return data from the successful request
        self.assertEqual(len(result), 100)
    
    def test_calculate_appropriate_limit(self):
        """Test calculating appropriate bar limit."""
        # Test with no dates (should return default)
        limit = calculate_appropriate_limit('1min')
        self.assertEqual(limit, 5000)
        
        # Test with dates
        start_date = datetime(2024, 4, 1, 9, 30)
        end_date = datetime(2024, 4, 1, 16, 0)
        
        # For 1-minute data, that's 6.5 hours = 390 minutes = 390 bars
        limit = calculate_appropriate_limit('1min', start_date, end_date)
        # Should return about 390 bars plus some margin
        self.assertTrue(390 < limit < 500)
        
        # Test with different intervals
        limit = calculate_appropriate_limit('5min', start_date, end_date)
        # Should be about 1/5 of the 1-minute limit
        self.assertTrue(70 < limit < 100)
        
        # Test with string dates
        limit = calculate_appropriate_limit('1min', '2024-04-01 09:30:00', '2024-04-01 16:00:00')
        self.assertTrue(390 < limit < 500)
    
    def test_interval_to_seconds(self):
        """Test converting interval string to seconds."""
        # Test various formats
        self.assertEqual(interval_to_seconds('1min'), 60)
        self.assertEqual(interval_to_seconds('5min'), 300)
        self.assertEqual(interval_to_seconds('1h'), 3600)
        self.assertEqual(interval_to_seconds('4h'), 14400)
        self.assertEqual(interval_to_seconds('1day'), 86400)
        self.assertEqual(interval_to_seconds('1week'), 604800)
        self.assertEqual(interval_to_seconds('1month'), 2592000)
        
        # Test default for unknown format
        self.assertEqual(interval_to_seconds('unknown'), 60)
        self.assertEqual(interval_to_seconds(''), 60)
    
    def test_estimate_days_per_request(self):
        """Test estimating days per request."""
        # Test various intervals
        self.assertEqual(estimate_days_per_request('1min'), 4)
        self.assertEqual(estimate_days_per_request('5min'), 18)
        self.assertEqual(estimate_days_per_request('15min'), 50)
        self.assertEqual(estimate_days_per_request('1h'), 210)
        self.assertEqual(estimate_days_per_request('1day'), 5000)
        
        # Test default for unknown interval
        self.assertEqual(estimate_days_per_request('unknown'), 3)
    
    def test_resample_ohlcv(self):
        """Test resampling OHLCV data."""
        # Create 1-minute data
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        df_1min = self.sample_df.copy()
        
        # Resample to 5-minute
        df_5min = resample_ohlcv(df_1min, '5min')
        
        # Check the result
        self.assertEqual(len(df_5min), 20)  # 100 minutes = 20 5-minute bars
        self.assertEqual(list(df_5min.columns), ['open', 'high', 'low', 'close', 'volume'])
        
        # Check OHLC relationships
        for i in range(0, 100, 5):
            end_idx = min(i + 5, 100)
            original_chunk = df_1min.iloc[i:end_idx]
            resampled_idx = i // 5
            if resampled_idx >= len(df_5min):
                continue
                
            # First value in chunk should match open in resampled
            self.assertEqual(original_chunk['open'].iloc[0], df_5min['open'].iloc[resampled_idx])
            
            # Max should match high
            self.assertEqual(original_chunk['high'].max(), df_5min['high'].iloc[resampled_idx])
            
            # Min should match low
            self.assertEqual(original_chunk['low'].min(), df_5min['low'].iloc[resampled_idx])
            
            # Last value should match close
            self.assertEqual(original_chunk['close'].iloc[-1], df_5min['close'].iloc[resampled_idx])
            
            # Sum should match volume
            self.assertEqual(original_chunk['volume'].sum(), df_5min['volume'].iloc[resampled_idx])
        
        # Test with invalid data
        invalid_df = pd.DataFrame({'a': [1, 2, 3]})  # No datetime index
        result = resample_ohlcv(invalid_df, '5min')
        self.assertIs(result, invalid_df)  # Should return the input unchanged
    
    def test_interval_to_pandas_freq(self):
        """Test converting interval to pandas frequency."""
        # Test various formats
        self.assertEqual(interval_to_pandas_freq('1min'), '1T')
        self.assertEqual(interval_to_pandas_freq('5min'), '5T')
        self.assertEqual(interval_to_pandas_freq('1h'), '1H')
        self.assertEqual(interval_to_pandas_freq('4h'), '4H')
        self.assertEqual(interval_to_pandas_freq('1day'), '1D')
        self.assertEqual(interval_to_pandas_freq('1week'), '1W')
        self.assertEqual(interval_to_pandas_freq('1month'), '1M')
        
        # Test invalid inputs
        self.assertIsNone(interval_to_pandas_freq(''))
        self.assertIsNone(interval_to_pandas_freq('unknown'))
    
    @patch('ohlcv_transport.data.retrieval.download_historical_data')
    @patch('ohlcv_transport.data.retrieval.process_ohlcv_data')
    def test_get_complete_dataset(self, mock_process, mock_download):
        """Test getting a complete dataset."""
        # Setup mocks
        mock_download.return_value = self.sample_df
        mock_process.return_value = self.sample_df.copy()
        
        # Call the function
        result = get_complete_dataset('AAPL', interval='1min', days_back=10)
        
        # Check the result
        self.assertEqual(len(result), 100)
        
        # Verify the mocks were called correctly
        mock_download.assert_called_once_with(
            symbol='AAPL',
            interval='1min',
            days_back=10,
            use_cache=True
        )
        mock_process.assert_called_once_with(self.sample_df)
        
        # Test with preprocessing disabled
        mock_download.reset_mock()
        mock_process.reset_mock()
        
        result = get_complete_dataset('AAPL', interval='1min', days_back=10, preprocess=False)
        
        mock_download.assert_called_once()
        mock_process.assert_not_called()
        
        # Test with empty result
        mock_download.return_value = pd.DataFrame()
        result = get_complete_dataset('AAPL', interval='1min', days_back=10)
        self.assertTrue(result.empty)
        mock_process.assert_not_called()


if __name__ == '__main__':
    unittest.main()