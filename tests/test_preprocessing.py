"""
Tests for the OHLCV data preprocessing module.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pywt

from ohlcv_transport.data.preprocessing import (
    clean_ohlcv,
    detect_outliers,
    replace_outliers,
    wavelet_denoise,
    denoise_ohlcv,
    calculate_derived_features,
    prepare_data_for_transport,
    process_ohlcv_data
)


class TestPreprocessing(unittest.TestCase):
    """Tests for the data preprocessing utilities."""
    
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
        self.df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        # Add some known anomalies for testing
        # 1. Add a missing value
        self.df.loc[dates[10], 'close'] = np.nan
        
        # 2. Add outliers
        self.df.loc[dates[20], 'close'] = close[20] * 1.5  # 50% higher
        self.df.loc[dates[30], 'volume'] = volume[30] * 10  # 10x higher
        
        # 3. Add invalid OHLC values
        self.df.loc[dates[40], 'low'] = high[40] + 1  # Low > High
        self.df.loc[dates[50], 'open'] = high[50] + 2  # Open > High
        
        # 4. Add zero volume
        self.df.loc[dates[60], 'volume'] = 0
    
    def test_clean_ohlcv(self):
        """Test the clean_ohlcv function."""
        # Clean the data
        cleaned_df = clean_ohlcv(self.df)
        
        # Check that NaN values were filled
        self.assertFalse(cleaned_df['close'].isna().any())
        
        # Check that invalid OHLC values were fixed
        self.assertLessEqual(cleaned_df.loc[self.df.index[40], 'low'], cleaned_df.loc[self.df.index[40], 'high'])
        self.assertLessEqual(cleaned_df.loc[self.df.index[50], 'open'], cleaned_df.loc[self.df.index[50], 'high'])
        
        # Check that zero volume was handled
        if 'volume' in cleaned_df.columns:
            self.assertTrue(np.isnan(cleaned_df.loc[self.df.index[60], 'volume']))
    
    def test_detect_outliers(self):
        """Test the detect_outliers function."""
        # Detect outliers
        outliers = detect_outliers(self.df)
        
        # Check that the known outliers were detected
        self.assertTrue(outliers.loc[self.df.index[20], 'close'])
        self.assertTrue(outliers.loc[self.df.index[30], 'volume'])
        
        # Check outlier count (we expect at least these two outliers)
        self.assertGreaterEqual(outliers['close'].sum(), 1)
        self.assertGreaterEqual(outliers['volume'].sum(), 1)
    
    def test_replace_outliers(self):
        """Test the replace_outliers function."""
        # Detect outliers
        outliers = detect_outliers(self.df)
        
        # Replace outliers
        replaced_df = replace_outliers(self.df, outliers)
        
        # Check that the known outliers were replaced (values should be different)
        self.assertNotEqual(self.df.loc[self.df.index[20], 'close'], replaced_df.loc[self.df.index[20], 'close'])
        self.assertNotEqual(self.df.loc[self.df.index[30], 'volume'], replaced_df.loc[self.df.index[30], 'volume'])
        
        # The replaced values should be more reasonable (closer to neighbors)
        close_neighbors = self.df['close'].iloc[18:23].drop(self.df.index[20])
        close_mean = close_neighbors.mean()
        self.assertLess(abs(replaced_df.loc[self.df.index[20], 'close'] - close_mean), 
                      abs(self.df.loc[self.df.index[20], 'close'] - close_mean))
        
        volume_neighbors = self.df['volume'].iloc[28:33].drop(self.df.index[30])
        volume_mean = volume_neighbors.mean()
        self.assertLess(abs(replaced_df.loc[self.df.index[30], 'volume'] - volume_mean), 
                      abs(self.df.loc[self.df.index[30], 'volume'] - volume_mean))
    
    def test_wavelet_denoise(self):
        """Test the wavelet_denoise function."""
        # Create noisy data
        np.random.seed(42)
        x = np.linspace(0, 1, 100)
        original = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
        noise = np.random.normal(0, 0.1, 100)
        noisy = original + noise
        
        # Apply wavelet denoising
        denoised = wavelet_denoise(noisy, wavelet='db4', level=3)
        
        # The denoised signal should be closer to the original than the noisy signal
        original_mse = np.mean((original - noisy) ** 2)
        denoised_mse = np.mean((original - denoised) ** 2)
        self.assertLess(denoised_mse, original_mse)
        
        # Test with pandas Series
        noisy_series = pd.Series(noisy, index=x)
        denoised_series = wavelet_denoise(noisy_series)
        self.assertIsInstance(denoised_series, pd.Series)
        self.assertEqual(len(denoised_series), len(noisy_series))
        
        # Test with NaN values
        noisy_with_nan = noisy.copy()
        noisy_with_nan[5] = np.nan
        denoised_with_nan = wavelet_denoise(noisy_with_nan)
        self.assertFalse(np.isnan(denoised_with_nan).any())
    
    def test_denoise_ohlcv(self):
        """Test the denoise_ohlcv function."""
        # Apply wavelet denoising to OHLCV data
        denoised_df = denoise_ohlcv(self.df)
        
        # Check that the output has the same shape
        self.assertEqual(denoised_df.shape, self.df.shape)
        
        # Check that OHLC relationship is preserved
        for i in range(len(denoised_df)):
            self.assertLessEqual(denoised_df['low'].iloc[i], denoised_df['high'].iloc[i])
            self.assertLessEqual(denoised_df['low'].iloc[i], denoised_df['open'].iloc[i])
            self.assertLessEqual(denoised_df['low'].iloc[i], denoised_df['close'].iloc[i])
            self.assertGreaterEqual(denoised_df['high'].iloc[i], denoised_df['open'].iloc[i])
            self.assertGreaterEqual(denoised_df['high'].iloc[i], denoised_df['close'].iloc[i])
        
        # The denoised data should vary less than the original
        # (has lower standard deviation of returns)
        original_std = np.std(np.diff(self.df['close']))
        denoised_std = np.std(np.diff(denoised_df['close']))
        self.assertLess(denoised_std, original_std)
    
    def test_calculate_derived_features(self):
        """Test the calculate_derived_features function."""
        # Calculate derived features
        features_df = calculate_derived_features(self.df)
        
        # Check that the expected features were added
        expected_features = [
            'norm_range', 'volume_ratio', 'log_return', 
            'sma20', 'sma50', 'volatility',
            'typical_price', 'avg_price', 'price_velocity',
            'vwap', 'rel_volume', 'norm_day_position'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
        
        # Check some relationships that should hold
        # Normalized range should be positive
        self.assertTrue((features_df['norm_range'] >= 0).all())
        
        # Typical price should be between low and high
        self.assertTrue((features_df['typical_price'] >= features_df['low']).all())
        self.assertTrue((features_df['typical_price'] <= features_df['high']).all())
        
        # Average price should also be between low and high
        self.assertTrue((features_df['avg_price'] >= features_df['low']).all())
        self.assertTrue((features_df['avg_price'] <= features_df['high']).all())
        
        # Normalized day position should be between 0 and 1
        self.assertTrue((features_df['norm_day_position'] >= 0).all())
        self.assertTrue((features_df['norm_day_position'] <= 1).all())
    
    def test_prepare_data_for_transport(self):
        """Test the prepare_data_for_transport function."""
        # Prepare data for transport
        price_grids, supply_weights, demand_weights = prepare_data_for_transport(self.df, num_price_points=50)
        
        # Check the output shapes
        self.assertEqual(len(price_grids), len(self.df) - 4)  # 4 invalid rows
        self.assertEqual(len(supply_weights), len(price_grids))
        self.assertEqual(len(demand_weights), len(price_grids))
        
        # Check that each grid and weight vector has the right length
        self.assertEqual(len(price_grids[0]), 50)
        self.assertEqual(len(supply_weights[0]), 50)
        self.assertEqual(len(demand_weights[0]), 50)
        
        # Weights should sum to approximately the volume
        for i in range(len(supply_weights)):
            self.assertAlmostEqual(np.sum(supply_weights[i]), self.df['volume'].iloc[i], delta=self.df['volume'].iloc[i] * 0.01)
            self.assertAlmostEqual(np.sum(demand_weights[i]), self.df['volume'].iloc[i], delta=self.df['volume'].iloc[i] * 0.01)
        
        # Price grids should span the range from low to high
        for i in range(len(price_grids)):
            idx = i
            while idx < len(self.df) and np.isnan(self.df['low'].iloc[idx]):
                idx += 1
            if idx >= len(self.df):
                continue
                
            low = self.df['low'].iloc[idx]
            high = self.df['high'].iloc[idx]
            self.assertLessEqual(price_grids[i][0], low * 0.9)  # Grid extends below low
            self.assertGreaterEqual(price_grids[i][-1], high * 1.1)  # Grid extends above high
    
    def test_process_ohlcv_data(self):
        """Test the complete process_ohlcv_data function."""
        # Process the data
        processed_df = process_ohlcv_data(self.df)
        
        # Check that the data was cleaned, outliers handled, and features calculated
        self.assertEqual(processed_df.shape[0], self.df.shape[0])
        self.assertGreater(processed_df.shape[1], self.df.shape[1])  # Should have added features
        
        # Check for missing values in important columns
        for col in ['open', 'high', 'low', 'close']:
            self.assertFalse(processed_df[col].isna().any())
        
        # The original outliers should have been fixed
        self.assertNotEqual(self.df.loc[self.df.index[20], 'close'], processed_df.loc[self.df.index[20], 'close'])
        
        # The invalid OHLC relationships should have been fixed
        self.assertLessEqual(processed_df.loc[self.df.index[40], 'low'], processed_df.loc[self.df.index[40], 'high'])
        self.assertLessEqual(processed_df.loc[self.df.index[50], 'open'], processed_df.loc[self.df.index[50], 'high'])
        
        # Should have all the derived features
        expected_features = [
            'norm_range', 'volume_ratio', 'log_return', 'sma20', 'volatility'
        ]
        for feature in expected_features:
            self.assertIn(feature, processed_df.columns)


if __name__ == '__main__':
    unittest.main()