"""
Tests for the signal generators module.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ohlcv_transport.signals.generators import (
    calculate_base_imbalance_signal,
    calculate_multi_timeframe_signal,
    add_signal_features,
    apply_kalman_filter,
    apply_wavelet_denoising,
    apply_exponential_smoothing,
    generate_trading_signal,
    apply_minimum_holding_period,
    calculate_trading_metrics,
    process_signal_pipeline
)


class TestSignalGenerators(unittest.TestCase):
    """Tests for the signal generators module."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # Create synthetic Wasserstein distances with trend and noise
        np.random.seed(42)  # For reproducibility
        trend = np.linspace(0, 2, 100) + np.sin(np.linspace(0, 6, 100))
        noise = np.random.normal(0, 0.2, 100)
        w_distance = trend + noise
        
        # Add price data for testing trading metrics
        close = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        high = close + np.random.uniform(0, 0.5, 100)
        low = close - np.random.uniform(0, 0.5, 100)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'w_distance': w_distance,
            'close': close,
            'high': high,
            'low': low
        }, index=dates)
        
        # Add some additional features
        self.df['imbalance_signal'] = np.random.normal(0, 1, 100)
        self.df['position'] = np.random.choice([-1, 0, 1], 100)
    
    def test_calculate_base_imbalance_signal(self):
        """Test calculating base imbalance signal."""
        # Calculate base signal
        result = calculate_base_imbalance_signal(self.df, 'w_distance', lookback=20)
        
        # Check result
        self.assertIn('imbalance_signal', result.columns)
        self.assertEqual(len(result), len(self.df))
        
        # Signal should be normalized with mean around 0 and std around 1
        # (except for the first lookback-1 values which will have higher variance)
        signal = result['imbalance_signal'].iloc[20:]
        self.assertAlmostEqual(signal.mean(), 0, delta=0.5)
        self.assertAlmostEqual(signal.std(), 1, delta=0.5)
        
        # Test with missing w_distance column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = calculate_base_imbalance_signal(empty_df, 'w_distance')
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    def test_calculate_multi_timeframe_signal(self):
        """Test calculating multi-timeframe signal."""
        # Create DataFrame with different timeframes
        df_1min = self.df.copy()
        
        # Create synthetic 5-min data (fewer rows)
        dates_5min = pd.date_range(start='2024-04-01 09:30:00', periods=20, freq='5min')
        df_5min = pd.DataFrame({
            'imbalance_signal': np.random.normal(0, 1, 20)
        }, index=dates_5min)
        
        # Create dictionary of DataFrames
        df_dict = {
            '1min': df_1min,
            '5min': df_5min
        }
        
        # Calculate multi-timeframe signal
        result = calculate_multi_timeframe_signal(df_dict)
        
        # Check result
        self.assertIn('multi_tf_signal', result.columns)
        self.assertEqual(len(result), len(df_1min))  # Should be at lowest granularity
        
        # Test with empty dictionary
        result = calculate_multi_timeframe_signal({})
        self.assertTrue(result.empty)
    
    def test_add_signal_features(self):
        """Test adding signal features."""
        # Add features
        result = add_signal_features(self.df)
        
        # Check result
        self.assertIn('imbalance_signal_change', result.columns)
        self.assertIn('imbalance_signal_accel', result.columns)
        self.assertIn('imbalance_signal_squared', result.columns)
        self.assertIn('imbalance_signal_power', result.columns)
        self.assertIn('imbalance_signal_sma5', result.columns)
        self.assertIn('imbalance_signal_sma20', result.columns)
        
        # Verify feature calculations
        # Signal squared should be positive
        self.assertTrue((result['imbalance_signal_squared'] >= 0).all())
        
        # Signal power should preserve sign
        self.assertTrue(np.allclose(
            np.sign(result['imbalance_signal']),
            np.sign(result['imbalance_signal_power'])
        ))
        
        # Test with missing signal column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = add_signal_features(empty_df)
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    def test_apply_kalman_filter(self):
        """Test applying Kalman filter."""
        # Apply Kalman filter
        result = apply_kalman_filter(self.df)
        
        # Check result
        self.assertIn('kalman_signal', result.columns)
        self.assertEqual(len(result), len(self.df))
        
        # Filtered signal should be smoother (lower standard deviation of changes)
        orig_diff_std = self.df['imbalance_signal'].diff().std()
        kalman_diff_std = result['kalman_signal'].diff().std()
        self.assertLess(kalman_diff_std, orig_diff_std)
        
        # Test with missing signal column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = apply_kalman_filter(empty_df)
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    @patch('ohlcv_transport.signals.generators.pywt')
    def test_apply_wavelet_denoising(self, mock_pywt):
        """Test applying wavelet denoising."""
        # Mock PyWavelets functions
        mock_pywt.wavedec.return_value = [
            np.array([0.1, 0.2]),  # Approximation coefficients
            np.array([0.01, 0.02])  # Detail coefficients
        ]
        mock_pywt.threshold.return_value = np.array([0.005, 0.01])
        mock_pywt.waverec.return_value = np.random.normal(0, 0.5, 100)
        
        # Apply wavelet denoising
        result = apply_wavelet_denoising(self.df)
        
        # Check result
        self.assertIn('wavelet_signal', result.columns)
        self.assertEqual(len(result), len(self.df))
        
        # Verify that PyWavelets functions were called
        mock_pywt.wavedec.assert_called_once()
        mock_pywt.threshold.assert_called_once()
        mock_pywt.waverec.assert_called_once()
        
        # Test with missing signal column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = apply_wavelet_denoising(empty_df)
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    def test_apply_exponential_smoothing(self):
        """Test applying exponential smoothing."""
        # Apply exponential smoothing
        result = apply_exponential_smoothing(self.df, alpha=0.2)
        
        # Check result
        self.assertIn('ema_signal', result.columns)
        self.assertEqual(len(result), len(self.df))
        
        # Smoothed signal should be smoother (lower standard deviation of changes)
        orig_diff_std = self.df['imbalance_signal'].diff().std()
        ema_diff_std = result['ema_signal'].diff().std()
        self.assertLess(ema_diff_std, orig_diff_std)
        
        # Test with missing signal column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = apply_exponential_smoothing(empty_df)
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    def test_generate_trading_signal(self):
        """Test generating trading signals."""
        # Generate trading signals
        result = generate_trading_signal(
            self.df, 
            threshold_long=1.0,
            threshold_short=-1.0
        )
        
        # Check result
        self.assertIn('position', result.columns)
        self.assertEqual(len(result), len(self.df))
        
        # Verify signal generation logic
        self.assertTrue((result.loc[self.df['imbalance_signal'] >= 1.0, 'position'] == 1).all())
        self.assertTrue((result.loc[self.df['imbalance_signal'] <= -1.0, 'position'] == -1).all())
        self.assertTrue((result.loc[(self.df['imbalance_signal'] > -1.0) & 
                                   (self.df['imbalance_signal'] < 1.0), 'position'] == 0).all())
        
        # Test with missing signal column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = generate_trading_signal(empty_df)
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    def test_apply_minimum_holding_period(self):
        """Test applying minimum holding period."""
        # Create test DataFrame with oscillating positions
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=20, freq='1min')
        positions = [0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0]
        df = pd.DataFrame({'position': positions}, index=dates)
        
        # Apply minimum holding period
        result = apply_minimum_holding_period(df, min_periods=3)
        
        # Check that positions are held for at least the minimum period
        modified_positions = result['position'].values
        
        # Count how many times position changes
        position_changes = np.sum(np.diff(modified_positions) != 0)
        original_changes = np.sum(np.diff(positions) != 0)
        
        # Should have fewer position changes after applying minimum holding period
        self.assertLess(position_changes, original_changes)
        
        # Check specific cases (full verification is complex due to the algorithm)
        # Find the first position change and verify it's held for at least min_periods
        for i in range(1, len(positions)):
            if modified_positions[i] != modified_positions[i-1]:
                position = modified_positions[i]
                # Verify this position is held for at least min_periods
                if i + 3 <= len(modified_positions):
                    self.assertTrue(all(modified_positions[i:i+3] == position))
                break
        
        # Test with missing position column
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = apply_minimum_holding_period(empty_df)
        self.assertEqual(result.equals(empty_df), True)  # Should return original DataFrame
    
    def test_calculate_trading_metrics(self):
        """Test calculating trading metrics."""
        # Generate some test data with known properties
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # Create synthetic price data with trend
        close = 100 + np.linspace(0, 10, 100)  # Price increases from 100 to 110
        
        # Create positions: first half long, second half short
        positions = np.array([1] * 50 + [-1] * 50)
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': close,
            'position': positions
        }, index=dates)
        
        # Calculate metrics
        metrics = calculate_trading_metrics(df)
        
        # Check metrics
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # First half should be profitable (price trend up, long position)
        # Second half should be unprofitable (price trend up, short position)
        # Total return should be close to zero or slightly positive due to timing effects
        
        # Test with missing columns
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        metrics = calculate_trading_metrics(empty_df)
        self.assertEqual(len(metrics), 0)  # Should return empty dictionary
    
    def test_process_signal_pipeline(self):
        """Test complete signal processing pipeline."""
        # Process signal pipeline
        result = process_signal_pipeline(
            self.df,
            smooth_method='kalman',
            threshold_long=1.0,
            threshold_short=-1.0,
            min_holding=3
        )
        
        # Check result has necessary columns
        self.assertIn('imbalance_signal', result.columns)
        self.assertIn('kalman_signal', result.columns)
        self.assertIn('position', result.columns)
        
        # Test with adaptive thresholds
        result_adaptive = process_signal_pipeline(
            self.df,
            smooth_method='kalman',
            threshold_long=1.0,
            threshold_short=-1.0,
            min_holding=3,
            adaptive_threshold=True
        )
        
        # Should have adaptive threshold columns
        self.assertIn('signal_regime', result_adaptive.columns)
        self.assertIn('threshold_upper', result_adaptive.columns)
        self.assertIn('threshold_lower', result_adaptive.columns)


if __name__ == '__main__':
    unittest.main()