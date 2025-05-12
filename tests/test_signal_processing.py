"""
Tests for the signal processing module.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ohlcv_transport.signals.processing import (
    normalize_signal,
    combine_signals,
    detect_signal_regime,
    adaptive_thresholds,
    apply_signal_filter,
    fractional_differentiation,
    detect_divergence,
    atr_bands,
    signal_to_position_sizing,
    calculate_dynamic_lookback,
    ewma_crossover_signal
)


class TestSignalProcessing(unittest.TestCase):
    """Tests for the signal processing module."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # Create synthetic signals with trend, cycles, and noise
        np.random.seed(42)  # For reproducibility
        trend = np.linspace(-1, 1, 100)
        cycles = np.sin(np.linspace(0, 6 * np.pi, 100))
        noise = np.random.normal(0, 0.2, 100)
        
        signal1 = trend + cycles + noise
        signal2 = -trend + cycles + noise  # Opposite trend
        signal3 = cycles + noise  # No trend
        
        # Add price data for testing
        close = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        high = close + np.random.uniform(0, 0.5, 100)
        low = close - np.random.uniform(0, 0.5, 100)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'signal1': signal1,
            'signal2': signal2,
            'signal3': signal3,
            'close': close,
            'high': high,
            'low': low
        }, index=dates)
    
    def test_normalize_signal(self):
        """Test normalizing signals."""
        # Test z-score normalization
        result = normalize_signal(self.df['signal1'], lookback=20, z_score=True)
        
        # Check result is Series
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))
        
        # After lookback periods, signal should have mean close to 0 and std close to 1
        normalized = result.iloc[20:]
        self.assertAlmostEqual(normalized.mean(), 0, delta=0.5)
        self.assertAlmostEqual(normalized.std(), 1, delta=0.5)
        
        # Test min-max normalization
        result = normalize_signal(self.df['signal2'], lookback=20, z_score=False)
        
        # Check result
        self.assertEqual(len(result), len(self.df))
        
        # Min-max scaled values should be between -1 and 1
        self.assertTrue((result.iloc[20:] >= -1).all() and (result.iloc[20:] <= 1).all())
        
        # Test with empty series
        empty_series = pd.Series([])
        result = normalize_signal(empty_series)
        self.assertTrue(result.empty)
    
    def test_combine_signals(self):
        """Test combining signals."""
        # Create signal dictionary
        signals = {
            'signal1': self.df['signal1'],
            'signal2': self.df['signal2'],
            'signal3': self.df['signal3']
        }
        
        # Test weighted sum method
        weights = {'signal1': 0.5, 'signal2': 0.3, 'signal3': 0.2}
        result = combine_signals(signals, weights, method='weighted_sum')
        
        # Check result
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))
        
        # Test voting method
        result = combine_signals(signals, weights, method='voting')
        
        # Check result
        self.assertEqual(len(result), len(self.df))
        self.assertTrue((result >= -1).all() and (result <= 1).all())
        
        # Test max method
        result = combine_signals(signals, method='max')
        
        # Check result
        self.assertEqual(len(result), len(self.df))
        
        # Max value at each point should equal the maximum of all signals
        for i in range(len(result)):
            max_val = max(signals['signal1'].iloc[i], signals['signal2'].iloc[i], signals['signal3'].iloc[i])
            self.assertAlmostEqual(result.iloc[i], max_val)
        
        # Test min method
        result = combine_signals(signals, method='min')
        
        # Check result
        self.assertEqual(len(result), len(self.df))
        
        # Min value at each point should equal the minimum of all signals
        for i in range(len(result)):
            min_val = min(signals['signal1'].iloc[i], signals['signal2'].iloc[i], signals['signal3'].iloc[i])
            self.assertAlmostEqual(result.iloc[i], min_val)
        
        # Test with invalid method
        result = combine_signals(signals, method='invalid')
        self.assertTrue(result.empty)
        
        # Test with empty dictionary
        result = combine_signals({})
        self.assertTrue(result.empty)
    
    def test_detect_signal_regime(self):
        """Test detecting signal regimes."""
        # Create test signal with regime change
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # First half: low volatility, second half: high volatility
        low_vol = np.random.normal(0, 0.2, 50)
        high_vol = np.random.normal(0, 1.0, 50)
        signal = np.concatenate([low_vol, high_vol])
        signal_series = pd.Series(signal, index=dates)
        
        # Detect regimes
        result = detect_signal_regime(signal_series, lookback=20, threshold=1.5)
        
        # Check result
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(signal_series))
        
        # First half should be mostly regime 0 (low volatility)
        # Second half should have some regime 1 (high volatility)
        # Due to the rolling calculation, there's a transition phase
        self.assertTrue((result.iloc[:30] == 0).all())  # First 30 should be regime 0
        self.assertTrue((result.iloc[70:] == 1).any())  # Last 30 should have some regime 1
        
        # Test with empty series
        empty_series = pd.Series([])
        result = detect_signal_regime(empty_series)
        self.assertTrue(result.empty)
    
    def test_adaptive_thresholds(self):
        """Test calculating adaptive thresholds."""
        # Create test signal series
        signal_series = self.df['signal1']
        
        # Create regime series (0 = normal, 1 = high volatility)
        regime_series = pd.Series(0, index=signal_series.index)
        regime_series.iloc[50:] = 1  # Second half is high volatility regime
        
        # Calculate adaptive thresholds
        upper_thresh, lower_thresh = adaptive_thresholds(
            signal_series, 
            regime_series,
            base_threshold=1.0,
            vol_scaling=True
        )
        
        # Check results
        self.assertEqual(len(upper_thresh), len(signal_series))
        self.assertEqual(len(lower_thresh), len(signal_series))
        
        # Upper threshold should be positive, lower threshold should be negative
        self.assertTrue((upper_thresh > 0).all())
        self.assertTrue((lower_thresh < 0).all())
        
        # In high volatility regime, thresholds should be wider
        self.assertTrue(upper_thresh.iloc[60:70].mean() > upper_thresh.iloc[30:40].mean())
        self.assertTrue(lower_thresh.iloc[60:70].mean() < lower_thresh.iloc[30:40].mean())
        
        # Test without volatility scaling
        upper_thresh, lower_thresh = adaptive_thresholds(
            signal_series, 
            regime_series,
            base_threshold=1.0,
            vol_scaling=False
        )
        
        # Without vol scaling, thresholds should be fixed in each regime
        self.assertTrue((upper_thresh.iloc[:50] == 1.0).all())
        self.assertTrue((upper_thresh.iloc[50:] == 1.5).all())
        self.assertTrue((lower_thresh.iloc[:50] == -1.0).all())
        self.assertTrue((lower_thresh.iloc[50:] == -1.5).all())
        
        # Test with empty series
        empty_series = pd.Series([])
        upper, lower = adaptive_thresholds(empty_series, empty_series)
        self.assertTrue(upper.empty)
        self.assertTrue(lower.empty)
    
    def test_apply_signal_filter(self):
        """Test applying signal filters."""
        # Create test signal with noise and trend
        signal_series = self.df['signal1']
        
        # Apply low-pass filter
        result = apply_signal_filter(signal_series, filter_type='lowpass', cutoff=0.1)
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # Filtered signal should be smoother (lower standard deviation of differences)
        orig_diff_std = signal_series.diff().std()
        filt_diff_std = result.diff().std()
        self.assertLess(filt_diff_std, orig_diff_std)
        
        # Apply high-pass filter
        result = apply_signal_filter(signal_series, filter_type='highpass', cutoff=0.3)
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # High-pass filter should remove trend and low-frequency components
        self.assertAlmostEqual(result.mean(), 0, delta=0.5)  # Should have zero mean
        
        # Apply bandpass filter
        result = apply_signal_filter(
            signal_series, 
            filter_type='bandpass', 
            cutoff=(0.1, 0.3)
        )
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # Test with invalid filter type
        result = apply_signal_filter(signal_series, filter_type='invalid')
        self.assertTrue(result.equals(signal_series))  # Should return original series
        
        # Test with empty series
        empty_series = pd.Series([])
        result = apply_signal_filter(empty_series)
        self.assertTrue(result.empty)
    
    def test_fractional_differentiation(self):
        """Test fractional differentiation."""
        # Create test signal with strong trend
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        trend = np.linspace(0, 10, 100)  # Strong linear trend
        signal_series = pd.Series(trend, index=dates)
        
        # Apply fractional differentiation
        result = fractional_differentiation(signal_series, order=0.5, window=10)
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # First window values should be zero (not enough history)
        self.assertTrue((result.iloc[:10] == 0).all())
        
        # Fractionally differentiated series should be more stationary
        # Correlation with original series should be reduced
        orig_corr = signal_series.iloc[20:].autocorr(lag=1)
        diff_corr = result.iloc[20:].autocorr(lag=1)
        self.assertLess(diff_corr, orig_corr)
        
        # Test with empty series
        empty_series = pd.Series([])
        result = fractional_differentiation(empty_series)
        self.assertTrue(result.empty)
    
    def test_detect_divergence(self):
        """Test detecting divergences."""
        # Create price and signal series with divergences
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # Price makes higher highs
        price = np.linspace(100, 110, 100)
        
        # Signal makes lower highs (bearish divergence)
        signal = np.linspace(1, 0, 100) + np.sin(np.linspace(0, 6 * np.pi, 100)) * 0.2
        
        price_series = pd.Series(price, index=dates)
        signal_series = pd.Series(signal, index=dates)
        
        # Detect divergences
        result = detect_divergence(price_series, signal_series, window=10)
        
        # Check result
        self.assertEqual(len(result), len(price_series))
        
        # Should have some bearish divergences (when price is at local max, signal is lower)
        self.assertTrue((result == -1).any())
        
        # Test with empty series
        empty_series = pd.Series([])
        result = detect_divergence(empty_series, empty_series)
        self.assertTrue(result.empty)
    
    def test_atr_bands(self):
        """Test calculating ATR bands."""
        # Create test DataFrame with OHLC data
        upper_band, lower_band = atr_bands(
            self.df,
            price_col='close',
            high_col='high',
            low_col='low',
            period=14,
            multiplier=2.0
        )
        
        # Check results
        self.assertEqual(len(upper_band), len(self.df))
        self.assertEqual(len(lower_band), len(self.df))
        
        # Upper band should be above close price, lower band below
        self.assertTrue((upper_band > self.df['close']).all())
        self.assertTrue((lower_band < self.df['close']).all())
        
        # Distance from bands to price should be related to volatility
        # Calculate volatility for each segment
        vol_first = self.df['close'].iloc[:50].diff().std()
        vol_last = self.df['close'].iloc[50:].diff().std()
        
        # If second half is more volatile, band distance should be larger
        if vol_last > vol_first:
            band_dist_first = (upper_band.iloc[20:50] - self.df['close'].iloc[20:50]).mean()
            band_dist_last = (upper_band.iloc[70:] - self.df['close'].iloc[70:]).mean()
            self.assertGreater(band_dist_last, band_dist_first)
        
        # Test with missing columns
        empty_df = pd.DataFrame({'other_col': [1, 2, 3]})
        upper, lower = atr_bands(empty_df)
        self.assertTrue(upper.empty)
        self.assertTrue(lower.empty)
    
    def test_signal_to_position_sizing(self):
        """Test converting signal to position sizes."""
        # Create test signal series
        signal_series = self.df['signal1']
        
        # Convert to position sizes
        result = signal_to_position_sizing(
            signal_series,
            base_position=1.0,
            max_position=2.0,
            threshold=1.0
        )
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # Position size should be proportional to signal magnitude
        self.assertTrue((result * signal_series >= 0).all())  # Same sign as signal
        
        # Position should be limited by max_position
        self.assertTrue((result.abs() <= 2.0).all())
        
        # For signals at threshold, position should be base_position
        threshold_idx = (signal_series.abs() - 1.0).abs() < 0.01
        if threshold_idx.any():
            positions_at_threshold = result[threshold_idx]
            signals_at_threshold = signal_series[threshold_idx]
            self.assertTrue(all(
                abs(pos) - 1.0 < 0.01 for pos in positions_at_threshold
            ))
        
        # Test with empty series
        empty_series = pd.Series([])
        result = signal_to_position_sizing(empty_series)
        self.assertTrue(result.empty)
    
    def test_calculate_dynamic_lookback(self):
        """Test calculating dynamic lookback periods."""
        # Create test signal with varying volatility
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
        
        # First half: low volatility, second half: high volatility
        low_vol = np.random.normal(0, 0.2, 50)
        high_vol = np.random.normal(0, 1.0, 50)
        signal = np.concatenate([low_vol, high_vol])
        signal_series = pd.Series(signal, index=dates)
        
        # Calculate dynamic lookback
        result = calculate_dynamic_lookback(
            signal_series,
            min_lookback=10,
            max_lookback=50
        )
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # Lookback periods should be within specified range
        self.assertTrue((result >= 10).all())
        self.assertTrue((result <= 50).all())
        
        # High volatility periods should have shorter lookback
        avg_lookback_first = result.iloc[20:40].mean()
        avg_lookback_last = result.iloc[60:80].mean()
        self.assertGreater(avg_lookback_first, avg_lookback_last)
        
        # Test with empty series
        empty_series = pd.Series([])
        result = calculate_dynamic_lookback(empty_series)
        self.assertTrue(result.empty)
    
    def test_ewma_crossover_signal(self):
        """Test generating EWMA crossover signals."""
        # Create test signal series
        signal_series = self.df['signal1']
        
        # Generate crossover signals
        result = ewma_crossover_signal(
            signal_series,
            fast_period=5,
            slow_period=20
        )
        
        # Check result
        self.assertEqual(len(result), len(signal_series))
        
        # Signal should be -1, 0, or 1
        self.assertTrue(set(np.unique(result)).issubset({-1, 0, 1}))
        
        # Most values should be 0 (no crossover)
        self.assertTrue((result == 0).sum() > len(result) * 0.9)
        
        # Should have some crossover signals
        self.assertTrue(((result == 1) | (result == -1)).any())
        
        # Test with empty series
        empty_series = pd.Series([])
        result = ewma_crossover_signal(empty_series)
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()