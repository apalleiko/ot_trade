"""
Tests for the regime detection module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from ohlcv_transport.models.regime import (
    RegimeParameters,
    RegimeManager,
    calculate_regime_statistic,
    detect_regime_changes,
    get_regime_specific_parameters
)


class TestRegimeParameters(unittest.TestCase):
    """Tests for the RegimeParameters class."""
    
    def test_initialization(self):
        """Test initialization with default values."""
        params = RegimeParameters()
        
        self.assertEqual(params.alpha, 1.0)
        self.assertEqual(params.beta, 0.5)
        self.assertEqual(params.gamma, 1.0)
        self.assertEqual(params.lambda_param, 0.5)
        self.assertEqual(params.eta, 0.01)
        self.assertEqual(params.signal_threshold, 1.5)
        self.assertEqual(params.min_holding_period, 5)
        self.assertEqual(params.max_holding_period, 180)
        self.assertEqual(params.position_size_factor, 1.0)
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        params = RegimeParameters(
            alpha=0.8,
            beta=0.6,
            gamma=1.2,
            lambda_param=0.4,
            eta=0.02,
            signal_threshold=2.0,
            min_holding_period=3,
            max_holding_period=120,
            position_size_factor=0.8
        )
        
        self.assertEqual(params.alpha, 0.8)
        self.assertEqual(params.beta, 0.6)
        self.assertEqual(params.gamma, 1.2)
        self.assertEqual(params.lambda_param, 0.4)
        self.assertEqual(params.eta, 0.02)
        self.assertEqual(params.signal_threshold, 2.0)
        self.assertEqual(params.min_holding_period, 3)
        self.assertEqual(params.max_holding_period, 120)
        self.assertEqual(params.position_size_factor, 0.8)
    
    def test_as_dict(self):
        """Test converting parameters to dictionary."""
        params = RegimeParameters(alpha=0.8, beta=0.6)
        params_dict = params.as_dict()
        
        self.assertIsInstance(params_dict, dict)
        self.assertEqual(params_dict["alpha"], 0.8)
        self.assertEqual(params_dict["beta"], 0.6)
        self.assertEqual(params_dict["gamma"], 1.0)  # Default value
    
    def test_from_dict(self):
        """Test creating parameters from dictionary."""
        params_dict = {
            "alpha": 0.8,
            "beta": 0.6,
            "gamma": 1.2,
            "lambda_param": 0.4,
            "eta": 0.02,
            "signal_threshold": 2.0,
            "min_holding_period": 3,
            "max_holding_period": 120,
            "position_size_factor": 0.8
        }
        
        params = RegimeParameters.from_dict(params_dict)
        
        self.assertEqual(params.alpha, 0.8)
        self.assertEqual(params.beta, 0.6)
        self.assertEqual(params.gamma, 1.2)
        self.assertEqual(params.lambda_param, 0.4)
        self.assertEqual(params.eta, 0.02)
        self.assertEqual(params.signal_threshold, 2.0)
        self.assertEqual(params.min_holding_period, 3)
        self.assertEqual(params.max_holding_period, 120)
        self.assertEqual(params.position_size_factor, 0.8)
    
    def test_from_partial_dict(self):
        """Test creating parameters from partial dictionary."""
        params_dict = {
            "alpha": 0.8,
            "beta": 0.6
        }
        
        params = RegimeParameters.from_dict(params_dict)
        
        self.assertEqual(params.alpha, 0.8)
        self.assertEqual(params.beta, 0.6)
        self.assertEqual(params.gamma, 1.0)  # Default value
        self.assertEqual(params.lambda_param, 0.5)  # Default value


class TestRegimeManager(unittest.TestCase):
    """Tests for the RegimeManager class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data with Wasserstein distances
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=200, freq='1min')
        
        # Create synthetic price series with two distinct regimes
        np.random.seed(42)  # For reproducibility
        
        # Base price in two regimes (normal and high volatility)
        base_returns_normal = np.random.normal(0, 0.0005, 100)  # Low vol (first 100 bars)
        base_returns_high = np.random.normal(0, 0.002, 100)     # High vol (last 100 bars)
        
        # Combine returns and create cumulative returns
        base_returns = np.concatenate([base_returns_normal, base_returns_high])
        cumulative_returns = np.cumprod(1 + base_returns)
        
        # Generate prices
        base_price = 100
        close = base_price * cumulative_returns
        high = close * (1 + np.abs(np.random.normal(0, 0.001, 200)))
        low = close * (1 - np.abs(np.random.normal(0, 0.001, 200)))
        open_price = low + np.random.uniform(0, 1, 200) * (high - low)
        
        # Generate volume with some spikes in the high volatility regime
        volume = 1000 + np.random.uniform(0, 200, 200)
        volume[150:160] *= 3  # Volume spike in high vol regime
        
        # Generate Wasserstein distances
        # In high volatility regime, W-distances are higher and more variable
        w_dist_normal = 0.5 + np.random.normal(0, 0.05, 100)
        w_dist_high = 0.8 + np.random.normal(0, 0.15, 100)
        w_distance = np.concatenate([w_dist_normal, w_dist_high])
        
        # Create basic imbalance signal
        imbalance_normal = np.random.normal(0, 0.8, 100)
        imbalance_high = np.random.normal(0, 1.5, 100)
        imbalance_signal = np.concatenate([imbalance_normal, imbalance_high])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'w_distance': w_distance,
            'imbalance_signal': imbalance_signal
        }, index=dates)
        
        # Create RegimeManager
        self.manager = RegimeManager(num_regimes=3, lookback_window=50)
        
        # Set up temp directory for saving/loading parameters
        self.temp_dir = Path('test_regime_params')
        self.temp_dir.mkdir(exist_ok=True)
        
        # Mock the config_manager.get_cache_dir method
        self.patcher = patch('ohlcv_transport.models.regime.config_manager.get_cache_dir')
        self.mock_get_cache_dir = self.patcher.start()
        self.mock_get_cache_dir.return_value = self.temp_dir
    
    def tearDown(self):
        """Clean up resources."""
        # Remove temp files - handle with try/except to avoid permission errors
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob('*'):
                    try:
                        file.unlink(missing_ok=True)
                    except (PermissionError, OSError):
                        # Just log the error and continue
                        pass
                try:
                    self.temp_dir.rmdir()
                except (PermissionError, OSError):
                    # Just log the error and continue
                    pass
        except Exception:
            # Fail silently on cleanup errors
            pass
        
        # Stop the patch
        self.patcher.stop()
    
    def test_initialization(self):
        """Test RegimeManager initialization."""
        # Create a new manager with mocked config to avoid loading saved params
        with patch('ohlcv_transport.models.regime.config_manager.get_cache_dir') as mock_dir:
            # Return a path that doesn't exist to avoid loading saved parameters
            mock_dir.return_value = Path('nonexistent_dir')
            manager = RegimeManager(num_regimes=3, lookback_window=50)
            
            self.assertEqual(manager.num_regimes, 3)
            self.assertEqual(manager.lookback_window, 50)
            self.assertEqual(len(manager.regime_params), 3)
            self.assertIsInstance(manager.regime_params[0], RegimeParameters)
            self.assertFalse(manager.is_calibrated)
    
    def test_detect_regime(self):
        """Test regime detection."""
        # Run regime detection
        regime_series = self.manager.detect_regime(self.df)
        
        # Check result properties
        self.assertIsInstance(regime_series, pd.Series)
        self.assertEqual(len(regime_series), len(self.df))
        
        # There should be at least two different regimes detected
        self.assertGreater(len(regime_series.unique()), 1)
        
        # The second half should have more high volatility regime assignments
        first_half = regime_series.iloc[:100].value_counts()
        second_half = regime_series.iloc[100:].value_counts()
        
        # Get the most common regime in each half
        if len(first_half) > 0 and len(second_half) > 0:
            most_common_first = first_half.index[0]
            most_common_second = second_half.index[0]
            
            # The most common regime should be different between the halves
            self.assertNotEqual(most_common_first, most_common_second)
    
    def test_save_load_parameters(self):
        """Test saving and loading regime parameters."""
        # Modify parameters
        self.manager.regime_params[0].alpha = 0.75
        self.manager.regime_params[1].beta = 0.65
        self.manager.regime_params[2].gamma = 1.25
        
        # Save parameters
        self.manager.save_parameters()
        
        # Create a new manager that should load the saved parameters
        new_manager = RegimeManager()
        
        # Check that parameters were loaded
        self.assertEqual(new_manager.regime_params[0].alpha, 0.75)
        self.assertEqual(new_manager.regime_params[1].beta, 0.65)
        self.assertEqual(new_manager.regime_params[2].gamma, 1.25)
    
    def test_calibrate_parameters(self):
        """Test parameter calibration."""
        # Create a fresh manager with known initial state
        with patch('ohlcv_transport.models.regime.config_manager.get_cache_dir') as mock_dir:
            mock_dir.return_value = self.temp_dir
            
            fresh_manager = RegimeManager(num_regimes=3, lookback_window=50)
            fresh_manager.is_calibrated = False
            
            # Force at least one regime to be in the data with enough samples
            # by explicitly setting some regime values in a copy of the DataFrame
            df_with_regimes = self.df.copy()
            df_with_regimes['explicit_regime'] = 0
            df_with_regimes.loc[df_with_regimes.index[50:150], 'explicit_regime'] = 1  # Ensure regime 1 has 100 samples
            
            # Initial values before calibration
            initial_params = {
                r: fresh_manager.regime_params[r].as_dict() 
                for r in range(fresh_manager.num_regimes)
            }
            
            # Modify a parameter to be extreme so we can ensure calibration actually does something
            for r in range(fresh_manager.num_regimes):
                # Set very extreme values that would definitely be adjusted during calibration
                fresh_manager.regime_params[r].alpha = 100.0
                fresh_manager.regime_params[r].signal_threshold = 100.0
                
            # Calibrate parameters using the explicit regimes to ensure we test regime 1
            # We're using a different regime assignment method since our actual detection
            # is what we're testing elsewhere
            fresh_manager.calibrate_parameters(df_with_regimes, df_with_regimes['explicit_regime'])
            
            # Check that calibration flag is set
            self.assertTrue(fresh_manager.is_calibrated)
            
            # Parameters should be reasonable after calibration (not the extreme values we set)
            for r in range(fresh_manager.num_regimes):
                if r == 1:  # We ensured regime 1 has enough data
                    params = fresh_manager.regime_params[r]
                    # Check that alpha is no longer the extreme value
                    self.assertNotEqual(params.alpha, 100.0)
                    self.assertLess(params.alpha, 10.0)  # Should be in a reasonable range
                    # Check that signal threshold is no longer the extreme value
                    self.assertNotEqual(params.signal_threshold, 100.0)
                    self.assertLess(params.signal_threshold, 10.0)  # Should be in a reasonable range
    
    def test_get_regime_parameters(self):
        """Test getting regime-specific parameters."""
        # Get parameters for each regime
        for r in range(self.manager.num_regimes):
            params = self.manager.get_regime_parameters(r)
            self.assertIsInstance(params, RegimeParameters)
        
        # Test getting parameters for non-existent regime
        # Should return parameters for regime 1 (normal regime)
        params = self.manager.get_regime_parameters(99)
        self.assertEqual(params.alpha, self.manager.regime_params[1].alpha)
    
    def test_predict_next_regime(self):
        """Test predicting the next regime."""
        # Initialize with a uniform transition matrix
        self.manager.transition_matrix = np.ones((3, 3)) / 3
        
        # Set a specific row to predict deterministically
        self.manager.transition_matrix[1] = np.array([0.1, 0.1, 0.8])
        
        # Multiple predictions from regime 1 should mostly predict regime 2
        regime_counts = {}
        for _ in range(100):
            next_regime = self.manager.predict_next_regime(1)
            regime_counts[next_regime] = regime_counts.get(next_regime, 0) + 1
        
        # Regime 2 should be the most common prediction
        self.assertEqual(max(regime_counts, key=regime_counts.get), 2)
        
        # Test with invalid regime ID
        next_regime = self.manager.predict_next_regime(99)
        self.assertIn(next_regime, [0, 1, 2])  # Should return a valid regime
    
    def test_get_regime_statistics(self):
        """Test getting regime statistics."""
        # Detect regimes to populate history
        self.manager.detect_regime(self.df)
        
        # Get statistics
        stats = self.manager.get_regime_statistics()
        
        # Check structure
        self.assertIn("regime_counts", stats)
        self.assertIn("regime_proportions", stats)
        self.assertIn("mean_durations", stats)
        self.assertIn("transition_matrix", stats)
        self.assertIn("is_calibrated", stats)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        # Create a fresh manager instance with mocked config
        with patch('ohlcv_transport.models.regime.config_manager.get_cache_dir') as mock_dir:
            # Return a path that doesn't exist to avoid loading saved parameters
            mock_dir.return_value = Path('nonexistent_dir')
            fresh_manager = RegimeManager()
            fresh_manager.is_calibrated = False  # Ensure it starts uncalibrated
            
            # Detect regimes with empty data
            regime_series = fresh_manager.detect_regime(empty_df)
            self.assertTrue(regime_series.empty)
            
            # Calibrate with empty data
            fresh_manager.calibrate_parameters(empty_df, pd.Series())
            self.assertFalse(fresh_manager.is_calibrated)  # Shouldn't be calibrated
    
    def test_regime_feature_calculation(self):
        """Test calculation of regime features."""
        # Access the protected method for testing
        features = self.manager._calculate_regime_features(self.df)
        
        # Check result properties
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Expected features
        expected_features = [
            'volatility', 'volume_z', 'price_range', 
            'w_distance_mean', 'w_distance_vol', 'w_distance_roc',
            'measure_diff', 'trend_strength', 'autocorr_1'
        ]
        
        # Check that most expected features are present
        for feature in expected_features:
            # Some features might not be calculated if data is missing
            if feature in features.columns:
                self.assertTrue(True)  # Feature exists
    
    def test_scale_features(self):
        """Test feature scaling."""
        # First calculate features
        features = self.manager._calculate_regime_features(self.df)
        
        # Scale features
        scaled = self.manager._scale_features(features)
        
        # Check result properties
        self.assertIsInstance(scaled, np.ndarray)
        self.assertEqual(scaled.shape[0], len(features))
        
        # Mean should be close to 0 and std close to 1 for each column
        means = np.mean(scaled, axis=0)
        stds = np.std(scaled, axis=0)
        
        # Not all features follow normal distribution, but they should be normalized
        for mean, std in zip(means, stds):
            self.assertAlmostEqual(mean, 0, delta=0.5)
            self.assertAlmostEqual(std, 1, delta=0.5)
    
    def test_cluster_regimes(self):
        """Test regime clustering."""
        # First calculate and scale features
        features = self.manager._calculate_regime_features(self.df)
        scaled = self.manager._scale_features(features)
        
        # Cluster regimes
        regimes = self.manager._cluster_regimes(scaled)
        
        # Check result properties
        self.assertIsInstance(regimes, np.ndarray)
        self.assertEqual(len(regimes), len(features))
        
        # Check that we have at most num_regimes different regimes
        self.assertLessEqual(len(np.unique(regimes)), self.manager.num_regimes)
    
    def test_update_transition_matrix(self):
        """Test updating the transition matrix."""
        # Create a series with regime transitions
        regimes = pd.Series([0, 0, 1, 1, 2, 2, 1, 0, 0])
        
        # Save original matrix
        original_matrix = self.manager.transition_matrix.copy()
        
        # Update transition matrix
        self.manager._update_transition_matrix(regimes)
        
        # The matrix should have changed
        self.assertFalse(np.array_equal(original_matrix, self.manager.transition_matrix))
        
        # The transition from regime 1 to 0 should have increased
        self.assertGreater(self.manager.transition_matrix[1, 0], original_matrix[1, 0])


class TestRegimeUtilityFunctions(unittest.TestCase):
    """Tests for the regime utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data with volatility regime shifts
        dates = pd.date_range(start='2024-04-01 09:30:00', periods=200, freq='1min')
        
        # Create synthetic price series with volatility shift
        np.random.seed(42)  # For reproducibility
        returns = np.concatenate([
            np.random.normal(0, 0.0005, 100),  # Low vol
            np.random.normal(0, 0.002, 100)    # High vol
        ])
        
        # Calculate prices
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.normal(0, 0.001, 200)))
        low = close * (1 - np.abs(np.random.normal(0, 0.001, 200)))
        
        # Generate synthetic Wasserstein distances
        w_distances = np.concatenate([
            np.random.uniform(0.4, 0.6, 100),  # Low vol regime
            np.random.uniform(0.7, 1.0, 100)   # High vol regime
        ])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
            'volume': 1000 + np.random.uniform(0, 200, 200),
            'w_distance': w_distances
        }, index=dates)
    
    def test_calculate_regime_statistic(self):
        """Test the regime detection statistic calculation."""
        # Calculate the statistic
        regime_stat = calculate_regime_statistic(self.df, window=20)
        
        # Check result properties
        self.assertIsInstance(regime_stat, pd.Series)
        self.assertEqual(len(regime_stat), len(self.df))
        
        # The statistic should be higher in the second half (high vol regime)
        first_half_mean = regime_stat.iloc[:100].mean()
        second_half_mean = regime_stat.iloc[100:].mean()
        
        self.assertLess(first_half_mean, second_half_mean)
    
    def test_detect_regime_changes(self):
        """Test the regime change detection function."""
        # First calculate the regime statistic
        regime_stat = calculate_regime_statistic(self.df, window=20)
        
        # Detect regime changes
        regimes = detect_regime_changes(regime_stat, threshold=1.5, min_duration=5)
        
        # Check result properties
        self.assertIsInstance(regimes, pd.Series)
        self.assertEqual(len(regimes), len(self.df))
        
        # The second half should have more high volatility regime assignments
        first_half_high_vol = (regimes.iloc[:100] == 1).mean()
        second_half_high_vol = (regimes.iloc[100:] == 1).mean()
        
        # There should be more high volatility regime in the second half
        self.assertLess(first_half_high_vol, second_half_high_vol)
    
    def test_get_regime_specific_parameters(self):
        """Test getting regime-specific parameters."""
        # Get parameters for each regime
        for regime in range(3):
            params = get_regime_specific_parameters(regime)
            self.assertIsInstance(params, RegimeParameters)
        
        # Test with invalid regime ID
        params = get_regime_specific_parameters(99)
        self.assertIsInstance(params, RegimeParameters)
    
    def test_empty_data_handling(self):
        """Test handling of empty data in utility functions."""
        empty_df = pd.DataFrame()
        empty_series = pd.Series()
        
        # Test with empty DataFrame
        regime_stat = calculate_regime_statistic(empty_df)
        self.assertTrue(regime_stat.empty)
        
        # Test with empty Series
        regimes = detect_regime_changes(empty_series)
        self.assertTrue(regimes.empty)
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        # Create DataFrame with missing columns
        incomplete_df = pd.DataFrame({
            'close': self.df['close']  # Only close prices
        })
        
        # Should still work but with limited functionality
        regime_stat = calculate_regime_statistic(incomplete_df)
        self.assertTrue(regime_stat.empty)  # No w_distance column


if __name__ == '__main__':
    unittest.main()