"""
Tests for the optimal transport module.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ohlcv_transport.models.empirical_measures import (
    EmpiricalMeasure,
    create_empirical_measures
)
from ohlcv_transport.models.transport import (
    create_cost_matrix,
    sinkhorn_algorithm,
    wasserstein_distance,
    compute_transport,
    compute_transport_batch,
    calculate_imbalance_signal,
    process_ohlcv_with_transport,
    visualize_transport_plan,
    visualize_imbalance_signal
)


class TestTransport(unittest.TestCase):
    """Tests for the optimal transport utilities."""
    
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
        
        # Create sample empirical measures
        # Supply measure (skewed higher)
        grid1 = np.linspace(99.0, 101.0, 50)
        weights1 = np.exp(-0.5 * ((grid1 - 100.5) / 0.3) ** 2)
        self.supply_measure = EmpiricalMeasure(grid1, weights1)
        
        # Demand measure (skewed lower)
        grid2 = np.linspace(99.0, 101.0, 50)
        weights2 = np.exp(-0.5 * ((grid2 - 99.5) / 0.3) ** 2)
        self.demand_measure = EmpiricalMeasure(grid2, weights2)
    
    def test_create_cost_matrix(self):
        """Test creating cost matrix."""
        # Basic cost matrix with L1 norm
        cost_matrix = create_cost_matrix(
            self.supply_measure.price_grid,
            self.demand_measure.price_grid,
            lambda_param=0.0  # Disable normalized component
        )
        
        # Check dimensions
        self.assertEqual(cost_matrix.shape, (50, 50))
        
        # Check some properties
        # Cost should be zero for same price
        for i in range(50):
            self.assertAlmostEqual(cost_matrix[i, i], abs(self.supply_measure.price_grid[i] - self.demand_measure.price_grid[i]))
        
        # Cost should be positive and symmetric
        self.assertTrue(np.all(cost_matrix >= 0))
        
        # Test with normalized component
        cost_matrix_norm = create_cost_matrix(
            self.supply_measure.price_grid,
            self.demand_measure.price_grid,
            lambda_param=1.0,
            avg_price=100.0,
            price_vol=0.5
        )
        
        # Normalized component should make cost matrix different
        self.assertFalse(np.allclose(cost_matrix, cost_matrix_norm))
    
    def test_sinkhorn_algorithm(self):
        """Test Sinkhorn algorithm."""
        # Create simple test case
        mu = np.array([0.5, 0.5])
        nu = np.array([0.5, 0.5])
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        # Run Sinkhorn algorithm
        plan, info = sinkhorn_algorithm(mu, nu, cost, reg_param=0.1)
        
        # Check basic properties
        self.assertEqual(plan.shape, (2, 2))
        self.assertTrue(info['converged'])
        self.assertGreater(info['iterations'], 0)
        
        # Check that marginals are preserved (approximately)
        mu_margin = np.sum(plan, axis=1)
        nu_margin = np.sum(plan, axis=0)
        
        np.testing.assert_almost_equal(mu_margin, mu, decimal=5)
        np.testing.assert_almost_equal(nu_margin, nu, decimal=5)
        
        # Test with real measures
        cost_matrix = create_cost_matrix(
            self.supply_measure.price_grid,
            self.demand_measure.price_grid
        )
        
        plan, info = sinkhorn_algorithm(
            self.supply_measure.weights,
            self.demand_measure.weights,
            cost_matrix,
            reg_param=0.01
        )
        
        # Check basic properties
        self.assertEqual(plan.shape, (50, 50))
        self.assertTrue(info['converged'] or info['iterations'] == 100)
        
        # Test log domain version
        plan_log, info_log = sinkhorn_algorithm(
            self.supply_measure.weights,
            self.demand_measure.weights,
            cost_matrix,
            reg_param=0.01,
            log_domain=True
        )
        
        # Should give similar results to standard version
        self.assertEqual(plan_log.shape, plan.shape)
        
        # Test with invalid inputs
        empty_plan, empty_info = sinkhorn_algorithm(
            np.zeros(10), 
            np.ones(10), 
            np.ones((10, 10))
        )
        
        self.assertFalse(empty_info['converged'])
        self.assertIsNotNone(empty_info['error'])
    
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        # Simple test case
        plan = np.array([[0.25, 0.25], [0.25, 0.25]])
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        # Expected distance is 0.5 (half the mass moves distance 1)
        distance = wasserstein_distance(plan, cost)
        self.assertEqual(distance, 0.5)
        
        # Test with zeros
        zero_plan = np.zeros((2, 2))
        zero_distance = wasserstein_distance(zero_plan, cost)
        self.assertEqual(zero_distance, 0.0)
    
    def test_compute_transport(self):
        """Test compute_transport function."""
        # Compute transport between sample measures
        result = compute_transport(
            self.supply_measure,
            self.demand_measure,
            lambda_param=0.5,
            reg_param=0.01,
            return_plan=True
        )
        
        # Check result structure
        self.assertIn('distance', result)
        self.assertIn('iterations', result)
        self.assertIn('converged', result)
        self.assertIn('time_ms', result)
        self.assertIn('transport_plan', result)
        self.assertIn('cost_matrix', result)
        
        # Distance should be positive
        self.assertGreater(result['distance'], 0.0)
        
        # Check transport plan shape
        self.assertEqual(result['transport_plan'].shape, (50, 50))
        
        # Test without returning plan
        result_no_plan = compute_transport(
            self.supply_measure,
            self.demand_measure,
            return_plan=False
        )
        
        # Should not include transport plan
        self.assertNotIn('transport_plan', result_no_plan)
        self.assertNotIn('cost_matrix', result_no_plan)
    
    def test_compute_transport_batch(self):
        """Test compute_transport_batch function."""
        # Create a batch of measures
        supply_measures = [self.supply_measure] * 5
        demand_measures = [self.demand_measure] * 5
        
        # Compute batch transport
        distances = compute_transport_batch(
            supply_measures,
            demand_measures
        )
        
        # Check result
        self.assertEqual(len(distances), 5)
        self.assertTrue(np.all(distances > 0))
        
        # All distances should be the same (same measures)
        self.assertTrue(np.allclose(distances, distances[0]))
        
        # Test with empty lists
        empty_distances = compute_transport_batch([], [])
        self.assertEqual(len(empty_distances), 0)
        
        # Test with mismatched lengths
        mismatch_distances = compute_transport_batch(
            supply_measures,
            demand_measures[:3]
        )
        self.assertEqual(len(mismatch_distances), 0)
    
    def test_calculate_imbalance_signal(self):
        """Test calculate_imbalance_signal function."""
        # Create sample distances
        distances = np.linspace(1.0, 3.0, 50)
        
        # Calculate signal
        signal = calculate_imbalance_signal(distances, lookback=20)
        
        # Check dimensions
        self.assertEqual(len(signal), len(distances))
        
        # First elements should be zero (not enough history)
        self.assertAlmostEqual(signal[0], 0.0)
        
        # Last element should be positive (increasing trend)
        self.assertGreater(signal[-1], 0.0)
        
        # Test with small input
        small_distances = np.array([1.0, 2.0])
        small_signal = calculate_imbalance_signal(small_distances, lookback=5)
        self.assertEqual(len(small_signal), 2)
        self.assertTrue(np.all(small_signal == 0.0))
        
        # Test with constant distances (std = 0)
        const_distances = np.ones(30)
        const_signal = calculate_imbalance_signal(const_distances, lookback=20)
        self.assertEqual(len(const_signal), 30)
        self.assertTrue(np.all(const_signal == 0.0))
    
    def test_process_ohlcv_with_transport(self):
        """Test process_ohlcv_with_transport function."""
        # Process sample DataFrame
        result_df = process_ohlcv_with_transport(
            self.df.iloc[:20],
            num_price_points=50,
            alpha=1.0,
            lambda_param=0.5,
            reg_param=0.01,
            lookback=10
        )
        
        # Check that new columns were added
        self.assertIn('w_distance', result_df.columns)
        self.assertIn('imbalance_signal', result_df.columns)
        self.assertIn('signal_change', result_df.columns)
        self.assertIn('signal_squared', result_df.columns)
        self.assertIn('signal_power', result_df.columns)
        
        # Check that distances were calculated
        self.assertTrue(np.all(result_df['w_distance'].iloc[:20] > 0))
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        empty_result = process_ohlcv_with_transport(empty_df)
        self.assertTrue(empty_result.empty)
    
    @patch('matplotlib.pyplot.show')
    def test_visualize_transport_plan(self, mock_show):
        """Test visualize_transport_plan function."""
        try:
            import matplotlib.pyplot as plt
            
            # Create a transport plan
            result = compute_transport(
                self.supply_measure,
                self.demand_measure,
                return_plan=True
            )
            
            # Visualize the transport plan
            fig = visualize_transport_plan(
                self.supply_measure,
                self.demand_measure,
                result['transport_plan'],
                show_plot=True
            )
            
            # Should return a valid figure and call plt.show()
            self.assertIsNotNone(fig)
            mock_show.assert_called_once()
            
            # Clean up
            plt.close(fig)
            
        except ImportError:
            # Skip test if matplotlib not available
            self.skipTest("Matplotlib not available")
    
    @patch('matplotlib.pyplot.show')
    def test_visualize_imbalance_signal(self, mock_show):
        """Test visualize_imbalance_signal function."""
        try:
            import matplotlib.pyplot as plt
            
            # Process sample DataFrame
            result_df = process_ohlcv_with_transport(
                self.df.iloc[:20],
                num_price_points=50
            )
            
            # Visualize the imbalance signal
            fig = visualize_imbalance_signal(
                result_df,
                show_plot=True
            )
            
            # Should return a valid figure and call plt.show()
            self.assertIsNotNone(fig)
            mock_show.assert_called_once()
            
            # Test with invalid data
            invalid_df = pd.DataFrame({'a': [1, 2, 3]})
            invalid_fig = visualize_imbalance_signal(invalid_df, show_plot=False)
            self.assertIsNone(invalid_fig)
            
            # Clean up
            if fig:
                plt.close(fig)
            
        except ImportError:
            # Skip test if matplotlib not available
            self.skipTest("Matplotlib not available")


if __name__ == '__main__':
    unittest.main()