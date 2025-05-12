"""
Tests for the empirical measures module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ohlcv_transport.models.empirical_measures import (
    generate_price_grid,
    normal_density,
    calculate_supply_weights,
    calculate_demand_weights,
    EmpiricalMeasure,
    create_empirical_measures,
    calculate_measure_statistics,
    visualize_empirical_measures,
    visualize_measure_evolution
)


class TestEmpiricalMeasures(unittest.TestCase):
    """Tests for empirical measures utilities."""
    
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
        
        # Single bar data for detailed testing
        self.single_bar = {
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000.0
        }
        
        # Average price and volatility for the single bar
        self.avg_price = (self.single_bar['open'] + self.single_bar['high'] + 
                        self.single_bar['low'] + self.single_bar['close']) / 4
        self.price_vol = (self.single_bar['high'] - self.single_bar['low']) / 4
        
        # Create price grid for testing
        self.price_grid = np.linspace(98.0, 102.0, 100)
    
    def test_generate_price_grid(self):
        """Test generating a price grid."""
        # Test normal case
        grid = generate_price_grid(99.0, 101.0, num_points=50)
        
        # Check size and properties
        self.assertEqual(len(grid), 50)
        self.assertLess(grid[0], 99.0)  # Should extend below low
        self.assertGreater(grid[-1], 101.0)  # Should extend above high
        
        # Test handling of zero range
        grid_zero_range = generate_price_grid(100.0, 100.0, num_points=20)
        
        # Should still create a valid grid
        self.assertEqual(len(grid_zero_range), 20)
        self.assertLess(grid_zero_range[0], 100.0)
        self.assertGreater(grid_zero_range[-1], 100.0)
        
        # Test negative prices
        grid_negative = generate_price_grid(-101.0, -99.0, num_points=30)
        
        # Should handle negative prices correctly
        self.assertEqual(len(grid_negative), 30)
        self.assertLess(grid_negative[0], -101.0)
        self.assertGreater(grid_negative[-1], -99.0)
    
    def test_normal_density(self):
        """Test calculating normal density values."""
        # Test standard case
        x = np.array([0.0, 1.0, 2.0])
        density = normal_density(x, 1.0, 1.0)
        
        # Check result properties
        self.assertEqual(len(density), 3)
        self.assertGreater(density[1], density[0])  # Density should be highest at mean
        self.assertGreater(density[1], density[2])
        
        # Test zero standard deviation handling
        density_zero_std = normal_density(x, 1.0, 0.0)
        
        # Should still return valid densities
        self.assertEqual(len(density_zero_std), 3)
        self.assertTrue(np.all(np.isfinite(density_zero_std)))
        
        # Test large values
        x_large = np.array([1000.0, 2000.0])
        density_large = normal_density(x_large, 1500.0, 500.0)
        
        # Should return valid densities for large values
        self.assertEqual(len(density_large), 2)
        self.assertTrue(np.all(np.isfinite(density_large)))
    
    def test_calculate_supply_weights(self):
        """Test calculating supply weights."""
        # Test standard case
        weights = calculate_supply_weights(
            self.price_grid, 
            self.avg_price, 
            self.price_vol, 
            self.single_bar['low'], 
            self.single_bar['high'], 
            self.single_bar['volume']
        )
        
        # Check result properties
        self.assertEqual(len(weights), len(self.price_grid))
        self.assertAlmostEqual(np.sum(weights), self.single_bar['volume'], delta=1.0)
        
        # Pattern check: weights should be lower at the low end
        self.assertLess(weights[0], weights[len(weights)//2])
        
        # Test zero range handling
        weights_zero_range = calculate_supply_weights(
            self.price_grid, 
            100.0, 
            0.0, 
            100.0, 
            100.0, 
            1000.0
        )
        
        # Should still return valid weights
        self.assertEqual(len(weights_zero_range), len(self.price_grid))
        self.assertTrue(np.all(np.isfinite(weights_zero_range)))
        
        # Test alpha parameter effect
        weights_alpha2 = calculate_supply_weights(
            self.price_grid, 
            self.avg_price, 
            self.price_vol, 
            self.single_bar['low'], 
            self.single_bar['high'], 
            self.single_bar['volume'],
            alpha=2.0
        )
        
        # Higher alpha should make the skew more pronounced
        self.assertLess(weights_alpha2[0] / weights_alpha2[-1], 
                       weights[0] / weights[-1])
    
    def test_calculate_demand_weights(self):
        """Test calculating demand weights."""
        # Test standard case
        weights = calculate_demand_weights(
            self.price_grid, 
            self.avg_price, 
            self.price_vol, 
            self.single_bar['low'], 
            self.single_bar['high'], 
            self.single_bar['volume']
        )
        
        # Check result properties
        self.assertEqual(len(weights), len(self.price_grid))
        self.assertAlmostEqual(np.sum(weights), self.single_bar['volume'], delta=1.0)
        
        # Pattern check: weights should be lower at the high end
        self.assertLess(weights[-1], weights[len(weights)//2])
        
        # Test zero range handling
        weights_zero_range = calculate_demand_weights(
            self.price_grid, 
            100.0, 
            0.0, 
            100.0, 
            100.0, 
            1000.0
        )
        
        # Should still return valid weights
        self.assertEqual(len(weights_zero_range), len(self.price_grid))
        self.assertTrue(np.all(np.isfinite(weights_zero_range)))
        
        # Test alpha parameter effect
        weights_alpha2 = calculate_demand_weights(
            self.price_grid, 
            self.avg_price, 
            self.price_vol, 
            self.single_bar['low'], 
            self.single_bar['high'], 
            self.single_bar['volume'],
            alpha=2.0
        )
        
        # Higher alpha should make the skew more pronounced
        self.assertLess(weights_alpha2[-1] / weights_alpha2[0], 
                       weights[-1] / weights[0])
    
    def test_empirical_measure_class(self):
        """Test the EmpiricalMeasure class."""
        # Create a simple measure
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        measure = EmpiricalMeasure(grid, weights)
        
        # Test basic properties
        self.assertAlmostEqual(measure.mean, 3.0, delta=0.01)  # Expected mean for this distribution
        self.assertAlmostEqual(measure.variance, 1.2, delta=0.01)  # Expected variance
        self.assertAlmostEqual(measure.total_mass, 1.0, delta=0.01)
        
        # Test sparse representation with high threshold to include only the peak
        sparse_prices, sparse_weights = measure.to_sparse(threshold=0.5)
        
        # Should only include the significant weights (in this case just the middle one)
        self.assertEqual(len(sparse_prices), 1)
        self.assertEqual(sparse_prices[0], 3.0)
        self.assertEqual(sparse_weights[0], 0.4)
    
    def test_create_empirical_measures(self):
        """Test creating empirical measures from OHLCV data."""
        # Create measures from the sample DataFrame
        measures = create_empirical_measures(self.df.iloc[:10])
        
        # Check basic structure
        self.assertIn('supply', measures)
        self.assertIn('demand', measures)
        
        # Should have created measures for each valid bar
        self.assertEqual(len(measures['supply']), 10)
        self.assertEqual(len(measures['demand']), 10)
        
        # Check properties of first measure
        supply_measure = measures['supply'][0]
        demand_measure = measures['demand'][0]
        
        # Check that the measures have the expected properties
        self.assertEqual(len(supply_measure.price_grid), 100)  # Default grid size
        self.assertEqual(len(demand_measure.price_grid), 100)
        
        # Supply measure should have higher mean than demand
        self.assertGreater(supply_measure.mean, demand_measure.mean)
        
        # Total mass should match the bar volume
        bar_volume = self.df.iloc[0]['volume']
        self.assertAlmostEqual(supply_measure.total_mass, bar_volume, delta=1.0)
        self.assertAlmostEqual(demand_measure.total_mass, bar_volume, delta=1.0)
        
        # Test with invalid data
        empty_df = pd.DataFrame()
        empty_measures = create_empirical_measures(empty_df)
        
        # Should return empty lists with valid structure
        self.assertEqual(len(empty_measures['supply']), 0)
        self.assertEqual(len(empty_measures['demand']), 0)
        
        # Test with missing columns
        invalid_df = pd.DataFrame({'open': [100.0], 'high': [101.0]})  # Missing columns
        invalid_measures = create_empirical_measures(invalid_df)
        
        # Should return empty lists with valid structure
        self.assertEqual(len(invalid_measures['supply']), 0)
        self.assertEqual(len(invalid_measures['demand']), 0)
    
    def test_calculate_measure_statistics(self):
        """Test calculating statistics for a list of measures."""
        # Create a list of test measures
        measures = []
        for i in range(5):
            grid = np.linspace(99.0 + i, 101.0 + i, 50)
            weights = np.ones(50) * (10.0 + i)
            measures.append(EmpiricalMeasure(grid, weights))
        
        # Calculate statistics
        stats = calculate_measure_statistics(measures)
        
        # Check stats structure
        self.assertIn('count', stats)
        self.assertIn('mean_price', stats)
        self.assertIn('mean_variance', stats)
        self.assertIn('mean_mass', stats)
        
        # Check values
        self.assertEqual(stats['count'], 5)
        self.assertAlmostEqual(stats['mean_price'], 102.0, delta=0.1)  # Average mean across measures
        self.assertGreater(stats['mean_variance'], 0.0)  # Should be positive
        self.assertAlmostEqual(stats['mean_mass'], 600.0, delta=1.0)  # Average mass of 50*(10+11+12+13+14)/5 = 600
        
        # Test with empty list
        empty_stats = calculate_measure_statistics([])
        
        # Should return valid structure with None values
        self.assertEqual(empty_stats['count'], 0)
        self.assertIsNone(empty_stats['mean_price'])
        self.assertIsNone(empty_stats['mean_variance'])
        self.assertIsNone(empty_stats['mean_mass'])
    
    @patch('matplotlib.pyplot.show')
    def test_visualize_empirical_measures(self, mock_show):
        """Test visualizing empirical measures."""
        try:
            import matplotlib.pyplot as plt
            
            # Create sample measures
            grid = np.linspace(99.0, 101.0, 50)
            supply_weights = np.exp(-0.5 * ((grid - 100.5) / 0.3) ** 2)
            demand_weights = np.exp(-0.5 * ((grid - 99.5) / 0.3) ** 2)
            
            supply_measure = EmpiricalMeasure(grid, supply_weights)
            demand_measure = EmpiricalMeasure(grid, demand_weights)
            
            # Test visualization
            fig = visualize_empirical_measures(supply_measure, demand_measure, show_plot=True)
            
            # Should return a valid figure and call plt.show()
            self.assertIsNotNone(fig)
            mock_show.assert_called_once()
            
            # Clean up
            plt.close(fig)
            
        except ImportError:
            # Skip test if matplotlib not available
            self.skipTest("Matplotlib not available")
    
    @patch('matplotlib.pyplot.show')
    def test_visualize_measure_evolution(self, mock_show):
        """Test visualizing measure evolution."""
        try:
            import matplotlib.pyplot as plt
            
            # Create sample measures
            measures = []
            timestamps = pd.date_range(start='2024-04-01', periods=5, freq='1d')
            
            grid = np.linspace(99.0, 101.0, 50)
            for i in range(5):
                weights = np.exp(-0.5 * ((grid - (100.0 + i * 0.1)) / 0.3) ** 2)
                measures.append(EmpiricalMeasure(grid, weights))
            
            # Test heatmap visualization
            fig_heatmap = visualize_measure_evolution(
                measures, 
                timestamps, 
                mode='heatmap', 
                show_plot=True
            )
            
            # Should return a valid figure and call plt.show()
            self.assertIsNotNone(fig_heatmap)
            self.assertEqual(mock_show.call_count, 1)
            
            # Test surface visualization
            fig_surface = visualize_measure_evolution(
                measures, 
                timestamps, 
                mode='surface', 
                show_plot=True
            )
            
            # Should return a valid figure and call plt.show() again
            self.assertIsNotNone(fig_surface)
            self.assertEqual(mock_show.call_count, 2)
            
            # Test invalid mode
            fig_invalid = visualize_measure_evolution(
                measures, 
                timestamps, 
                mode='invalid', 
                show_plot=False
            )
            
            # Should return None for invalid mode
            self.assertIsNone(fig_invalid)
            
            # Test empty measures list
            fig_empty = visualize_measure_evolution(
                [], 
                [], 
                show_plot=False
            )
            
            # Should return None for empty list
            self.assertIsNone(fig_empty)
            
            # Clean up
            if fig_heatmap:
                plt.close(fig_heatmap)
            if fig_surface:
                plt.close(fig_surface)
            
        except ImportError:
            # Skip test if matplotlib not available
            self.skipTest("Matplotlib not available")


if __name__ == '__main__':
    unittest.main()