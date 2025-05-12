"""
Empirical measure construction for OHLCV data.

This module provides functions for transforming OHLCV data into supply and demand
empirical measures using the methodology described in the Optimal Transport Framework.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse

# Setup logging
logger = logging.getLogger(__name__)


def generate_price_grid(low: float, high: float, 
                       num_points: int = 100, 
                       extend_pct: float = 0.1) -> np.ndarray:
    """
    Generate a price grid spanning from below low to above high.
    
    Args:
        low: Low price
        high: High price
        num_points: Number of price points in the grid
        extend_pct: Percentage to extend the grid beyond low and high
    
    Returns:
        Array of equally spaced prices
    """
    # Calculate price range
    price_range = high - low
    
    # Handle potential zero range (flat prices)
    if price_range <= 0:
        # Use a small default range (1% of price)
        price_range = max(0.01, abs(high) * 0.01)
        # Center around the flat price
        low = (high + low) / 2 - price_range / 2
        high = (high + low) / 2 + price_range / 2
    
    # Extend range by specified percentage
    extension = price_range * extend_pct
    min_price = low - extension
    max_price = high + extension
    
    # Generate equally spaced price points
    price_grid = np.linspace(min_price, max_price, num_points)
    
    return price_grid


def normal_density(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate normal density function values.
    
    Args:
        x: Input values
        mu: Mean
        sigma: Standard deviation
    
    Returns:
        Normal density function values
    """
    # Handle potential division by zero
    if sigma <= 0:
        # Default to a small sigma to avoid division by zero
        sigma = max(0.001, abs(mu) * 0.01)
    
    # Calculate z-scores
    z = (x - mu) / sigma
    
    # Calculate normal density: φ(z) = (1 / √(2π * σ²)) * e^(-(z²)/2)
    density = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * sigma)
    
    return density


def calculate_supply_weights(price_grid: np.ndarray, 
                           avg_price: float, 
                           price_vol: float, 
                           low: float, 
                           high: float, 
                           volume: float, 
                           alpha: float = 1.0) -> np.ndarray:
    """
    Calculate supply weighting function based on the paper formula.
    
    Formula: f^μ(p) = φ((p - p̄)/σp) * ((p - L)/(H - L))^α
    
    Args:
        price_grid: Array of price points
        avg_price: Average price (p̄)
        price_vol: Price volatility (σp)
        low: Low price (L)
        high: High price (H)
        volume: Bar volume for scaling
        alpha: Power law exponent (default: 1.0)
    
    Returns:
        Array of normalized supply weights
    """
    # Handle potential division by zero
    price_range = high - low
    if price_range <= 0:
        price_range = max(0.001, abs(high) * 0.01)
    
    # Calculate normal density component
    normal_component = normal_density(price_grid, avg_price, price_vol)
    
    # Calculate power law component: ((p - L)/(H - L))^α
    # Clip to avoid negative values and ensure a small positive value to prevent division by zero
    power_component = np.maximum(1e-10, (price_grid - low) / price_range) ** alpha
    
    # Combined weighting function
    weights = normal_component * power_component
    
    # Normalize to sum to volume
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights * (volume / weights_sum)
    
    return weights


def calculate_demand_weights(price_grid: np.ndarray, 
                           avg_price: float, 
                           price_vol: float, 
                           low: float, 
                           high: float, 
                           volume: float, 
                           alpha: float = 1.0) -> np.ndarray:
    """
    Calculate demand weighting function based on the paper formula.
    
    Formula: f^ν(p) = φ((p - p̄)/σp) * ((H - p)/(H - L))^α
    
    Args:
        price_grid: Array of price points
        avg_price: Average price (p̄)
        price_vol: Price volatility (σp)
        low: Low price (L)
        high: High price (H)
        volume: Bar volume for scaling
        alpha: Power law exponent (default: 1.0)
    
    Returns:
        Array of normalized demand weights
    """
    # Handle potential division by zero
    price_range = high - low
    if price_range <= 0:
        price_range = max(0.001, abs(high) * 0.01)
    
    # Calculate normal density component
    normal_component = normal_density(price_grid, avg_price, price_vol)
    
    # Calculate power law component: ((H - p)/(H - L))^α
    # Clip to avoid negative values and ensure a small positive value to prevent division by zero
    power_component = np.maximum(1e-10, (high - price_grid) / price_range) ** alpha
    
    # Combined weighting function
    weights = normal_component * power_component
    
    # Normalize to sum to volume
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights * (volume / weights_sum)
    
    return weights


class EmpiricalMeasure:
    """
    Empirical measure representation for optimal transport.
    
    This class provides a compact representation of an empirical measure
    as a collection of price points and associated weights.
    """
    
    def __init__(self, price_grid: np.ndarray, weights: np.ndarray):
        """
        Initialize an empirical measure.
        
        Args:
            price_grid: Array of price points
            weights: Array of weights for each price point
        """
        self.price_grid = price_grid
        self.weights = weights
        
        # Calculate statistics
        self.mean = self._calculate_mean()
        self.variance = self._calculate_variance()
        self.total_mass = np.sum(weights)
    
    def _calculate_mean(self) -> float:
        """
        Calculate the mean of the measure.
        
        Returns:
            Mean price
        """
        if self.weights.sum() == 0:
            return 0.0
        
        # Calculate using float64 precision and convert to Python float to avoid test precision issues
        return float(np.sum(self.price_grid * self.weights) / np.sum(self.weights))
    
    def _calculate_variance(self) -> float:
        """
        Calculate the variance of the measure.
        
        Returns:
            Price variance
        """
        if self.weights.sum() == 0:
            return 0.0
        
        # Calculate using float64 precision and convert to Python float
        return float(np.sum(((self.price_grid - self.mean) ** 2) * self.weights) / np.sum(self.weights))
    
    def to_sparse(self, threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to sparse representation by keeping only significant weights.
        
        Args:
            threshold: Relative threshold for weight significance
                      (as a fraction of the maximum weight)
        
        Returns:
            Tuple of (price_points, weights) for significant weights
        """
        # Calculate absolute threshold based on max weight
        max_weight = np.max(self.weights)
        if max_weight <= 0:
            return np.array([]), np.array([])
            
        abs_threshold = max_weight * threshold
        
        # Find indices of significant weights
        significant_indices = np.where(self.weights > abs_threshold)[0]
        
        # Return prices and weights for those indices
        return (
            self.price_grid[significant_indices],
            self.weights[significant_indices]
        )
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        sparse_prices, sparse_weights = self.to_sparse()
        
        return {
            'price_points': sparse_prices.tolist(),
            'weights': sparse_weights.tolist(),
            'mean': self.mean,
            'variance': self.variance,
            'total_mass': self.total_mass
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmpiricalMeasure':
        """
        Create an empirical measure from dictionary representation.
        
        Args:
            data: Dictionary representation
        
        Returns:
            EmpiricalMeasure instance
        """
        price_points = np.array(data['price_points'])
        weights = np.array(data['weights'])
        
        # Create full grid and weights by interpolation
        num_points = 100  # Default or use a value from data
        
        if len(price_points) > 0:
            grid = np.linspace(min(price_points), max(price_points), num_points)
            
            # Simple nearest-neighbor interpolation for weights
            measure = cls(grid, np.zeros(num_points))
            
            # Fill in known weights
            for p, w in zip(price_points, weights):
                idx = np.abs(grid - p).argmin()
                measure.weights[idx] = w
                
            # Update statistics
            measure.mean = data.get('mean', measure._calculate_mean())
            measure.variance = data.get('variance', measure._calculate_variance())
            measure.total_mass = data.get('total_mass', np.sum(weights))
            
            return measure
        else:
            # Empty measure
            return cls(np.array([]), np.array([]))


def create_empirical_measures(df: pd.DataFrame, 
                             num_price_points: int = 100, 
                             alpha: float = 1.0) -> Dict[str, List[EmpiricalMeasure]]:
    """
    Create supply and demand empirical measures from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        num_price_points: Number of price points in each measure
        alpha: Power law exponent for weighting functions
    
    Returns:
        Dictionary containing lists of supply and demand measures
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to create_empirical_measures")
        return {'supply': [], 'demand': []}
    
    # Check if required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return {'supply': [], 'demand': []}
    
    # Initialize result containers
    supply_measures = []
    demand_measures = []
    
    # Process each bar
    for idx, row in df.iterrows():
        # Skip rows with missing data
        if any(pd.isna(row[col]) for col in required_columns):
            continue
        
        # Extract OHLCV values
        open_price = float(row['open'])
        high_price = float(row['high'])
        low_price = float(row['low'])
        close_price = float(row['close'])
        volume = float(row['volume'])
        
        # Calculate average price and price volatility as defined in the paper
        avg_price = (open_price + high_price + low_price + close_price) / 4
        price_vol = (high_price - low_price) / 4
        
        # Handle near-zero price volatility
        if price_vol <= 0:
            price_vol = max(0.001, abs(avg_price) * 0.01)
        
        # Create price grid
        price_grid = generate_price_grid(low_price, high_price, num_price_points)
        
        # Calculate supply and demand weights
        supply_weights = calculate_supply_weights(
            price_grid, avg_price, price_vol, low_price, high_price, volume, alpha
        )
        
        demand_weights = calculate_demand_weights(
            price_grid, avg_price, price_vol, low_price, high_price, volume, alpha
        )
        
        # Create empirical measures
        supply_measure = EmpiricalMeasure(price_grid, supply_weights)
        demand_measure = EmpiricalMeasure(price_grid, demand_weights)
        
        # Add to result containers
        supply_measures.append(supply_measure)
        demand_measures.append(demand_measure)
    
    logger.info(f"Created {len(supply_measures)} pairs of empirical measures")
    
    return {
        'supply': supply_measures,
        'demand': demand_measures
    }


def calculate_measure_statistics(measures: List[EmpiricalMeasure]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of empirical measures.
    
    Args:
        measures: List of EmpiricalMeasure instances
    
    Returns:
        Dictionary of statistics
    """
    if not measures:
        return {
            'count': 0,
            'mean_price': None,
            'mean_variance': None,
            'mean_mass': None
        }
    
    # Calculate statistics - using Python floats to avoid numpy data types in test comparisons
    means = [float(m.mean) for m in measures]
    variances = [float(m.variance) for m in measures]
    masses = [float(m.total_mass) for m in measures]
    
    # Explicitly dividing by length to ensure consistent behavior
    mean_price = sum(means) / len(means)
    mean_variance = sum(variances) / len(variances)
    mean_mass = sum(masses) / len(masses)
    
    return {
        'count': len(measures),
        'mean_price': mean_price,
        'mean_variance': mean_variance,
        'mean_mass': mean_mass
    }


def visualize_empirical_measures(supply_measure: EmpiricalMeasure, 
                               demand_measure: EmpiricalMeasure,
                               title: str = "Empirical Measures",
                               show_plot: bool = True) -> Any:
    """
    Visualize supply and demand empirical measures.
    
    Args:
        supply_measure: Supply empirical measure
        demand_measure: Demand empirical measure
        title: Plot title
        show_plot: Whether to display the plot
    
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot supply measure
        ax.fill_between(
            supply_measure.price_grid,
            0,
            supply_measure.weights,
            alpha=0.4,
            label=f"Supply (μ = {supply_measure.mean:.2f}, σ² = {supply_measure.variance:.4f})"
        )
        
        # Plot demand measure
        ax.fill_between(
            demand_measure.price_grid,
            0,
            demand_measure.weights,
            alpha=0.4,
            label=f"Demand (μ = {demand_measure.mean:.2f}, σ² = {demand_measure.variance:.4f})"
        )
        
        # Add chart elements
        ax.set_title(title)
        ax.set_xlabel("Price")
        ax.set_ylabel("Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return fig
    except ImportError:
        logger.warning("Matplotlib is required for visualization")
        return None


def visualize_measure_evolution(
    measures: List[EmpiricalMeasure],
    timestamps: List[Any],
    title: str = "Measure Evolution",
    mode: str = 'heatmap',
    show_plot: bool = True) -> Any:
    """
    Visualize the evolution of empirical measures over time.
    
    Args:
        measures: List of EmpiricalMeasure instances
        timestamps: List of timestamps or indices corresponding to measures
        title: Plot title
        mode: Visualization mode ('heatmap' or 'surface')
        show_plot: Whether to display the plot
    
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        if not measures:
            logger.warning("No measures to visualize")
            return None
        
        # Create price grid (use the first measure's grid)
        prices = measures[0].price_grid
        
        # Create weights matrix (rows = time, columns = prices)
        weights_matrix = np.zeros((len(measures), len(prices)))
        
        for i, measure in enumerate(measures):
            weights_matrix[i, :] = measure.weights
        
        # Create figure
        if mode == 'heatmap':
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            im = ax.imshow(
                weights_matrix,
                aspect='auto',
                origin='lower',
                extent=[prices[0], prices[-1], 0, len(measures)],
                cmap='viridis'
            )
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Weight')
            
            # Add chart elements
            ax.set_title(title)
            ax.set_xlabel("Price")
            ax.set_ylabel("Time")
            
        elif mode == 'surface':
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create meshgrid
            time_idx = np.arange(len(measures))
            time_grid, price_grid = np.meshgrid(time_idx, prices)
            
            # Create surface plot
            surf = ax.plot_surface(
                price_grid, 
                time_grid, 
                weights_matrix.T,
                cmap=cm.viridis,
                linewidth=0,
                antialiased=True
            )
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Weight')
            
            # Add chart elements
            ax.set_title(title)
            ax.set_xlabel("Price")
            ax.set_ylabel("Time")
            ax.set_zlabel("Weight")
            
        else:
            logger.warning(f"Unknown visualization mode: {mode}")
            return None
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return fig
        
    except ImportError:
        logger.warning("Matplotlib is required for visualization")
        return None