"""
Optimal transport calculations for OHLCV data.

This module implements the entropy-regularized optimal transport algorithm
(Sinkhorn algorithm) for computing Wasserstein distances between
empirical measures constructed from OHLCV data.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse

from ohlcv_transport.models.empirical_measures import EmpiricalMeasure

# Setup logging
logger = logging.getLogger(__name__)


def create_cost_matrix(source_grid: np.ndarray, 
                     target_grid: np.ndarray, 
                     lambda_param: float = 0.5,
                     avg_price: Optional[float] = None,
                     price_vol: Optional[float] = None) -> np.ndarray:
    """
    Create cost matrix for optimal transport calculation.
    
    Cost function: c(p_i, p_j) = |p_i - p_j| + λ · |z_i - z_j|
    where z_i = (p_i - p̄)/σ_p is the normalized price.
    
    Args:
        source_grid: Source price grid
        target_grid: Target price grid
        lambda_param: Weight for the normalized price component
        avg_price: Average price (optional, for price normalization)
        price_vol: Price volatility (optional, for price normalization)
    
    Returns:
        Cost matrix
    """
    # Compute meshgrid for pairwise distances
    source_grid_2d = source_grid.reshape(-1, 1)
    target_grid_2d = target_grid.reshape(1, -1)
    
    # Calculate absolute price differences
    price_diff = np.abs(source_grid_2d - target_grid_2d)
    
    # If normalization parameters provided, add normalized component
    if lambda_param > 0 and avg_price is not None and price_vol is not None and price_vol > 0:
        # Calculate normalized prices
        source_norm = (source_grid - avg_price) / price_vol
        target_norm = (target_grid - avg_price) / price_vol
        
        # Calculate normalized price differences
        source_norm_2d = source_norm.reshape(-1, 1)
        target_norm_2d = target_norm.reshape(1, -1)
        norm_diff = np.abs(source_norm_2d - target_norm_2d)
        
        # Combined cost function
        cost_matrix = price_diff + lambda_param * norm_diff
    else:
        # Simple L1 cost
        cost_matrix = price_diff
    
    return cost_matrix


def sinkhorn_algorithm(mu: np.ndarray, 
                     nu: np.ndarray, 
                     cost_matrix: np.ndarray,
                     reg_param: float = 0.01,
                     max_iter: int = 100,
                     threshold: float = 1e-6,
                     log_domain: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute optimal transport plan using the Sinkhorn algorithm.
    
    Args:
        mu: Source distribution weights
        nu: Target distribution weights
        cost_matrix: Cost matrix
        reg_param: Regularization parameter (entropy weight)
        max_iter: Maximum number of iterations
        threshold: Convergence threshold
        log_domain: Whether to use the log-domain version for numerical stability
    
    Returns:
        Tuple of (transport_plan, info_dict)
    """
    # Start timing
    start_time = time.time()
    
    # Normalize distributions to sum to 1
    mu_sum = np.sum(mu)
    nu_sum = np.sum(nu)
    
    if mu_sum <= 0 or nu_sum <= 0:
        logger.warning("Empty distributions provided to Sinkhorn algorithm")
        return np.zeros_like(cost_matrix), {'error': 'Empty distributions', 'converged': False}
    
    mu_norm = mu / mu_sum
    nu_norm = nu / nu_sum
    
    # Dimensions
    n = len(mu_norm)
    m = len(nu_norm)
    
    # Check dimensions
    if cost_matrix.shape != (n, m):
        logger.error(f"Cost matrix shape {cost_matrix.shape} doesn't match distributions: {n}x{m}")
        return np.zeros((n, m)), {'error': 'Dimension mismatch', 'converged': False}
    
    # Initialize info dict
    info = {
        'iterations': 0,
        'converged': False,
        'error': None,
        'time_ms': 0,
        'final_error': None
    }
    
    try:
        if log_domain:
            # Log-domain Sinkhorn (more stable)
            # Initialize dual variables
            f = np.zeros(n)
            g = np.zeros(m)
            
            # Kernel matrix exp(-cost/reg_param)
            kernel = np.exp(-cost_matrix / reg_param)
            
            # Sinkhorn iterations
            for it in range(max_iter):
                # Store old values
                f_old = f.copy()
                
                # Update f: f = log(mu) - log(exp(g) K^T)
                log_kg = np.log(np.maximum(1e-20, kernel @ np.exp(g)))
                f = np.log(mu_norm) - log_kg
                
                # Update g: g = log(nu) - log(exp(f) K)
                log_kf = np.log(np.maximum(1e-20, kernel.T @ np.exp(f)))
                g = np.log(nu_norm) - log_kf
                
                # Check convergence
                err = np.max(np.abs(f - f_old))
                if err < threshold:
                    info['converged'] = True
                    break
                
                info['iterations'] = it + 1
            
            # Compute transport plan
            fi = np.exp(f).reshape(-1, 1)
            gj = np.exp(g).reshape(1, -1)
            transport_plan = fi * kernel * gj
            
            # Scale back to original mass
            transport_plan *= mu_sum
            
            # Calculate final error
            marginal_mu = np.sum(transport_plan, axis=1)
            marginal_nu = np.sum(transport_plan, axis=0)
            final_error = max(
                np.max(np.abs(marginal_mu - mu)),
                np.max(np.abs(marginal_nu - nu))
            )
            info['final_error'] = float(final_error)
            
        else:
            # Standard Sinkhorn (less stable but simpler)
            # Initialize dual variables
            u = np.ones(n)
            v = np.ones(m)
            
            # Kernel matrix exp(-cost/reg_param)
            kernel = np.exp(-cost_matrix / reg_param)
            
            # Sinkhorn iterations
            for it in range(max_iter):
                # Store old values
                u_old = u.copy()
                
                # Update u
                u = mu_norm / (kernel @ v)
                
                # Update v
                v = nu_norm / (kernel.T @ u)
                
                # Check convergence
                err = np.max(np.abs(u - u_old))
                if err < threshold:
                    info['converged'] = True
                    break
                
                info['iterations'] = it + 1
            
            # Compute transport plan
            transport_plan = np.diag(u) @ kernel @ np.diag(v)
            
            # Scale back to original mass
            transport_plan *= mu_sum
            
            # Calculate final error
            marginal_mu = np.sum(transport_plan, axis=1)
            marginal_nu = np.sum(transport_plan, axis=0)
            final_error = max(
                np.max(np.abs(marginal_mu - mu)),
                np.max(np.abs(marginal_nu - nu))
            )
            info['final_error'] = float(final_error)
        
    except Exception as e:
        logger.error(f"Error in Sinkhorn algorithm: {e}")
        info['error'] = str(e)
        transport_plan = np.zeros((n, m))
    
    # Calculate execution time
    info['time_ms'] = (time.time() - start_time) * 1000
    
    return transport_plan, info


def wasserstein_distance(transport_plan: np.ndarray, 
                        cost_matrix: np.ndarray) -> float:
    """
    Calculate Wasserstein distance from transport plan and cost matrix.
    
    Args:
        transport_plan: Optimal transport plan
        cost_matrix: Cost matrix
    
    Returns:
        Wasserstein distance (float)
    """
    return float(np.sum(transport_plan * cost_matrix))


def compute_transport(source_measure: EmpiricalMeasure, 
                    target_measure: EmpiricalMeasure,
                    lambda_param: float = 0.5,
                    reg_param: float = 0.01,
                    max_iter: int = 100,
                    threshold: float = 1e-6,
                    log_domain: bool = True,
                    avg_price: Optional[float] = None,
                    price_vol: Optional[float] = None,
                    return_plan: bool = False) -> Dict[str, Any]:
    """
    Compute optimal transport between two empirical measures.
    
    Args:
        source_measure: Source empirical measure
        target_measure: Target empirical measure
        lambda_param: Weight for normalized price component in cost
        reg_param: Regularization parameter for Sinkhorn algorithm
        max_iter: Maximum number of iterations
        threshold: Convergence threshold
        log_domain: Whether to use log-domain Sinkhorn (more stable)
        avg_price: Average price for normalization (optional)
        price_vol: Price volatility for normalization (optional)
        return_plan: Whether to include transport plan in result
    
    Returns:
        Dictionary with results
    """
    # Create cost matrix
    cost_matrix = create_cost_matrix(
        source_measure.price_grid,
        target_measure.price_grid,
        lambda_param=lambda_param,
        avg_price=avg_price if avg_price is not None else source_measure.mean,
        price_vol=price_vol if price_vol is not None else np.sqrt(source_measure.variance)
    )
    
    # Run Sinkhorn algorithm
    transport_plan, info = sinkhorn_algorithm(
        source_measure.weights,
        target_measure.weights,
        cost_matrix,
        reg_param=reg_param,
        max_iter=max_iter,
        threshold=threshold,
        log_domain=log_domain
    )
    
    # Calculate Wasserstein distance
    distance = wasserstein_distance(transport_plan, cost_matrix)
    
    # Prepare result
    result = {
        'distance': distance,
        'iterations': info['iterations'],
        'converged': info['converged'],
        'time_ms': info['time_ms'],
        'error': info['error'],
        'final_error': info['final_error']
    }
    
    # Include transport plan if requested
    if return_plan:
        result['transport_plan'] = transport_plan
        result['cost_matrix'] = cost_matrix
    
    return result


def compute_transport_batch(supply_measures: List[EmpiricalMeasure],
                          demand_measures: List[EmpiricalMeasure],
                          lambda_param: float = 0.5,
                          reg_param: float = 0.01,
                          max_iter: int = 100,
                          threshold: float = 1e-6,
                          log_domain: bool = True) -> np.ndarray:
    """
    Compute optimal transport for a batch of measure pairs.
    
    Args:
        supply_measures: List of supply empirical measures
        demand_measures: List of demand empirical measures
        lambda_param: Weight for normalized price component in cost
        reg_param: Regularization parameter for Sinkhorn algorithm
        max_iter: Maximum number of iterations
        threshold: Convergence threshold
        log_domain: Whether to use log-domain Sinkhorn
    
    Returns:
        Array of Wasserstein distances
    """
    # Check input lengths
    if len(supply_measures) != len(demand_measures):
        logger.error("Supply and demand measure lists must have the same length")
        return np.array([])
    
    # Initialize result array
    distances = np.zeros(len(supply_measures))
    
    # Process each pair
    for i, (supply, demand) in enumerate(zip(supply_measures, demand_measures)):
        # Compute transport
        result = compute_transport(
            supply,
            demand,
            lambda_param=lambda_param,
            reg_param=reg_param,
            max_iter=max_iter,
            threshold=threshold,
            log_domain=log_domain
        )
        
        # Store distance
        distances[i] = result['distance']
    
    return distances


def calculate_imbalance_signal(distances: np.ndarray, 
                              lookback: int = 20) -> np.ndarray:
    """
    Calculate market imbalance signal from Wasserstein distances.
    
    Formula: I_t = (W_1(μ_t, ν_t) - avg(W_1)_n) / std(W_1)_n
    
    Args:
        distances: Array of Wasserstein distances
        lookback: Number of periods for normalization
    
    Returns:
        Array of normalized imbalance signals
    """
    if len(distances) < lookback:
        logger.warning(f"Not enough data for lookback {lookback}, returning zeros")
        return np.zeros_like(distances)
    
    # Initialize result array
    signals = np.zeros_like(distances)
    
    # Calculate rolling mean and std for normalization
    for i in range(len(distances)):
        if i < lookback:
            # Not enough history, use available data
            window = distances[:i+1]
        else:
            # Use lookback window
            window = distances[i-lookback+1:i+1]
        
        # Calculate mean and std
        mean = np.mean(window)
        std = np.std(window)
        
        # Avoid division by zero
        if std > 0:
            signals[i] = (distances[i] - mean) / std
        else:
            signals[i] = 0.0
    
    return signals


def process_ohlcv_with_transport(df: pd.DataFrame,
                                num_price_points: int = 100,
                                alpha: float = 1.0,
                                lambda_param: float = 0.5,
                                reg_param: float = 0.01,
                                lookback: int = 20) -> pd.DataFrame:
    """
    Process OHLCV data with optimal transport calculation.
    
    This function takes OHLCV data, creates empirical measures,
    computes Wasserstein distances, and generates the imbalance signal.
    
    Args:
        df: DataFrame with OHLCV data
        num_price_points: Number of price points in each grid
        alpha: Power law exponent for weighting functions
        lambda_param: Weight for normalized price component in cost
        reg_param: Regularization parameter for Sinkhorn algorithm
        lookback: Number of periods for signal normalization
    
    Returns:
        DataFrame with original data and additional columns
    """
    # Import here to avoid circular imports
    from ohlcv_transport.models.empirical_measures import create_empirical_measures
    
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Create empirical measures
    logger.info("Creating empirical measures...")
    measures = create_empirical_measures(df, num_price_points=num_price_points, alpha=alpha)
    
    if len(measures['supply']) == 0:
        logger.warning("No valid measures created")
        return result_df
    
    # Compute Wasserstein distances
    logger.info("Computing Wasserstein distances...")
    distances = compute_transport_batch(
        measures['supply'],
        measures['demand'],
        lambda_param=lambda_param,
        reg_param=reg_param
    )
    
    # Calculate imbalance signal
    logger.info("Calculating imbalance signal...")
    signals = calculate_imbalance_signal(distances, lookback=lookback)
    
    # Add to DataFrame
    valid_idx = df.index[:len(distances)]
    result_df.loc[valid_idx, 'w_distance'] = distances
    result_df.loc[valid_idx, 'imbalance_signal'] = signals
    
    # Add signal derivatives
    result_df['signal_change'] = result_df['imbalance_signal'].diff()
    result_df['signal_squared'] = result_df['imbalance_signal'] ** 2
    result_df['signal_power'] = np.sign(result_df['imbalance_signal']) * np.abs(result_df['imbalance_signal']) ** 0.5
    
    logger.info("Optimal transport processing complete")
    return result_df


def visualize_transport_plan(source_measure: EmpiricalMeasure,
                           target_measure: EmpiricalMeasure,
                           transport_plan: np.ndarray,
                           title: str = "Optimal Transport Plan",
                           num_flows: int = 50,
                           show_plot: bool = True) -> Any:
    """
    Visualize the optimal transport plan between two measures.
    
    Args:
        source_measure: Source empirical measure
        target_measure: Target empirical measure
        transport_plan: Optimal transport plan matrix
        title: Plot title
        num_flows: Number of transport flows to visualize
        show_plot: Whether to display the plot
    
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot measures
        ax1.fill_between(
            source_measure.price_grid,
            0,
            source_measure.weights,
            alpha=0.4,
            label=f"Supply (μ = {source_measure.mean:.2f})"
        )
        
        ax1.fill_between(
            target_measure.price_grid,
            0,
            target_measure.weights,
            alpha=0.4,
            label=f"Demand (μ = {target_measure.mean:.2f})"
        )
        
        ax1.set_title("Empirical Measures")
        ax1.set_ylabel("Weight")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot transport plan
        # Find significant flows
        flat_idx = np.argsort(transport_plan.flatten())[-num_flows:]
        i_idx, j_idx = np.unravel_index(flat_idx, transport_plan.shape)
        
        # Plot heat map
        im = ax2.imshow(
            transport_plan,
            origin='lower',
            aspect='auto',
            extent=[
                target_measure.price_grid[0],
                target_measure.price_grid[-1],
                source_measure.price_grid[0],
                source_measure.price_grid[-1]
            ],
            cmap='viridis'
        )
        
        # Plot significant flows
        for i, j in zip(i_idx, j_idx):
            ax2.plot(
                [target_measure.price_grid[j], source_measure.price_grid[i]],
                [source_measure.price_grid[i], source_measure.price_grid[i]],
                'r-', alpha=0.2, linewidth=1
            )
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Mass Transport')
        
        # Add chart elements
        ax2.set_xlabel("Demand Price")
        ax2.set_ylabel("Supply Price")
        ax2.set_title("Transport Plan")
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(title, fontsize=16)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return fig
        
    except ImportError:
        logger.warning("Matplotlib is required for visualization")
        return None


def visualize_imbalance_signal(df: pd.DataFrame,
                             title: str = "Market Imbalance Signal",
                             show_plot: bool = True) -> Any:
    """
    Visualize the market imbalance signal with price.
    
    Args:
        df: DataFrame with OHLCV data and imbalance signal
        title: Plot title
        show_plot: Whether to display the plot
    
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        
        # Check if required columns exist
        if 'close' not in df.columns or 'imbalance_signal' not in df.columns:
            logger.warning("Required columns not found in DataFrame")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot price
        ax1.plot(df.index, df['close'], 'b-', label='Close Price')
        ax1.set_ylabel('Price')
        ax1.set_title('Price')
        ax1.grid(True, alpha=0.3)
        
        # Plot imbalance signal
        ax2.plot(df.index, df['imbalance_signal'], 'g-', label='Imbalance Signal')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.2)
        ax2.axhline(y=-1, color='r', linestyle='--', alpha=0.2)
        ax2.set_ylabel('Signal')
        ax2.set_title('Market Imbalance Signal')
        ax2.grid(True, alpha=0.3)
        
        # Highlight signal regions
        positive_mask = df['imbalance_signal'] > 1
        negative_mask = df['imbalance_signal'] < -1
        
        if positive_mask.any():
            ax2.fill_between(
                df.index, 0, df['imbalance_signal'],
                where=positive_mask,
                color='green', alpha=0.3
            )
        
        if negative_mask.any():
            ax2.fill_between(
                df.index, 0, df['imbalance_signal'],
                where=negative_mask,
                color='red', alpha=0.3
            )
        
        # Overall title
        fig.suptitle(title, fontsize=16)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return fig
        
    except ImportError:
        logger.warning("Matplotlib is required for visualization")
        return None