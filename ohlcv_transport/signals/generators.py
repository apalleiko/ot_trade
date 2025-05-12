"""
Signal generation utilities for the OHLCV Optimal Transport framework.

This module provides functions for generating trading signals from the
Wasserstein distances calculated by the optimal transport algorithm.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from scipy import stats, signal
import pywt

from ohlcv_transport.signals.processing import adaptive_thresholds, detect_signal_regime

# Setup logging
logger = logging.getLogger(__name__)


def calculate_base_imbalance_signal(df: pd.DataFrame,
                                   w_distance_col: str = 'w_distance',
                                   lookback: int = 20,
                                   output_col: str = 'imbalance_signal') -> pd.DataFrame:
    """
    Calculate the market imbalance signal based on Wasserstein distances.
    
    Formula: I_t = (W_1(μ_t, ν_t) - avg(W_1)_n) / std(W_1)_n
    
    Args:
        df: DataFrame with Wasserstein distances
        w_distance_col: Column name for Wasserstein distances
        lookback: Number of periods for normalization
        output_col: Output column name for the signal
    
    Returns:
        DataFrame with imbalance signal added
    """
    if df is None or df.empty or w_distance_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {w_distance_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Calculate rolling mean and standard deviation
    rolling_mean = df[w_distance_col].rolling(window=lookback, min_periods=1).mean()
    rolling_std = df[w_distance_col].rolling(window=lookback, min_periods=1).std()
    
    # Calculate normalized signal
    result[output_col] = np.zeros_like(df[w_distance_col])
    
    # Avoid division by zero
    valid_idx = rolling_std > 0
    result.loc[valid_idx, output_col] = (
        (df.loc[valid_idx, w_distance_col] - rolling_mean.loc[valid_idx]) / 
        rolling_std.loc[valid_idx]
    )
    
    logger.debug(f"Calculated imbalance signal using {lookback}-period lookback")
    return result


def calculate_multi_timeframe_signal(df_dict: Dict[str, pd.DataFrame],
                                    signal_col: str = 'imbalance_signal',
                                    output_col: str = 'multi_tf_signal') -> pd.DataFrame:
    """
    Calculate a combined signal from multiple timeframes.
    
    Formula: S_t = sum_j α_j I_t^(j)
    
    Args:
        df_dict: Dictionary mapping timeframes to DataFrames with signals
        signal_col: Column name for individual signals
        output_col: Output column name for combined signal
    
    Returns:
        DataFrame with combined signal at the lowest timeframe granularity
    """
    if not df_dict:
        logger.warning("Empty DataFrame dictionary provided")
        return pd.DataFrame()
    
    # Find the lowest timeframe (most granular) DataFrame
    timeframes = list(df_dict.keys())
    base_tf = min(timeframes, key=lambda x: len(df_dict[x]))
    result = df_dict[base_tf].copy()
    
    # Initialize multi-timeframe signal column
    result[output_col] = 0.0
    
    # Add each timeframe signal with optimal weighting
    weights = _calculate_optimal_weights(df_dict, signal_col)
    
    for tf, df in df_dict.items():
        if signal_col not in df.columns:
            logger.warning(f"{signal_col} not found in DataFrame for timeframe {tf}")
            continue
        
        # Resample higher timeframe signals to match the base timeframe
        if tf != base_tf:
            # Forward fill the signal to lower timeframes
            resampled_signal = df[signal_col].reindex(result.index, method='ffill')
        else:
            resampled_signal = df[signal_col]
        
        # Apply weight and add to multi-timeframe signal
        weight = weights.get(tf, 1.0 / len(df_dict))
        result[output_col] += weight * resampled_signal
    
    logger.info(f"Calculated multi-timeframe signal from {len(df_dict)} timeframes")
    return result


def _calculate_optimal_weights(df_dict: Dict[str, pd.DataFrame], 
                              signal_col: str = 'imbalance_signal',
                              lookback: int = 100) -> Dict[str, float]:
    """
    Calculate optimal weights for combining signals across timeframes.
    
    Formula: α_j = (β_j / σ_j²) / sqrt(sum_k (β_k²/σ_k²))
    
    Args:
        df_dict: Dictionary mapping timeframes to DataFrames with signals
        signal_col: Column name for signals
        lookback: Number of periods for calculating statistics
    
    Returns:
        Dictionary mapping timeframes to weights
    """
    # To compute beta (predictive power) and sigma (volatility) of each signal,
    # we calculate the correlation with future returns and the standard deviation
    
    betas = {}
    sigmas = {}
    
    for tf, df in df_dict.items():
        if signal_col not in df.columns:
            continue
        
        # Calculate signal standard deviation
        sigma = df[signal_col].tail(lookback).std()
        if sigma <= 0:
            sigma = 1.0  # Default to 1.0 to avoid division by zero
        
        # Calculate return correlation as proxy for beta
        if 'close' in df.columns:
            future_returns = df['close'].pct_change().shift(-1)
            beta = df[signal_col].tail(lookback).corr(future_returns.tail(lookback))
            if pd.isna(beta):
                beta = 0.1  # Default small positive value if correlation is NA
        else:
            beta = 0.1  # Default
        
        betas[tf] = abs(beta)  # Use absolute correlation
        sigmas[tf] = sigma
    
    # Calculate weights using the formula
    weights = {}
    total_factor = 0.0
    
    for tf in df_dict.keys():
        if tf in betas and tf in sigmas:
            factor = betas[tf] / (sigmas[tf] ** 2)
            weights[tf] = factor
            total_factor += factor ** 2
    
    # Normalize weights
    if total_factor > 0:
        normalizer = np.sqrt(total_factor)
        for tf in weights:
            weights[tf] = weights[tf] / normalizer
    else:
        # Equal weighting as fallback
        for tf in df_dict.keys():
            weights[tf] = 1.0 / len(df_dict)
    
    return weights


def add_signal_features(df: pd.DataFrame, 
                       signal_col: str = 'imbalance_signal') -> pd.DataFrame:
    """
    Add derived features from the base signal.
    
    Args:
        df: DataFrame with signal
        signal_col: Column name for the signal
    
    Returns:
        DataFrame with additional signal features
    """
    if df is None or df.empty or signal_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {signal_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Calculate signal derivative (change)
    result[f'{signal_col}_change'] = result[signal_col].diff()
    
    # Calculate signal acceleration (second derivative)
    result[f'{signal_col}_accel'] = result[f'{signal_col}_change'].diff()
    
    # Calculate signal squared (captures magnitude regardless of direction)
    result[f'{signal_col}_squared'] = result[signal_col] ** 2
    
    # Calculate signal power transformation (preserves direction but reduces outliers)
    result[f'{signal_col}_power'] = np.sign(result[signal_col]) * np.abs(result[signal_col]) ** 0.5
    
    # Calculate signal moving averages
    result[f'{signal_col}_sma5'] = result[signal_col].rolling(window=5).mean()
    result[f'{signal_col}_sma20'] = result[signal_col].rolling(window=20).mean()
    
    # Add crossover indicators
    result[f'{signal_col}_cross_zero'] = np.sign(result[signal_col]) != np.sign(result[signal_col].shift(1))
    result[f'{signal_col}_cross_sma20'] = (
        np.sign(result[signal_col] - result[f'{signal_col}_sma20']) != 
        np.sign(result[signal_col].shift(1) - result[f'{signal_col}_sma20'].shift(1))
    )
    
    logger.debug(f"Added derived features for signal {signal_col}")
    return result


def apply_kalman_filter(df: pd.DataFrame,
                       signal_col: str = 'imbalance_signal',
                       process_variance: float = 1e-3,
                       measurement_variance: float = 1e-2,
                       output_col: str = 'kalman_signal') -> pd.DataFrame:
    """
    Apply Kalman filtering to smooth the signal.
    
    Args:
        df: DataFrame with signal
        signal_col: Column name for the signal
        process_variance: Process noise variance (Q)
        measurement_variance: Measurement noise variance (R)
        output_col: Output column name for filtered signal
    
    Returns:
        DataFrame with filtered signal added
    """
    if df is None or df.empty or signal_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {signal_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Initialize Kalman filter
    x_hat = 0.0  # Initial state estimate
    p = 1.0  # Initial estimate uncertainty
    
    # Storage for filtered values
    filtered_values = np.zeros_like(df[signal_col])
    
    # Process each observation
    for i, measurement in enumerate(df[signal_col]):
        if pd.isna(measurement):
            filtered_values[i] = x_hat if i > 0 else 0.0
            continue
        
        # Prediction update
        # x_hat = x_hat (no dynamic model)
        p = p + process_variance
        
        # Measurement update
        k = p / (p + measurement_variance)  # Kalman gain
        x_hat = x_hat + k * (measurement - x_hat)
        p = (1 - k) * p
        
        # Store filtered value
        filtered_values[i] = x_hat
    
    # Add filtered signal to result
    result[output_col] = filtered_values
    
    logger.debug(f"Applied Kalman filter to {signal_col}")
    return result


def apply_wavelet_denoising(df: pd.DataFrame,
                           signal_col: str = 'imbalance_signal',
                           wavelet: str = 'db4',
                           level: int = 3,
                           output_col: str = 'wavelet_signal') -> pd.DataFrame:
    """
    Apply wavelet denoising to the signal.
    
    Args:
        df: DataFrame with signal
        signal_col: Column name for the signal
        wavelet: Wavelet type
        level: Decomposition level
        output_col: Output column name for denoised signal
    
    Returns:
        DataFrame with denoised signal added
    """
    if df is None or df.empty or signal_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {signal_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Get the signal
    signal_data = df[signal_col].values
    
    # Handle NaN values
    nan_mask = np.isnan(signal_data)
    if nan_mask.any():
        # Interpolate NaN values
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) == 0:
            # All values are NaN
            return result
        
        interp_indices = np.arange(len(signal_data))
        signal_data = np.interp(interp_indices, valid_indices, signal_data[valid_indices])
    
    # Wavelet decomposition
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # Threshold calculation (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    
    # Apply thresholding to detailed coefficients
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    # Reconstruct signal
    denoised_data = pywt.waverec(coeffs, wavelet)
    
    # Ensure the reconstructed signal has the same length
    denoised_data = denoised_data[:len(signal_data)]
    
    # Add denoised signal to result
    result[output_col] = denoised_data
    
    logger.debug(f"Applied wavelet denoising to {signal_col}")
    return result


def apply_exponential_smoothing(df: pd.DataFrame,
                              signal_col: str = 'imbalance_signal',
                              alpha: float = 0.2,
                              output_col: str = 'ema_signal') -> pd.DataFrame:
    """
    Apply exponential smoothing to the signal.
    
    Args:
        df: DataFrame with signal
        signal_col: Column name for the signal
        alpha: Smoothing factor (0 < alpha < 1)
        output_col: Output column name for smoothed signal
    
    Returns:
        DataFrame with smoothed signal added
    """
    if df is None or df.empty or signal_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {signal_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Apply exponential smoothing
    result[output_col] = df[signal_col].ewm(alpha=alpha, adjust=False).mean()
    
    logger.debug(f"Applied exponential smoothing to {signal_col} with alpha={alpha}")
    return result


def generate_trading_signal(df: pd.DataFrame,
                           signal_col: str = 'imbalance_signal',
                           threshold_long: float = 1.0,
                           threshold_short: float = -1.0,
                           position_col: str = 'position') -> pd.DataFrame:
    """
    Generate a trading position signal from the imbalance signal.
    
    Args:
        df: DataFrame with signal
        signal_col: Column name for the signal
        threshold_long: Threshold for long positions
        threshold_short: Threshold for short positions
        position_col: Output column name for position signal
    
    Returns:
        DataFrame with position signal added
    """
    if df is None or df.empty or signal_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {signal_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Initialize position column
    result[position_col] = 0
    
    # Apply thresholds
    result.loc[df[signal_col] >= threshold_long, position_col] = 1
    result.loc[df[signal_col] <= threshold_short, position_col] = -1
    
    logger.debug(f"Generated trading signals using thresholds: long={threshold_long}, short={threshold_short}")
    return result


def apply_minimum_holding_period(df: pd.DataFrame,
                                position_col: str = 'position',
                                min_periods: int = 5) -> pd.DataFrame:
    """
    Apply a minimum holding period to trading positions.
    
    Args:
        df: DataFrame with position signals
        position_col: Column name for positions
        min_periods: Minimum holding periods
    
    Returns:
        DataFrame with adjusted positions
    """
    if df is None or df.empty or position_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {position_col} column not found")
        return df
    
    # Make a copy of the input DataFrame
    result = df.copy()
    
    # Get the position signal
    positions = result[position_col].values
    
    # Initialize counter for current position
    counter = 0
    last_position = 0
    
    # Adjust positions
    for i in range(len(positions)):
        if positions[i] != last_position:
            # New position
            if counter >= min_periods or counter == 0:
                # If minimum holding period satisfied or initial position
                counter = 1
                last_position = positions[i]
            else:
                # If minimum holding period not satisfied, maintain old position
                positions[i] = last_position
                counter += 1
        else:
            # Same position, increment counter
            counter += 1
    
    # Update position column
    result[position_col] = positions
    
    logger.debug(f"Applied minimum holding period of {min_periods}")
    return result


def calculate_trading_metrics(df: pd.DataFrame,
                             position_col: str = 'position',
                             price_col: str = 'close') -> Dict[str, float]:
    """
    Calculate trading performance metrics.
    
    Args:
        df: DataFrame with positions and prices
        position_col: Column name for positions
        price_col: Column name for prices
    
    Returns:
        Dictionary of trading metrics
    """
    if df is None or df.empty or position_col not in df.columns or price_col not in df.columns:
        logger.warning(f"Invalid DataFrame or required columns not found")
        return {}
    
    # Calculate returns
    price_returns = df[price_col].pct_change()
    
    # Calculate strategy returns (using previous position to avoid lookahead bias)
    strategy_returns = df[position_col].shift(1) * price_returns
    
    # Count trades
    trades = (df[position_col].shift(1) != df[position_col]).sum()
    
    # Calculate metrics
    total_return = strategy_returns.sum()
    annual_factor = 252 / len(df) * len(df.index.unique())  # Scale based on data frequency
    annualized_return = total_return * annual_factor
    
    volatility = strategy_returns.std() * np.sqrt(annual_factor)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
    
    # Calculate drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max - 1)
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (strategy_returns > 0).mean()
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': trades
    }
    
    return metrics


def process_signal_pipeline(df: pd.DataFrame, 
                          w_distance_col: str = 'w_distance',
                          lookback: int = 20,
                          smooth_method: str = 'kalman',
                          alpha: float = 0.2,
                          threshold_long: float = 1.0,
                          threshold_short: float = -1.0,
                          min_holding: int = 5,
                          adaptive_threshold: bool = False) -> pd.DataFrame:
    """
    Apply a complete signal processing pipeline.
    
    Args:
        df: DataFrame with Wasserstein distances
        w_distance_col: Column name for Wasserstein distances
        lookback: Number of periods for normalization
        smooth_method: Smoothing method ('kalman', 'wavelet', 'ema' or None)
        alpha: Smoothing factor for EMA
        threshold_long: Threshold for long positions
        threshold_short: Threshold for short positions
        min_holding: Minimum holding periods
        adaptive_threshold: Whether to use adaptive thresholds
    
    Returns:
        DataFrame with processed signals
    """
    if df is None or df.empty or w_distance_col not in df.columns:
        logger.warning(f"Invalid DataFrame or {w_distance_col} column not found")
        return df
    
    # Calculate base signal
    result = calculate_base_imbalance_signal(df, w_distance_col, lookback)
    
    # Add derived features
    result = add_signal_features(result)
    
    # Apply smoothing method
    if smooth_method == 'kalman':
        result = apply_kalman_filter(result)
        signal_col = 'kalman_signal'
    elif smooth_method == 'wavelet':
        result = apply_wavelet_denoising(result)
        signal_col = 'wavelet_signal'
    elif smooth_method == 'ema':
        result = apply_exponential_smoothing(result, alpha=alpha)
        signal_col = 'ema_signal'
    else:
        signal_col = 'imbalance_signal'
    
    # Apply adaptive thresholds if requested
    if adaptive_threshold and len(result) > lookback:
        # Detect regime
        regime = detect_signal_regime(result[signal_col], lookback=lookback)
        result['signal_regime'] = regime
        
        # Calculate adaptive thresholds
        upper_thresh, lower_thresh = adaptive_thresholds(
            result[signal_col], 
            regime,
            base_threshold=threshold_long,
            vol_scaling=True,
            lookback=lookback
        )
        
        # Generate trading signal with adaptive thresholds
        result['threshold_upper'] = upper_thresh
        result['threshold_lower'] = lower_thresh
        
        # Initialize position column
        result['position'] = 0
        
        # Apply adaptive thresholds
        result.loc[result[signal_col] >= result['threshold_upper'], 'position'] = 1
        result.loc[result[signal_col] <= result['threshold_lower'], 'position'] = -1
    else:
        # Generate trading signal with fixed thresholds
        result = generate_trading_signal(
            result, 
            signal_col=signal_col,
            threshold_long=threshold_long,
            threshold_short=threshold_short
        )
    
    # Apply minimum holding period
    result = apply_minimum_holding_period(result, min_periods=min_holding)
    
    logger.info(f"Applied complete signal processing pipeline")
    return result