"""
Signal processing utilities for the OHLCV Optimal Transport framework.

This module provides functions for processing and enhancing trading signals,
including denoising, filtering, and combining signals from multiple sources.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from scipy import stats, signal

# Setup logging
logger = logging.getLogger(__name__)


def normalize_signal(signal_series: pd.Series, 
                    lookback: int = 20, 
                    z_score: bool = True) -> pd.Series:
    """
    Normalize a signal using z-score or min-max scaling.
    
    Args:
        signal_series: Series containing the signal values
        lookback: Lookback window for normalization
        z_score: If True, use z-score normalization, otherwise min-max scaling
    
    Returns:
        Normalized signal series
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Make a copy of the input series
    result = signal_series.copy()
    
    if z_score:
        # Calculate rolling mean and standard deviation
        rolling_mean = signal_series.rolling(window=lookback, min_periods=1).mean()
        rolling_std = signal_series.rolling(window=lookback, min_periods=1).std()
        
        # Apply z-score normalization
        valid_idx = rolling_std > 0  # Avoid division by zero
        result[valid_idx] = ((signal_series[valid_idx] - rolling_mean[valid_idx]) / 
                             rolling_std[valid_idx])
    else:
        # Calculate rolling min and max
        rolling_min = signal_series.rolling(window=lookback, min_periods=1).min()
        rolling_max = signal_series.rolling(window=lookback, min_periods=1).max()
        
        # Apply min-max scaling
        valid_idx = (rolling_max - rolling_min) > 0  # Avoid division by zero
        result[valid_idx] = ((signal_series[valid_idx] - rolling_min[valid_idx]) / 
                             (rolling_max[valid_idx] - rolling_min[valid_idx]))
        
        # Scale to [-1, 1] range
        result = result * 2 - 1
    
    logger.debug(f"Normalized signal using {'z-score' if z_score else 'min-max'} with {lookback}-period lookback")
    return result


def combine_signals(signals: Dict[str, pd.Series], 
                   weights: Optional[Dict[str, float]] = None, 
                   method: str = 'weighted_sum') -> pd.Series:
    """
    Combine multiple signals into a single signal.
    
    Args:
        signals: Dictionary mapping signal names to signal series
        weights: Dictionary mapping signal names to weights (optional)
        method: Combination method ('weighted_sum', 'voting', 'max', 'min')
    
    Returns:
        Combined signal series
    """
    if not signals:
        logger.warning("Empty signals dictionary provided")
        return pd.Series()
    
    # Get a common index from all signals
    indices = [s.index for s in signals.values() if not s.empty]
    if not indices:
        return pd.Series()
    
    common_index = indices[0]
    for idx in indices[1:]:
        common_index = common_index.union(idx)
    
    # Reindex all signals to the common index
    aligned_signals = {name: s.reindex(common_index) for name, s in signals.items()}
    
    # Apply weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(signals) for name in signals}
    
    # Ensure all signals have weights
    for name in signals:
        if name not in weights:
            weights[name] = 1.0 / len(signals)
    
    # Initialize result
    result = pd.Series(0.0, index=common_index)
    
    if method == 'weighted_sum':
        # Calculate weighted sum
        for name, signal in aligned_signals.items():
            result += weights[name] * signal.fillna(0)
            
    elif method == 'voting':
        # Determine sign and apply weights as votes
        for name, signal in aligned_signals.items():
            result += weights[name] * np.sign(signal.fillna(0))
        
        # Normalize voting result to [-1, 1]
        max_vote = sum(abs(w) for w in weights.values())
        if max_vote > 0:
            result /= max_vote
            
    elif method == 'max':
        # Take the maximum signal value at each point
        df = pd.DataFrame(aligned_signals)
        result = df.max(axis=1)
        
    elif method == 'min':
        # Take the minimum signal value at each point
        df = pd.DataFrame(aligned_signals)
        result = df.min(axis=1)
        
    else:
        logger.warning(f"Unknown combination method: {method}")
        return pd.Series()
    
    logger.debug(f"Combined {len(signals)} signals using {method} method")
    return result


def detect_signal_regime(signal_series: pd.Series, 
                        lookback: int = 50,
                        threshold: float = 2.0) -> pd.Series:
    """
    Detect signal regime changes.
    
    Args:
        signal_series: Series containing the signal values
        lookback: Lookback window for baseline statistics
        threshold: Number of standard deviations for regime change detection
    
    Returns:
        Series indicating regime (0 = normal, 1 = high volatility)
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Calculate signal volatility
    signal_vol = signal_series.rolling(window=lookback, min_periods=5).std()
    
    # Calculate volatility of volatility (meta-volatility)
    meta_vol = signal_vol.rolling(window=lookback, min_periods=5).std()
    
    # Calculate normalized volatility
    norm_vol = (signal_vol - signal_vol.rolling(window=lookback, min_periods=5).mean()) / \
               signal_vol.rolling(window=lookback, min_periods=5).std()
    
    # Detect regime
    regime = pd.Series(0, index=signal_series.index)  # Default to normal regime
    regime[norm_vol > threshold] = 1  # High volatility regime
    
    logger.debug(f"Detected signal regimes using {lookback}-period lookback and {threshold} threshold")
    return regime


def adaptive_thresholds(signal_series: pd.Series,
                       regime_series: pd.Series,
                       base_threshold: float = 1.0,
                       vol_scaling: bool = True,
                       lookback: int = 50) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate adaptive thresholds for signal based on regime and volatility.
    
    Args:
        signal_series: Series containing the signal values
        regime_series: Series indicating regime (0 = normal, 1 = high volatility)
        base_threshold: Base threshold value
        vol_scaling: Whether to scale thresholds by volatility
        lookback: Lookback window for volatility calculation
    
    Returns:
        Tuple of (upper_threshold, lower_threshold) series
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series(), pd.Series()
    
    # Initialize threshold series
    upper_threshold = pd.Series(base_threshold, index=signal_series.index)
    lower_threshold = pd.Series(-base_threshold, index=signal_series.index)
    
    # Adjust thresholds based on regime
    if not regime_series.empty:
        # Increase thresholds in high volatility regime
        high_vol_idx = regime_series == 1
        upper_threshold[high_vol_idx] = base_threshold * 1.5
        lower_threshold[high_vol_idx] = -base_threshold * 1.5
    
    # Scale thresholds by volatility if requested
    if vol_scaling:
        # Calculate signal volatility
        signal_vol = signal_series.rolling(window=lookback, min_periods=5).std()
        
        # Calculate volatility ratio (current vol / average vol)
        avg_vol = signal_vol.rolling(window=lookback, min_periods=5).mean()
        vol_ratio = signal_vol / avg_vol
        vol_ratio = vol_ratio.fillna(1.0).clip(0.5, 2.0)  # Limit scaling range
        
        # Apply volatility scaling
        upper_threshold = upper_threshold * vol_ratio
        lower_threshold = lower_threshold * vol_ratio
    
    logger.debug(f"Calculated adaptive thresholds using regime information and volatility scaling")
    return upper_threshold, lower_threshold


def apply_signal_filter(signal_series: pd.Series,
                       filter_type: str = 'lowpass',
                       cutoff: float = 0.1,
                       order: int = 3) -> pd.Series:
    """
    Apply digital filter to signal series.
    
    Args:
        signal_series: Series containing the signal values
        filter_type: Filter type ('lowpass', 'highpass', 'bandpass')
        cutoff: Cutoff frequency or frequencies (normalized to [0, 1])
        order: Filter order
    
    Returns:
        Filtered signal series
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Handle NaN values
    values = signal_series.values
    nan_mask = np.isnan(values)
    if nan_mask.any():
        # Interpolate NaN values
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) == 0:
            return signal_series  # All NaN
            
        interp_indices = np.arange(len(values))
        values = np.interp(interp_indices, valid_indices, values[valid_indices])
    
    # Design filter
    if filter_type == 'lowpass':
        b, a = signal.butter(order, cutoff, btype='low')
    elif filter_type == 'highpass':
        b, a = signal.butter(order, cutoff, btype='high')
    elif filter_type == 'bandpass':
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            logger.warning("Bandpass filter requires cutoff to be a tuple of (low, high)")
            return signal_series
        b, a = signal.butter(order, cutoff, btype='band')
    else:
        logger.warning(f"Unknown filter type: {filter_type}")
        return signal_series
    
    # Apply filter
    filtered_values = signal.filtfilt(b, a, values)
    
    # Create result series
    result = pd.Series(filtered_values, index=signal_series.index)
    
    logger.debug(f"Applied {filter_type} filter to signal with cutoff {cutoff} and order {order}")
    return result


def fractional_differentiation(signal_series: pd.Series,
                              order: float = 0.5,
                              window: int = 10) -> pd.Series:
    """
    Apply fractional differentiation to a signal series.
    
    Args:
        signal_series: Series containing the signal values
        order: Fractional differentiation order (0 < order < 1)
        window: Window size for approximation
    
    Returns:
        Fractionally differentiated series
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Ensure order is in valid range
    if order <= 0 or order >= 1:
        logger.warning(f"Differentiation order should be between 0 and 1, got {order}")
        order = min(max(0.1, order), 0.9)
    
    # Calculate weights for the specified window
    weights = [1.0]
    for k in range(1, window):
        weights.append(weights[-1] * (k - 1 - order) / k)
    weights = np.array(weights)[::-1]
    
    # Create result series
    result = pd.Series(0.0, index=signal_series.index)
    
    # Apply fractional differentiation
    for i in range(window, len(signal_series)):
        # Get the windowed data
        window_data = signal_series.iloc[i-window:i].values
        
        # Apply weights
        result.iloc[i] = np.sum(weights * window_data)
    
    logger.debug(f"Applied fractional differentiation with order {order} and window {window}")
    return result


def detect_divergence(price_series: pd.Series,
                     signal_series: pd.Series,
                     window: int = 10) -> pd.Series:
    """
    Detect divergences between price and signal.
    
    Args:
        price_series: Series containing price values
        signal_series: Series containing signal values
        window: Window size for divergence detection
    
    Returns:
        Series indicating divergence type (0=none, 1=bullish, -1=bearish)
    """
    if price_series is None or signal_series is None or price_series.empty or signal_series.empty:
        logger.warning("Empty series provided")
        return pd.Series()
    
    # Ensure series have the same index
    common_index = price_series.index.intersection(signal_series.index)
    if len(common_index) == 0:
        logger.warning("No common index between price and signal series")
        return pd.Series()
    
    price = price_series.reindex(common_index)
    signal = signal_series.reindex(common_index)
    
    # Initialize result
    divergence = pd.Series(0, index=common_index)
    
    # Calculate rolling min/max for price and signal
    price_max = price.rolling(window=window).max()
    price_min = price.rolling(window=window).min()
    signal_max = signal.rolling(window=window).max()
    signal_min = signal.rolling(window=window).min()
    
    # Detect bullish divergence (price makes lower low, signal makes higher low)
    bullish_div = (price == price_min) & (signal > signal_min)
    
    # Detect bearish divergence (price makes higher high, signal makes lower high)
    bearish_div = (price == price_max) & (signal < signal_max)
    
    # Assign divergence values
    divergence[bullish_div] = 1   # Bullish divergence
    divergence[bearish_div] = -1  # Bearish divergence
    
    logger.debug(f"Detected divergences using {window}-period window")
    return divergence


def atr_bands(df: pd.DataFrame,
             price_col: str = 'close',
             high_col: str = 'high',
             low_col: str = 'low',
             period: int = 14,
             multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate ATR bands for volatility-adjusted signal thresholds.
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Column name for price
        high_col: Column name for high price
        low_col: Column name for low price
        period: Period for ATR calculation
        multiplier: Multiplier for band width
    
    Returns:
        Tuple of (upper_band, lower_band) series
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.Series(), pd.Series()
    
    # Check if required columns exist
    required_cols = [price_col, high_col, low_col]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Required column '{col}' not found in DataFrame")
            return pd.Series(), pd.Series()
    
    # Calculate True Range
    tr1 = df[high_col] - df[low_col]
    tr2 = abs(df[high_col] - df[price_col].shift(1))
    tr3 = abs(df[low_col] - df[price_col].shift(1))
    
    # Use the maximum of the three methods
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Average True Range
    atr = tr.rolling(period).mean()
    
    # Calculate upper and lower bands
    upper_band = df[price_col] + multiplier * atr
    lower_band = df[price_col] - multiplier * atr
    
    logger.debug(f"Calculated ATR bands using {period}-period ATR and {multiplier} multiplier")
    return upper_band, lower_band


def signal_to_position_sizing(signal_series: pd.Series,
                             base_position: float = 1.0,
                             max_position: float = 1.0,
                             threshold: float = 1.0) -> pd.Series:
    """
    Convert signal values to position sizes.
    
    Args:
        signal_series: Series containing the signal values
        base_position: Base position size (for signal = threshold)
        max_position: Maximum allowed position size
        threshold: Signal threshold for base position
    
    Returns:
        Series with position sizes
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Initialize result with zeros
    position_sizes = pd.Series(0.0, index=signal_series.index)
    
    # Calculate position sizes based on signal strength
    # For positive signals
    pos_mask = signal_series > 0
    if pos_mask.any():
        position_sizes[pos_mask] = np.minimum(
            base_position * signal_series[pos_mask] / threshold,
            max_position
        )
    
    # For negative signals
    neg_mask = signal_series < 0
    if neg_mask.any():
        position_sizes[neg_mask] = np.maximum(
            base_position * signal_series[neg_mask] / threshold,
            -max_position
        )
    
    logger.debug(f"Converted signal to position sizes using base={base_position}, max={max_position}, threshold={threshold}")
    return position_sizes


def calculate_dynamic_lookback(signal_series: pd.Series,
                              min_lookback: int = 10,
                              max_lookback: int = 100) -> pd.Series:
    """
    Calculate dynamic lookback period based on signal characteristics.
    
    Args:
        signal_series: Series containing the signal values
        min_lookback: Minimum lookback period
        max_lookback: Maximum lookback period
    
    Returns:
        Series with dynamic lookback periods
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Initialize result with default lookback
    lookback = pd.Series(min_lookback, index=signal_series.index)
    
    # Calculate signal volatility over different windows
    vol_short = signal_series.rolling(window=min_lookback).std()
    vol_long = signal_series.rolling(window=max_lookback).std()
    
    # Calculate volatility ratio
    vol_ratio = vol_short / vol_long
    vol_ratio = vol_ratio.fillna(1.0)
    
    # Scale lookback periods based on volatility ratio
    # Higher short-term volatility -> shorter lookback
    # Lower short-term volatility -> longer lookback
    scaled_lookback = min_lookback + (max_lookback - min_lookback) * (1.0 - vol_ratio)
    
    # Ensure lookback is within bounds
    lookback = np.clip(scaled_lookback, min_lookback, max_lookback)
    
    logger.debug(f"Calculated dynamic lookback periods between {min_lookback} and {max_lookback}")
    return lookback


def ewma_crossover_signal(signal_series: pd.Series,
                         fast_period: int = 5,
                         slow_period: int = 20) -> pd.Series:
    """
    Generate crossover signals from fast and slow EMAs of the signal.
    
    Args:
        signal_series: Series containing the signal values
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
    
    Returns:
        Series with crossover signals (1=buy, -1=sell, 0=neutral)
    """
    if signal_series is None or signal_series.empty:
        logger.warning("Empty signal series provided")
        return pd.Series()
    
    # Calculate fast and slow EMAs
    fast_ema = signal_series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = signal_series.ewm(span=slow_period, adjust=False).mean()
    
    # Initialize crossover signal
    crossover = pd.Series(0, index=signal_series.index)
    
    # Detect crossovers
    crossover[fast_ema > slow_ema] = 1  # Fast EMA above slow EMA: bullish
    crossover[fast_ema < slow_ema] = -1  # Fast EMA below slow EMA: bearish
    
    # Generate signals only on crossovers (changes in position)
    signal = crossover.diff().fillna(0)
    
    logger.debug(f"Generated EMA crossover signals using {fast_period}/{slow_period} periods")
    return signal