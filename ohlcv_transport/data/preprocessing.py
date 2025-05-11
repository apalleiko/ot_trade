"""
Data preprocessing utilities for OHLCV data.

This module provides functions for cleaning, transforming, and enhancing OHLCV data,
including handling missing values, detecting outliers, applying wavelet denoising,
and calculating derived features.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import pywt
from scipy import stats
from scipy.interpolate import interp1d

# Setup logging
logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame, 
                forward_fill_limit: int = 5,
                interpolate_limit: int = 20,
                remove_zero_volume: bool = True) -> pd.DataFrame:
    """
    Clean OHLCV data by handling missing values and invalid data.
    
    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        forward_fill_limit: Maximum number of consecutive NaN values to fill using forward fill
        interpolate_limit: Maximum number of consecutive NaN values to fill using interpolation
        remove_zero_volume: Whether to replace zero volume with NaN
    
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to clean_ohlcv")
        return df
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure the dataframe has a datetime index
    if not isinstance(result.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex. Attempting to convert.")
        if 'datetime' in result.columns:
            result['datetime'] = pd.to_datetime(result['datetime'])
            result = result.set_index('datetime').sort_index()
        else:
            logger.error("Cannot convert index to datetime: 'datetime' column not found")
            return result
    
    # Ensure all required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in result.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            return result
    
    # Ensure numeric types
    for col in required_columns:
        result[col] = pd.to_numeric(result[col], errors='coerce')
    
    # Replace zero volume with NaN if specified
    if remove_zero_volume:
        result.loc[result['volume'] == 0, 'volume'] = np.nan
    
    # Check for invalid OHLC values (high < low, etc.)
    invalid_ohlc = (
        (result['high'] < result['low']) | 
        (result['open'] > result['high']) | 
        (result['open'] < result['low']) | 
        (result['close'] > result['high']) | 
        (result['close'] < result['low'])
    )
    
    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        logger.warning(f"Found {invalid_count} bars with invalid OHLC values")
        
        # Set invalid bars to NaN
        result.loc[invalid_ohlc, ['open', 'high', 'low', 'close']] = np.nan
    
    # Handle missing data
    # 1. First try forward fill for short gaps
    result = result.fillna(method='ffill', limit=forward_fill_limit)
    
    # 2. Then use interpolation for medium gaps
    result = result.interpolate(method='time', limit=interpolate_limit, limit_direction='both')
    
    # Report on remaining missing values
    missing_counts = result.isna().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Remaining missing values after cleaning: {missing_counts.to_dict()}")
    
    return result


def detect_outliers(df: pd.DataFrame, 
                   columns: List[str] = None,
                   window: int = 20, 
                   mad_threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in OHLCV data using Median Absolute Deviation (MAD).
    
    Args:
        df: DataFrame with OHLCV data
        columns: Columns to check for outliers (default: close, volume)
        window: Rolling window size for calculating median and MAD
        mad_threshold: Threshold for outlier detection (number of MADs)
    
    Returns:
        DataFrame with boolean mask indicating outliers
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to detect_outliers")
        return pd.DataFrame()
    
    if columns is None:
        columns = ['close', 'volume']
    
    # Create empty DataFrame for outlier mask with same index as input df
    outliers = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Calculate rolling median
        rolling_median = df[col].rolling(window=window, center=True, min_periods=window//2).median()
        
        # Calculate absolute deviations from the median
        deviations = np.abs(df[col] - rolling_median)
        
        # Calculate the MAD (Median Absolute Deviation)
        mad = deviations.rolling(window=window, center=True, min_periods=window//2).median()
        
        # Scale MAD (for normal distribution)
        mad_scaled = mad * 1.4826
        
        # Flag values that exceed the threshold
        outliers[col] = deviations > (mad_threshold * mad_scaled)
        
        # Log number of outliers detected
        num_outliers = outliers[col].sum()
        if num_outliers > 0:
            logger.info(f"Detected {num_outliers} outliers in column '{col}' ({num_outliers/len(df)*100:.2f}%)")
    
    return outliers


def replace_outliers(df: pd.DataFrame, 
                    outliers: pd.DataFrame,
                    method: str = 'interpolate') -> pd.DataFrame:
    """
    Replace outliers in OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        outliers: Boolean mask DataFrame indicating outliers
        method: Method to use for replacement ('interpolate', 'median', 'ffill')
    
    Returns:
        DataFrame with outliers replaced
    """
    if df is None or df.empty or outliers is None or outliers.empty:
        logger.warning("Empty DataFrame provided to replace_outliers")
        return df
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    for col in outliers.columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get indices of outliers
        outlier_indices = outliers.index[outliers[col]]
        
        if len(outlier_indices) == 0:
            continue
        
        logger.debug(f"Replacing {len(outlier_indices)} outliers in column '{col}'")
        
        # Replace outliers based on specified method
        if method == 'interpolate':
            # Set outliers to NaN
            result.loc[outlier_indices, col] = np.nan
            # Interpolate those NaN values
            result[col] = result[col].interpolate(method='time', limit_direction='both')
            
        elif method == 'median':
            # Replace with rolling median
            window = 20  # Same as default in detect_outliers
            rolling_median = result[col].rolling(window=window, center=True, min_periods=window//2).median()
            result.loc[outlier_indices, col] = rolling_median.loc[outlier_indices]
            
        elif method == 'ffill':
            # Set outliers to NaN
            result.loc[outlier_indices, col] = np.nan
            # Forward fill
            result[col] = result[col].fillna(method='ffill')
            # Backward fill any remaining NaNs at the beginning
            result[col] = result[col].fillna(method='bfill')
            
        else:
            logger.warning(f"Unknown replacement method: {method}")
    
    return result


def wavelet_denoise(data: Union[pd.Series, np.ndarray], 
                    wavelet: str = 'db4',
                    level: int = 3,
                    threshold_mode: str = 'soft') -> Union[pd.Series, np.ndarray]:
    """
    Apply wavelet denoising to a time series.
    
    Args:
        data: Time series data (Series or ndarray)
        wavelet: Wavelet type (default: 'db4' - Daubechies 4)
        level: Decomposition level
        threshold_mode: Thresholding mode ('soft' or 'hard')
    
    Returns:
        Denoised time series (same type as input)
    """
    # Handle input type
    is_series = isinstance(data, pd.Series)
    if is_series:
        index = data.index
        input_data = data.values
    else:
        input_data = data
    
    # Check for NaN values
    if np.isnan(input_data).any():
        logger.warning("NaN values found in data for wavelet denoising. Filling with interpolation.")
        if is_series:
            temp_series = pd.Series(input_data)
            temp_series = temp_series.interpolate(method='linear', limit_direction='both')
            input_data = temp_series.values
        else:
            # Simple linear interpolation for ndarray
            nan_indices = np.isnan(input_data)
            input_data = np.copy(input_data)  # Create a copy
            
            # Get indices of non-NaN values
            valid_indices = np.where(~nan_indices)[0]
            if len(valid_indices) > 0:
                # Create interpolation function
                if len(valid_indices) > 1:
                    f = interp1d(valid_indices, input_data[valid_indices], 
                                bounds_error=False, fill_value=input_data[valid_indices].mean())
                else:
                    # If only one valid value, fill with that value
                    fill_value = input_data[valid_indices[0]]
                    f = lambda x: np.full_like(x, fill_value, dtype=float)
                
                # Apply interpolation
                all_indices = np.arange(len(input_data))
                input_data = f(all_indices)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(input_data, wavelet, level=level)
    
    # Calculate threshold
    # Using the VisuShrink method: threshold = sigma * sqrt(2 * log(n))
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate of noise level
    n = len(input_data)
    threshold = sigma * np.sqrt(2 * np.log(n))
    
    # Apply thresholding to detail coefficients only
    new_coeffs = list(coeffs)
    for i in range(1, len(new_coeffs)):  # Skip approximation coefficients
        if threshold_mode == 'soft':
            new_coeffs[i] = pywt.threshold(new_coeffs[i], threshold, mode='soft')
        else:  # 'hard'
            new_coeffs[i] = pywt.threshold(new_coeffs[i], threshold, mode='hard')
    
    # Reconstruct signal
    denoised_data = pywt.waverec(new_coeffs, wavelet)
    
    # Ensure same length as input (sometimes wavelet transform changes length slightly)
    denoised_data = denoised_data[:len(input_data)]
    
    # Return in same format as input
    if is_series:
        return pd.Series(denoised_data, index=index)
    else:
        return denoised_data


def denoise_ohlcv(df: pd.DataFrame, 
                 wavelet: str = 'db4',
                 level: int = 3,
                 denoise_columns: List[str] = None) -> pd.DataFrame:
    """
    Apply wavelet denoising to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        wavelet: Wavelet type
        level: Decomposition level
        denoise_columns: Columns to denoise (default: open, high, low, close)
    
    Returns:
        DataFrame with denoised data
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to denoise_ohlcv")
        return df
    
    if denoise_columns is None:
        denoise_columns = ['open', 'high', 'low', 'close']
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    for col in denoise_columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Skip columns with all NaN
        if result[col].isna().all():
            logger.warning(f"Column '{col}' contains all NaN values")
            continue
        
        # Apply wavelet denoising
        logger.debug(f"Applying wavelet denoising to column '{col}'")
        result[col] = wavelet_denoise(result[col], wavelet=wavelet, level=level)
    
    # Ensure OHLC relationship is preserved after denoising
    # High should be the highest, low should be the lowest
    result['high'] = result[['open', 'high', 'low', 'close']].max(axis=1)
    result['low'] = result[['open', 'high', 'low', 'close']].min(axis=1)
    
    return result


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional derived features
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to calculate_derived_features")
        return df
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Basic price and volume features
    
    # Normalized bar range (as defined in the implementation steps)
    result['norm_range'] = (result['high'] - result['low']) / result['close']
    
    # Volume ratio
    result['volume_ratio'] = result['volume'] / result['volume'].shift(1)
    
    # Log returns
    result['log_return'] = np.log(result['close'] / result['close'].shift(1))
    
    # Moving averages (20, 50 periods)
    result['sma20'] = result['close'].rolling(window=20, min_periods=1).mean()
    result['sma50'] = result['close'].rolling(window=50, min_periods=1).mean()
    
    # Volatility estimate (20-period standard deviation of returns)
    result['volatility'] = result['log_return'].rolling(window=20, min_periods=5).std()
    
    # Additional useful features
    
    # Typical price
    result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
    
    # Average price as defined in the research paper
    result['avg_price'] = (result['open'] + result['high'] + result['low'] + result['close']) / 4
    
    # Price velocity (rate of change of price)
    result['price_velocity'] = result['close'].diff(5) / 5  # 5-period rate of change
    
    # Volume-weighted average price
    result['vwap'] = (result['typical_price'] * result['volume']).cumsum() / result['volume'].cumsum()
    
    # Relative volume (compared to 20-period average)
    result['rel_volume'] = result['volume'] / result['volume'].rolling(window=20, min_periods=1).mean()
    
    # Normalized daily price range
    # Assuming data is in 1-minute intervals, calculate day boundaries
    result['date'] = result.index.date
    day_highs = result.groupby('date')['high'].transform('max')
    day_lows = result.groupby('date')['low'].transform('min')
    result['norm_day_position'] = (result['close'] - day_lows) / (day_highs - day_lows)
    
    # Clean up
    result = result.drop(columns=['date'])
    
    # Replace infinite values with NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result


def prepare_data_for_transport(df: pd.DataFrame, 
                              num_price_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare price grids and weights for optimal transport calculation.
    
    Args:
        df: DataFrame with processed OHLCV data
        num_price_points: Number of price points in the grid
        
    Returns:
        Tuple of (price_grid, supply_weights, demand_weights)
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to prepare_data_for_transport")
        return np.array([]), np.array([]), np.array([])
    
    # Extract needed values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    volumes = df['volume'].values
    
    # Parameter for extending the price range slightly
    delta_ratio = 0.1
    
    # Create the price grid for each bar
    price_grids = []
    supply_weights = []
    demand_weights = []
    
    for i in range(len(df)):
        # Skip if any required data is missing
        if any(np.isnan([highs[i], lows[i], opens[i], closes[i], volumes[i]])):
            continue
        
        # Extend the range slightly
        price_range = highs[i] - lows[i]
        delta = price_range * delta_ratio
        min_price = lows[i] - delta
        max_price = highs[i] + delta
        
        # Create price grid
        price_grid = np.linspace(min_price, max_price, num_price_points)
        
        # Calculate average price and price volatility as in the paper
        avg_price = (opens[i] + highs[i] + lows[i] + closes[i]) / 4
        price_vol = (highs[i] - lows[i]) / 4
        
        # Calculate weighting functions for supply and demand
        # Using the formulas from the paper:
        # f^μ(p) = φ((p - p̄)/σp) * ((p - L)/(H - L))^α
        # f^ν(p) = φ((p - p̄)/σp) * ((H - p)/(H - L))^α
        
        # First, compute the normal density component
        if price_vol == 0:  # Avoid division by zero
            normal_density = np.ones(num_price_points)
        else:
            z_scores = (price_grid - avg_price) / price_vol
            normal_density = np.exp(-0.5 * z_scores**2) / (np.sqrt(2 * np.pi) * price_vol)
        
        # Compute the power law components with α=1
        if price_range == 0:  # Avoid division by zero
            supply_power = np.zeros(num_price_points)
            supply_power[price_grid >= lows[i]] = 1.0
            
            demand_power = np.zeros(num_price_points)
            demand_power[price_grid <= highs[i]] = 1.0
        else:
            supply_power = np.maximum(0, (price_grid - lows[i]) / price_range)
            demand_power = np.maximum(0, (highs[i] - price_grid) / price_range)
        
        # Combined weighting functions
        supply_weights_raw = normal_density * supply_power
        demand_weights_raw = normal_density * demand_power
        
        # Normalize weights to sum to 1 (scaled by volume)
        supply_weights_norm = supply_weights_raw / np.sum(supply_weights_raw) * volumes[i]
        demand_weights_norm = demand_weights_raw / np.sum(demand_weights_raw) * volumes[i]
        
        # Add to results
        price_grids.append(price_grid)
        supply_weights.append(supply_weights_norm)
        demand_weights.append(demand_weights_norm)
    
    return price_grids, supply_weights, demand_weights


def process_ohlcv_data(df: pd.DataFrame, 
                      clean: bool = True, 
                      handle_outliers: bool = True,
                      denoise: bool = True,
                      calculate_features: bool = True) -> pd.DataFrame:
    """
    Complete data processing pipeline for OHLCV data.
    
    Args:
        df: Raw OHLCV DataFrame
        clean: Whether to clean the data
        handle_outliers: Whether to detect and replace outliers
        denoise: Whether to apply wavelet denoising
        calculate_features: Whether to calculate derived features
    
    Returns:
        Processed DataFrame
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to process_ohlcv_data")
        return df
    
    logger.info(f"Processing OHLCV data with {len(df)} bars")
    
    result = df.copy()
    
    # 1. Clean data
    if clean:
        logger.info("Cleaning OHLCV data")
        result = clean_ohlcv(result)
    
    # 2. Handle outliers
    if handle_outliers:
        logger.info("Detecting and replacing outliers")
        outliers = detect_outliers(result)
        if not outliers.empty:
            result = replace_outliers(result, outliers)
    
    # 3. Apply wavelet denoising
    if denoise:
        logger.info("Applying wavelet denoising")
        result = denoise_ohlcv(result)
    
    # 4. Calculate derived features
    if calculate_features:
        logger.info("Calculating derived features")
        result = calculate_derived_features(result)
    
    logger.info("OHLCV data processing complete")
    return result