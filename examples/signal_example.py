"""
Example script demonstrating the signal generation and processing functionality.

This script shows how to:
1. Generate signals from OHLCV data using optimal transport
2. Process and enhance signals for better trading decisions
3. Apply multi-timeframe signal combination
4. Evaluate trading performance
"""
import os
import sys
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ohlcv_transport.signals.generators import process_signal_pipeline
from ohlcv_transport.signals.processing import apply_signal_filter, atr_bands, fractional_differentiation

# Add parent directory to Python path to make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from ohlcv_transport.config import config_manager
from ohlcv_transport.data import (
    get_complete_dataset, 
    get_multi_timeframe_data
)
from ohlcv_transport.models import (
    process_ohlcv_with_transport
)
from ohlcv_transport.signals import (
    calculate_base_imbalance_signal,
    calculate_multi_timeframe_signal,
    add_signal_features,
    apply_kalman_filter,
    apply_wavelet_denoising,
    apply_exponential_smoothing,
    generate_trading_signal,
    apply_minimum_holding_period,
    calculate_trading_metrics,
    normalize_signal,
    combine_signals,
    detect_signal_regime,
    adaptive_thresholds,
    detect_divergence,
    signal_to_position_sizing,
    calculate_dynamic_lookback,
    ewma_crossover_signal
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("signal_example")


def setup_api_key():
    """Setup the API key if not already configured."""
    if not config_manager.validate_api_key():
        print("\n" + "="*50)
        print("Twelve Data API key not configured.")
        print("Get a free API key at: https://twelvedata.com/apikey")
        print("="*50 + "\n")
        
        api_key = input("Enter your Twelve Data API key: ").strip()
        
        if api_key:
            config_manager.set_api_key(api_key)
            print("API key configured successfully!")
        else:
            print("No API key provided. Exiting.")
            sys.exit(1)


def get_sample_data(use_api=True):
    """
    Get sample data for demonstration purposes.
    
    Args:
        use_api: Whether to use the API or generate synthetic data
        
    Returns:
        DataFrame with OHLCV data and transport-based features
    """
    # Option 1: Use real data from API
    if use_api and config_manager.validate_api_key():
        print("Retrieving sample data from API...")
        try:
            df = get_complete_dataset(
                symbol="SPY",
                interval="1h",  # Use daily data for more reliability
                days_back=100,
                preprocess=True,
                use_cache=True
            )
            
            print(f"Retrieved {len(df)} bars for SPY")
            
            # Apply optimal transport to calculate Wasserstein distances
            df = process_ohlcv_with_transport(df)
            
            return df
        except Exception as e:
            print(f"Error retrieving data: {e}")
            print("Falling back to synthetic data")
    
    # Option 2: Generate synthetic data
    print("Generating synthetic sample data...")
    
    # Date range
    dates = pd.date_range(start='2024-04-01 09:30:00', periods=200, freq='1min')
    
    # Base price series with random walk
    np.random.seed(42)  # For reproducibility
    close = 100 + np.cumsum(np.random.normal(0, 0.1, 200))
    
    # Add a trend
    close = close + np.linspace(0, 5, 200)
    
    # Create OHLC data with realistic relationships
    high = close + np.random.uniform(0, 0.5, 200)
    low = close - np.random.uniform(0, 0.5, 200)
    open_price = low + np.random.uniform(0, 1, 200) * (high - low)
    
    # Volume with some randomness
    volume = 1000 + np.random.uniform(0, 500, 200)
    
    # Add volume spikes at certain points
    volume[30:35] *= 3
    volume[70:75] *= 2
    volume[150:155] *= 4
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        # Add synthetic Wasserstein distances
        'w_distance': 1.0 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, 200)) + 0.2 * np.random.normal(0, 1, 200)
    }, index=dates)
    
    print(f"Generated {len(df)} synthetic bars")
    return df


def get_multi_timeframe_sample_data(use_api=True):
    """
    Get sample data at multiple timeframes.
    
    Args:
        use_api: Whether to use the API or generate synthetic data
        
    Returns:
        Dictionary mapping timeframes to DataFrames
    """
    if use_api and config_manager.validate_api_key():
        print("Retrieving multi-timeframe data from API...")
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            
            # Get data at multiple timeframes
            timeframes = ['1min', '5min', '15min', '1h']
            multi_tf_data = get_multi_timeframe_data(
                symbol="SPY",
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date
            )
            
            # Process each timeframe with optimal transport
            processed_data = {}
            for tf, df in multi_tf_data.items():
                if not df.empty:
                    processed_data[tf] = process_ohlcv_with_transport(df)
                    print(f"Processed {len(processed_data[tf])} bars for {tf} timeframe")
            
            return processed_data
        except Exception as e:
            print(f"Error retrieving multi-timeframe data: {e}")
            print("Falling back to synthetic data")
    
    # Generate synthetic multi-timeframe data
    print("Generating synthetic multi-timeframe data...")
    
    # Generate base timeframe (1min)
    df_1min = get_sample_data(use_api=False)
    
    # Resample to higher timeframes
    df_5min = df_1min.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'w_distance': 'mean'
    }).dropna()
    
    df_15min = df_1min.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'w_distance': 'mean'
    }).dropna()
    
    df_1h = df_1min.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'w_distance': 'mean'
    }).dropna()
    
    # Calculate base signal for each timeframe
    for df in [df_1min, df_5min, df_15min, df_1h]:
        df = calculate_base_imbalance_signal(df, 'w_distance')
    
    # Create dictionary of DataFrames
    multi_tf_data = {
        '1min': df_1min,
        '5min': df_5min,
        '15min': df_15min,
        '1h': df_1h
    }
    
    return multi_tf_data


def demonstrate_basic_signal_generation(df):
    """
    Demonstrate basic signal generation from Wasserstein distances.
    
    Args:
        df: DataFrame with Wasserstein distances
        
    Returns:
        DataFrame with basic signals added
    """
    print("\n" + "="*50)
    print("BASIC SIGNAL GENERATION")
    print("="*50)
    
    # Calculate basic imbalance signal
    result = calculate_base_imbalance_signal(df, lookback=20)
    print(f"Generated base imbalance signal using 20-period lookback")
    
    # Add derived features
    result = add_signal_features(result)
    print("Added derived signal features")
    
    # Print statistics
    if 'imbalance_signal' in result.columns:
        signal = result['imbalance_signal'].dropna()
        print("\nImbalance signal statistics:")
        print(f"  Mean: {signal.mean():.4f}")
        print(f"  Std Dev: {signal.std():.4f}")
        print(f"  Min: {signal.min():.4f}")
        print(f"  Max: {signal.max():.4f}")
        print(f"  Autocorrelation(1): {signal.autocorr(lag=1):.4f}")
        
        # Plot signal
        if input("\nVisualize basic signal? (y/n, default: y): ").strip().lower() != 'n':
            plt.figure(figsize=(12, 8))
            
            # Plot price and signal
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(result.index, result['close'], 'b-')
            ax1.set_ylabel('Price')
            ax1.set_title('Price vs Signal')
            
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(result.index, result['imbalance_signal'], 'g-')
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax2.set_ylabel('Signal')
            
            plt.tight_layout()
            plt.show()
    
    return result


def demonstrate_signal_smoothing(df):
    """
    Demonstrate different signal smoothing techniques.
    
    Args:
        df: DataFrame with base signal
        
    Returns:
        DataFrame with smoothed signals added
    """
    print("\n" + "="*50)
    print("SIGNAL SMOOTHING TECHNIQUES")
    print("="*50)
    
    if 'imbalance_signal' not in df.columns:
        print("Base signal not found. Calculating...")
        df = calculate_base_imbalance_signal(df)
    
    # Apply Kalman filter
    df = apply_kalman_filter(df)
    print("Applied Kalman filter smoothing")
    
    # Apply wavelet denoising
    df = apply_wavelet_denoising(df)
    print("Applied wavelet denoising")
    
    # Apply exponential smoothing
    df = apply_exponential_smoothing(df, alpha=0.2)
    print("Applied exponential smoothing with alpha=0.2")
    
    # Compare smoothing methods
    if input("\nVisualize smoothed signals? (y/n, default: y): ").strip().lower() != 'n':
        plt.figure(figsize=(12, 10))
        
        # Plot all signals
        ax = plt.subplot(2, 1, 1)
        ax.plot(df.index, df['imbalance_signal'], 'b-', alpha=0.5, label='Raw')
        ax.plot(df.index, df['kalman_signal'], 'r-', label='Kalman')
        ax.plot(df.index, df['wavelet_signal'], 'g-', label='Wavelet')
        ax.plot(df.index, df['ema_signal'], 'c-', label='EMA')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.set_ylabel('Signal')
        ax.set_title('Comparison of Smoothing Techniques')
        ax.legend()
        
        # Plot differences
        ax2 = plt.subplot(2, 1, 2, sharex=ax)
        ax2.plot(df.index, (df['kalman_signal'] - df['imbalance_signal']), 'r-', 
                 label='Kalman - Raw', alpha=0.7)
        ax2.plot(df.index, (df['wavelet_signal'] - df['imbalance_signal']), 'g-', 
                 label='Wavelet - Raw', alpha=0.7)
        ax2.plot(df.index, (df['ema_signal'] - df['imbalance_signal']), 'c-', 
                 label='EMA - Raw', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.set_ylabel('Difference')
        ax2.set_title('Signal Differences')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Calculate volatility reduction
    print("\nVolatility reduction from smoothing:")
    raw_vol = df['imbalance_signal'].diff().std()
    kalman_vol = df['kalman_signal'].diff().std()
    wavelet_vol = df['wavelet_signal'].diff().std()
    ema_vol = df['ema_signal'].diff().std()
    
    print(f"  Raw signal volatility: {raw_vol:.6f}")
    print(f"  Kalman signal volatility: {kalman_vol:.6f} ({(1 - kalman_vol/raw_vol)*100:.2f}% reduction)")
    print(f"  Wavelet signal volatility: {wavelet_vol:.6f} ({(1 - wavelet_vol/raw_vol)*100:.2f}% reduction)")
    print(f"  EMA signal volatility: {ema_vol:.6f} ({(1 - ema_vol/raw_vol)*100:.2f}% reduction)")
    
    return df


def demonstrate_multi_timeframe_signals(multi_tf_data):
    """
    Demonstrate multi-timeframe signal combination.
    
    Args:
        multi_tf_data: Dictionary mapping timeframes to DataFrames
        
    Returns:
        DataFrame with multi-timeframe signal added
    """
    print("\n" + "="*50)
    print("MULTI-TIMEFRAME SIGNAL COMBINATION")
    print("="*50)
    
    # Ensure all DataFrames have the base signal
    for tf, df in multi_tf_data.items():
        if 'imbalance_signal' not in df.columns:
            multi_tf_data[tf] = calculate_base_imbalance_signal(df)
    
    # Calculate multi-timeframe signal
    result_df = calculate_multi_timeframe_signal(multi_tf_data)
    print(f"Combined signals from {len(multi_tf_data)} timeframes")
    
    # Get a list of all timeframes
    timeframes = list(multi_tf_data.keys())
    timeframes.sort(key=lambda x: len(multi_tf_data[x]), reverse=True)  # Sort by bar count
    
    # Print correlation between signals at different timeframes
    print("\nCorrelation between timeframe signals:")
    base_tf = timeframes[-1]  # Lowest timeframe
    base_signal = multi_tf_data[base_tf]['imbalance_signal']
    
    for tf in timeframes[:-1]:  # Skip base timeframe
        tf_signal = multi_tf_data[tf]['imbalance_signal']
        
        # Resample higher timeframe signal to match base timeframe
        resampled_signal = tf_signal.reindex(base_signal.index, method='ffill')
        
        # Calculate correlation
        correlation = base_signal.corr(resampled_signal)
        print(f"  {base_tf} vs {tf}: {correlation:.4f}")
    
    # Compare multi-timeframe signal with base timeframe signal
    if 'multi_tf_signal' in result_df.columns:
        correlation = result_df['imbalance_signal'].corr(result_df['multi_tf_signal'])
        print(f"\nCorrelation between base and multi-timeframe signal: {correlation:.4f}")
        
        # Plot signals
        if input("\nVisualize multi-timeframe signals? (y/n, default: y): ").strip().lower() != 'n':
            plt.figure(figsize=(12, 10))
            
            # Plot price
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(result_df.index, result_df['close'], 'b-')
            ax1.set_ylabel('Price')
            ax1.set_title('Price Chart')
            
            # Plot base signal
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            for tf in timeframes:
                # Get signal
                tf_signal = multi_tf_data[tf]['imbalance_signal']
                
                # Resample to base timeframe
                resampled_signal = tf_signal.reindex(result_df.index, method='ffill')
                
                # Plot with alpha based on timeframe (higher tf = less transparent)
                alpha = 0.3 + 0.5 * timeframes.index(tf) / len(timeframes)
                ax2.plot(result_df.index, resampled_signal, alpha=alpha, label=tf)
            
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax2.set_ylabel('Signal')
            ax2.set_title('Signals at Different Timeframes')
            ax2.legend()
            
            # Plot multi-timeframe signal
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(result_df.index, result_df['imbalance_signal'], 'b-', alpha=0.5, label='Base')
            ax3.plot(result_df.index, result_df['multi_tf_signal'], 'r-', label='Multi-TF')
            ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            ax3.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax3.set_ylabel('Signal')
            ax3.set_title('Base vs Multi-Timeframe Signal')
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
    
    return result_df


def demonstrate_regime_detection(df):
    """
    Demonstrate signal regime detection and adaptive thresholds.
    
    Args:
        df: DataFrame with signal
        
    Returns:
        DataFrame with regime and adaptive thresholds added
    """
    print("\n" + "="*50)
    print("SIGNAL REGIME DETECTION")
    print("="*50)
    
    # Calculate base signal if not present
    if 'imbalance_signal' not in df.columns:
        df = calculate_base_imbalance_signal(df)
    
    # Use kalman-filtered signal if available, otherwise use base signal
    signal_col = 'kalman_signal' if 'kalman_signal' in df.columns else 'imbalance_signal'
    
    # Detect signal regimes
    regime = detect_signal_regime(df[signal_col], lookback=30, threshold=1.5)
    df['signal_regime'] = regime
    print("Detected signal regimes using 30-period lookback and 1.5 threshold")
    
    # Count occurrences of each regime
    regime_counts = df['signal_regime'].value_counts()
    print("\nRegime distribution:")
    for regime_id, count in regime_counts.items():
        print(f"  Regime {regime_id}: {count} bars ({count/len(df)*100:.1f}%)")
    
    # Calculate adaptive thresholds
    upper_thresh, lower_thresh = adaptive_thresholds(
        df[signal_col], 
        df['signal_regime'],
        base_threshold=1.0,
        vol_scaling=True
    )
    df['upper_threshold'] = upper_thresh
    df['lower_threshold'] = lower_thresh
    print("Calculated adaptive thresholds based on regimes")
    
    # Generate trading signals with adaptive thresholds
    df['adaptive_position'] = 0
    df.loc[df[signal_col] > df['upper_threshold'], 'adaptive_position'] = 1
    df.loc[df[signal_col] < df['lower_threshold'], 'adaptive_position'] = -1
    
    # Generate trading signals with fixed thresholds for comparison
    fixed_df = generate_trading_signal(
        df,
        signal_col=signal_col,
        threshold_long=1.0,
        threshold_short=-1.0
    )
    df['fixed_position'] = fixed_df['position']
    
    # Compare fixed vs adaptive thresholds
    fixed_signal_bars = (df['fixed_position'] != 0).sum()
    adaptive_signal_bars = (df['adaptive_position'] != 0).sum()
    
    print("\nSignal comparison:")
    print(f"  Fixed thresholds: {fixed_signal_bars} signal bars ({fixed_signal_bars/len(df)*100:.1f}%)")
    print(f"  Adaptive thresholds: {adaptive_signal_bars} signal bars ({adaptive_signal_bars/len(df)*100:.1f}%)")
    
    # Calculate difference in signals
    different_signals = (df['fixed_position'] != df['adaptive_position']).sum()
    print(f"  Signals differ in {different_signals} bars ({different_signals/len(df)*100:.1f}%)")
    
    # Visualize regimes and thresholds
    if input("\nVisualize regimes and thresholds? (y/n, default: y): ").strip().lower() != 'n':
        plt.figure(figsize=(12, 10))
        
        # Plot signal and thresholds
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df.index, df[signal_col], 'b-', label='Signal')
        ax1.plot(df.index, df['upper_threshold'], 'r--', label='Upper Threshold')
        ax1.plot(df.index, df['lower_threshold'], 'g--', label='Lower Threshold')
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Fixed Threshold')
        ax1.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # Highlight regimes with background color
        regime_changes = df['signal_regime'].diff().fillna(0) != 0
        regime_start_indices = df.index[regime_changes]
        
        for i in range(len(regime_start_indices)):
            start_idx = regime_start_indices[i]
            end_idx = regime_start_indices[i+1] if i < len(regime_start_indices) - 1 else df.index[-1]
            
            regime_value = df.loc[start_idx, 'signal_regime']
            color = 'lightcoral' if regime_value == 1 else 'lightblue'
            
            ax1.axvspan(start_idx, end_idx, alpha=0.2, color=color)
        
        ax1.set_ylabel('Signal')
        ax1.set_title('Signal with Adaptive Thresholds')
        ax1.legend()
        
        # Plot positions
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df.index, df['fixed_position'], 'b-', label='Fixed Thresholds')
        ax2.plot(df.index, df['adaptive_position'], 'r-', alpha=0.7, label='Adaptive Thresholds')
        ax2.set_ylabel('Position')
        ax2.set_title('Trading Positions: Fixed vs Adaptive')
        ax2.set_yticks([-1, 0, 1])
        ax2.legend()
        
        # Plot price
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(df.index, df['close'], 'k-')
        ax3.set_ylabel('Price')
        ax3.set_title('Price Chart')
        
        plt.tight_layout()
        plt.show()
    
    return df


def demonstrate_position_sizing(df):
    """
    Demonstrate position sizing based on signal strength.
    
    Args:
        df: DataFrame with signal
        
    Returns:
        DataFrame with position sizes added
    """
    print("\n" + "="*50)
    print("SIGNAL-BASED POSITION SIZING")
    print("="*50)
    
    # Use appropriate signal column
    signal_col = 'kalman_signal' if 'kalman_signal' in df.columns else 'imbalance_signal'
    
    # Convert signal to position sizes
    position_sizes = signal_to_position_sizing(
        df[signal_col],
        base_position=1.0,
        max_position=2.0,
        threshold=1.0
    )
    df['position_size'] = position_sizes
    print("Calculated position sizes based on signal strength")
    
    # Print position size statistics
    print("\nPosition size statistics:")
    print(f"  Mean absolute size: {df['position_size'].abs().mean():.4f}")
    print(f"  Max long position: {df['position_size'].max():.4f}")
    print(f"  Max short position: {df['position_size'].min():.4f}")
    
    # Binary position signal (for comparison)
    df['binary_position'] = np.sign(df[signal_col])
    
    # Calculate returns using position sizing vs binary positions
    if 'close' in df.columns:
        price_returns = df['close'].pct_change()
        df['sized_returns'] = df['position_size'].shift(1) * price_returns
        df['binary_returns'] = df['binary_position'].shift(1) * price_returns
        
        # Calculate cumulative returns
        df['sized_cum_return'] = (1 + df['sized_returns']).cumprod()
        df['binary_cum_return'] = (1 + df['binary_returns']).cumprod()
        
        # Compare performance
        print("\nPerformance comparison:")
        print(f"  Position sizing total return: {(df['sized_cum_return'].iloc[-1] - 1) * 100:.2f}%")
        print(f"  Binary position total return: {(df['binary_cum_return'].iloc[-1] - 1) * 100:.2f}%")
        
        # Calculate metrics
        sized_sharpe = df['sized_returns'].mean() / df['sized_returns'].std() * np.sqrt(252)
        binary_sharpe = df['binary_returns'].mean() / df['binary_returns'].std() * np.sqrt(252)
        
        print(f"  Position sizing Sharpe ratio: {sized_sharpe:.4f}")
        print(f"  Binary position Sharpe ratio: {binary_sharpe:.4f}")
        
        # Visualize position sizing
        if input("\nVisualize position sizing? (y/n, default: y): ").strip().lower() != 'n':
            plt.figure(figsize=(12, 10))
            
            # Plot signal and positions
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(df.index, df[signal_col], 'b-', label='Signal')
            ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            ax1.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3)
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax1.set_ylabel('Signal')
            ax1.set_title('Signal and Position Sizes')
            
            # Add position size as bars
            ax1t = ax1.twinx()
            ax1t.bar(df.index, df['position_size'], alpha=0.3, color='g', label='Position Size')
            ax1t.set_ylabel('Position Size')
            
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1t.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            
            # Plot returns
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(df.index, df['sized_returns'].rolling(5).mean(), 'g-', label='Sized Returns (5MA)')
            ax2.plot(df.index, df['binary_returns'].rolling(5).mean(), 'b-', label='Binary Returns (5MA)')
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax2.set_ylabel('Returns')
            ax2.set_title('Rolling Average Returns')
            ax2.legend()
            
            # Plot cumulative returns
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(df.index, df['sized_cum_return'], 'g-', label='Position Sizing')
            ax3.plot(df.index, df['binary_cum_return'], 'b-', label='Binary Positions')
            ax3.set_ylabel('Cumulative Return')
            ax3.set_title('Performance Comparison')
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
    
    return df


def demonstrate_advanced_features(df):
    """
    Demonstrate advanced signal processing features.
    
    Args:
        df: DataFrame with signal
        
    Returns:
        DataFrame with advanced features added
    """
    print("\n" + "="*50)
    print("ADVANCED SIGNAL PROCESSING")
    print("="*50)
    
    # Use appropriate signal column
    signal_col = 'kalman_signal' if 'kalman_signal' in df.columns else 'imbalance_signal'
    
    # Apply fractional differentiation
    frac_diff = fractional_differentiation(df[signal_col], order=0.4, window=20)
    df['frac_diff'] = frac_diff
    print("Applied fractional differentiation with order=0.4")
    
    # Apply signal filters
    df['lowpass_signal'] = apply_signal_filter(df[signal_col], filter_type='lowpass', cutoff=0.1)
    df['highpass_signal'] = apply_signal_filter(df[signal_col], filter_type='highpass', cutoff=0.3)
    df['bandpass_signal'] = apply_signal_filter(
        df[signal_col], 
        filter_type='bandpass', 
        cutoff=(0.05, 0.3)
    )
    print("Applied lowpass, highpass, and bandpass filters")
    
    # Detect divergences
    if 'close' in df.columns:
        divergence = detect_divergence(df['close'], df[signal_col], window=20)
        df['divergence'] = divergence
        print("Detected price-signal divergences")
        
        # Count divergences
        bullish_count = (divergence == 1).sum()
        bearish_count = (divergence == -1).sum()
        print(f"  Bullish divergences: {bullish_count}")
        print(f"  Bearish divergences: {bearish_count}")
    
    # Calculate ATR bands
    if all(col in df.columns for col in ['close', 'high', 'low']):
        upper_band, lower_band = atr_bands(df, period=14, multiplier=2.0)
        df['atr_upper'] = upper_band
        df['atr_lower'] = lower_band
        print("Calculated ATR bands with 14-period ATR")
    
    # Calculate dynamic lookback periods
    dynamic_lookback = calculate_dynamic_lookback(
        df[signal_col],
        min_lookback=10,
        max_lookback=50
    )
    df['dynamic_lookback'] = dynamic_lookback
    print("Calculated dynamic lookback periods")
    
    # Generate EWMA crossover signals
    ewma_signal = ewma_crossover_signal(
        df[signal_col],
        fast_period=5,
        slow_period=20
    )
    df['ewma_signal'] = ewma_signal
    print("Generated EWMA crossover signals")
    
    # Visualize advanced features
    if input("\nVisualize advanced features? (y/n, default: y): ").strip().lower() != 'n':
        plt.figure(figsize=(15, 12))
        
        # Plot signal filters
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(df.index, df[signal_col], 'k-', alpha=0.5, label='Original')
        ax1.plot(df.index, df['lowpass_signal'], 'r-', label='Lowpass')
        ax1.plot(df.index, df['highpass_signal'], 'b-', label='Highpass')
        ax1.plot(df.index, df['bandpass_signal'], 'g-', label='Bandpass')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax1.set_title('Signal Filters')
        ax1.legend()
        
        # Plot fractional differentiation
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(df.index, df[signal_col], 'k-', alpha=0.5, label='Original')
        ax2.plot(df.index, df['frac_diff'], 'r-', label='Frac Diff (d=0.4)')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.set_title('Fractional Differentiation')
        ax2.legend()
        
        # Plot price with ATR bands and divergences
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(df.index, df['close'], 'k-', label='Close')
        
        if 'atr_upper' in df.columns and 'atr_lower' in df.columns:
            ax3.plot(df.index, df['atr_upper'], 'r--', alpha=0.5, label='ATR Upper')
            ax3.plot(df.index, df['atr_lower'], 'g--', alpha=0.5, label='ATR Lower')
        
        if 'divergence' in df.columns:
            # Plot bullish divergences
            bullish_idx = df.index[df['divergence'] == 1]
            if len(bullish_idx) > 0:
                ax3.scatter(bullish_idx, df.loc[bullish_idx, 'close'], 
                          marker='^', color='g', s=100, label='Bullish Div')
            
            # Plot bearish divergences
            bearish_idx = df.index[df['divergence'] == -1]
            if len(bearish_idx) > 0:
                ax3.scatter(bearish_idx, df.loc[bearish_idx, 'close'], 
                          marker='v', color='r', s=100, label='Bearish Div')
        
        ax3.set_title('Price with ATR Bands and Divergences')
        ax3.legend()
        
        # Plot dynamic lookback periods
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(df.index, df['dynamic_lookback'], 'b-')
        ax4.set_ylabel('Periods')
        ax4.set_title('Dynamic Lookback Periods')
        
        # Plot EWMA crossover signals
        ax5 = plt.subplot(3, 2, 5)
        
        # Plot signal
        ax5.plot(df.index, df[signal_col], 'k-', alpha=0.7, label='Signal')
        
        # Add EMA lines
        fast_ema = df[signal_col].ewm(span=5, adjust=False).mean()
        slow_ema = df[signal_col].ewm(span=20, adjust=False).mean()
        ax5.plot(df.index, fast_ema, 'r-', alpha=0.7, label='Fast EMA (5)')
        ax5.plot(df.index, slow_ema, 'b-', alpha=0.7, label='Slow EMA (20)')
        
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax5.set_title('Signal with EMAs')
        ax5.legend()
        
        # Plot crossover signals
        ax6 = plt.subplot(3, 2, 6, sharex=ax5)
        ax6.bar(df.index, df['ewma_signal'], color='g', alpha=0.7)
        ax6.set_title('EWMA Crossover Signals')
        ax6.set_yticks([-1, 0, 1])
        
        plt.tight_layout()
        plt.show()
    
    return df


def demonstrate_trading_strategy(df):
    """
    Demonstrate a complete trading strategy using processed signals.
    
    Args:
        df: DataFrame with signals
        
    Returns:
        DataFrame with trading strategy results
    """
    print("\n" + "="*50)
    print("COMPLETE TRADING STRATEGY")
    print("="*50)
    
    # Process signal pipeline
    if 'imbalance_signal' not in df.columns and 'w_distance' in df.columns:
        df = process_signal_pipeline(
            df, 
            w_distance_col='w_distance',
            lookback=20,
            smooth_method='kalman',
            threshold_long=1.0,
            threshold_short=-1.0,
            min_holding=3,
            adaptive_threshold=True
        )
        print("Applied complete signal processing pipeline")
    
    # Use the appropriate signal and position columns
    signal_col = 'kalman_signal' if 'kalman_signal' in df.columns else 'imbalance_signal'
    position_col = 'position' if 'position' in df.columns else 'adaptive_position'
    
    if position_col not in df.columns:
        # Generate trading signals with adaptive thresholds if available
        if 'upper_threshold' in df.columns and 'lower_threshold' in df.columns:
            df[position_col] = 0
            df.loc[df[signal_col] > df['upper_threshold'], position_col] = 1
            df.loc[df[signal_col] < df['lower_threshold'], position_col] = -1
        else:
            # Use fixed thresholds
            df = generate_trading_signal(
                df,
                signal_col=signal_col,
                threshold_long=1.0,
                threshold_short=-1.0,
                position_col=position_col
            )
        
        # Apply minimum holding period
        df = apply_minimum_holding_period(df, position_col=position_col, min_periods=3)
    
    # Calculate strategy returns and performance metrics
    if 'close' in df.columns:
        price_returns = df['close'].pct_change()
        df['strategy_returns'] = df[position_col].shift(1) * price_returns
        
        # Calculate buy & hold returns for comparison
        df['bh_returns'] = price_returns
        
        # Calculate cumulative returns
        df['strategy_cumulative'] = (1 + df['strategy_returns']).cumprod()
        df['bh_cumulative'] = (1 + df['bh_returns']).cumprod()
        
        # Calculate drawdowns
        df['strategy_dd'] = (df['strategy_cumulative'] / df['strategy_cumulative'].cummax() - 1)
        df['bh_dd'] = (df['bh_cumulative'] / df['bh_cumulative'].cummax() - 1)
        
        # Calculate performance metrics
        metrics = calculate_trading_metrics(df, position_col=position_col)
        
        # Print performance metrics
        print("\nStrategy performance metrics:")
        print(f"  Total return: {metrics['total_return']*100:.2f}%")
        print(f"  Annualized return: {metrics['annualized_return']*100:.2f}%")
        print(f"  Volatility: {metrics['volatility']*100:.2f}%")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Max drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Win rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Number of trades: {metrics['trade_count']}")
        
        # Compare with buy & hold
        bh_total_return = df['bh_cumulative'].iloc[-1] - 1
        bh_max_dd = df['bh_dd'].min()
        print(f"\nBuy & Hold total return: {bh_total_return*100:.2f}%")
        print(f"Buy & Hold max drawdown: {bh_max_dd*100:.2f}%")
        
        # Calculate outperformance
        outperformance = metrics['total_return'] - bh_total_return
        print(f"\nStrategy outperformance: {outperformance*100:.2f}%")
        
        # Visualize trading strategy results
        if input("\nVisualize trading strategy results? (y/n, default: y): ").strip().lower() != 'n':
            plt.figure(figsize=(12, 15))
            
            # Plot signal and positions
            ax1 = plt.subplot(4, 1, 1)
            ax1.plot(df.index, df[signal_col], 'b-', label='Signal')
            
            if 'upper_threshold' in df.columns and 'lower_threshold' in df.columns:
                ax1.plot(df.index, df['upper_threshold'], 'r--', alpha=0.5, label='Upper Threshold')
                ax1.plot(df.index, df['lower_threshold'], 'g--', alpha=0.5, label='Lower Threshold')
            else:
                ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Upper Threshold')
                ax1.axhline(y=-1.0, color='g', linestyle='--', alpha=0.5, label='Lower Threshold')
            
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Plot positions as background
            for i in range(1, len(df)):
                if df[position_col].iloc[i] == 1:  # Long position
                    ax1.axvspan(df.index[i-1], df.index[i], alpha=0.2, color='green')
                elif df[position_col].iloc[i] == -1:  # Short position
                    ax1.axvspan(df.index[i-1], df.index[i], alpha=0.2, color='red')
            
            ax1.set_ylabel('Signal')
            ax1.set_title('Signal and Positions')
            ax1.legend()
            
            # Plot price chart with positions
            ax2 = plt.subplot(4, 1, 2, sharex=ax1)
            ax2.plot(df.index, df['close'], 'k-')
            
            # Plot buy/sell markers
            buy_signals = (df[position_col] == 1) & (df[position_col].shift(1) != 1)
            sell_signals = (df[position_col] == -1) & (df[position_col].shift(1) != -1)
            exit_signals = (df[position_col] == 0) & (df[position_col].shift(1) != 0)
            
            if buy_signals.any():
                ax2.scatter(df.index[buy_signals], df.loc[buy_signals, 'close'], 
                           marker='^', color='g', s=100, label='Buy')
            
            if sell_signals.any():
                ax2.scatter(df.index[sell_signals], df.loc[sell_signals, 'close'], 
                           marker='v', color='r', s=100, label='Sell')
            
            if exit_signals.any():
                ax2.scatter(df.index[exit_signals], df.loc[exit_signals, 'close'], 
                           marker='o', color='blue', s=100, label='Exit')
            
            ax2.set_ylabel('Price')
            ax2.set_title('Price Chart with Entry/Exit Points')
            ax2.legend()
            
            # Plot cumulative returns
            ax3 = plt.subplot(4, 1, 3, sharex=ax1)
            ax3.plot(df.index, df['strategy_cumulative'], 'g-', label='Strategy')
            ax3.plot(df.index, df['bh_cumulative'], 'b-', label='Buy & Hold')
            ax3.set_ylabel('Cumulative Return')
            ax3.set_title('Performance Comparison')
            ax3.legend()
            
            # Plot drawdowns
            ax4 = plt.subplot(4, 1, 4, sharex=ax1)
            ax4.fill_between(df.index, 0, df['strategy_dd'] * 100, color='r', alpha=0.3, label='Strategy')
            ax4.fill_between(df.index, 0, df['bh_dd'] * 100, color='b', alpha=0.3, label='Buy & Hold')
            ax4.set_ylabel('Drawdown (%)')
            ax4.set_title('Drawdown Comparison')
            ax4.set_ylim(min(df['strategy_dd'].min(), df['bh_dd'].min()) * 100 * 1.1, 0)
            ax4.legend()
            
            plt.tight_layout()
            plt.show()
    
    return df


def main():
    """Main function to run the demonstration."""
    print("\nOHLCV Optimal Transport Signal Processing Demonstration")
    print("="*60 + "\n")
    
    # Setup API key if needed
    setup_api_key()
    
    # Ask if want to use API or synthetic data
    use_api = False
    api_key_valid = config_manager.validate_api_key()
    
    if api_key_valid:
        use_api = input("Use real market data from API? (y/n, default: n): ").strip().lower() == 'y'
    
    # Get sample data
    df = get_sample_data(use_api=use_api)
    
    # Get multi-timeframe data
    multi_tf_data = get_multi_timeframe_sample_data(use_api=use_api)
    
    while True:
        print("\nOptions:")
        print("1. Demonstrate basic signal generation")
        print("2. Demonstrate signal smoothing techniques")
        print("3. Demonstrate multi-timeframe signals")
        print("4. Demonstrate regime detection")
        print("5. Demonstrate position sizing")
        print("6. Demonstrate advanced features")
        print("7. Demonstrate complete trading strategy")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            df = demonstrate_basic_signal_generation(df)
        elif choice == '2':
            df = demonstrate_signal_smoothing(df)
        elif choice == '3':
            _ = demonstrate_multi_timeframe_signals(multi_tf_data)
        elif choice == '4':
            df = demonstrate_regime_detection(df)
        elif choice == '5':
            df = demonstrate_position_sizing(df)
        elif choice == '6':
            df = demonstrate_advanced_features(df)
        elif choice == '7':
            df = demonstrate_trading_strategy(df)
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()