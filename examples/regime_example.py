"""
Example script demonstrating the regime detection and calibration functionality.

This script shows how to:
1. Detect market regimes using Wasserstein distances
2. Calibrate strategy parameters based on the detected regime
3. Apply regime-specific parameters to trading decisions
4. Visualize regime changes and regime-specific performance
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

# Add parent directory to Python path to make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from ohlcv_transport.config import config_manager
from ohlcv_transport.data import (
    get_complete_dataset, 
    clean_ohlcv, 
    denoise_ohlcv
)
from ohlcv_transport.models import (
    process_ohlcv_with_transport,
    calculate_imbalance_signal,
    RegimeManager,
    calculate_regime_statistic,
    detect_regime_changes
)
from ohlcv_transport.signals import (
    process_signal_pipeline,
    generate_trading_signal,
    calculate_trading_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("regime_example")


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
            # Use daily data for SPY and QQQ over a longer period to capture different regimes
            df_spy = get_complete_dataset(
                symbol="SPY",
                interval="1day",
                days_back=365,  # 1 year of data
                preprocess=True,
                use_cache=True
            )
            
            # Process with optimal transport
            df_spy = process_ohlcv_with_transport(df_spy)
            
            print(f"Retrieved {len(df_spy)} days of data for SPY")
            return df_spy
        except Exception as e:
            print(f"Error retrieving data: {e}")
            print("Falling back to synthetic data")
    
    # Option 2: Generate synthetic data with regime changes
    print("Generating synthetic sample data with regime changes...")
    
    # Date range (500 days of daily data)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='1d')
    
    # Create price series with different regimes
    np.random.seed(42)  # For reproducibility
    
    # Regime 1: Low volatility bull market (days 0-199)
    returns_regime1 = np.random.normal(0.0005, 0.005, 200)  # Mean positive return, low vol
    
    # Regime 2: Medium volatility sideways market (days 200-299)
    returns_regime2 = np.random.normal(0.0, 0.01, 100)  # Zero mean return, medium vol
    
    # Regime 3: High volatility bear market (days 300-399)
    returns_regime3 = np.random.normal(-0.001, 0.02, 100)  # Mean negative return, high vol
    
    # Regime 4: Recovery market (days 400-499)
    returns_regime4 = np.random.normal(0.001, 0.008, 100)  # Mean positive return, medium vol
    
    # Combine all regimes
    returns = np.concatenate([returns_regime1, returns_regime2, returns_regime3, returns_regime4])
    
    # Generate price series
    base_price = 100
    close = base_price * np.cumprod(1 + returns)
    
    # Generate realistic OHLC data
    volatility = np.concatenate([
        np.ones(200) * 0.005,  # Regime 1 volatility
        np.ones(100) * 0.01,   # Regime 2 volatility
        np.ones(100) * 0.02,   # Regime 3 volatility
        np.ones(100) * 0.008   # Regime 4 volatility
    ])
    
    # Generate high and low prices based on volatility
    high = close * (1 + np.random.uniform(0, 1, 500) * volatility)
    low = close * (1 - np.random.uniform(0, 1, 500) * volatility)
    
    # Generate open prices between previous close and current high/low
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # First day
    open_price = prev_close + np.random.uniform(-1, 1, 500) * (high - low) * 0.2
    
    # Generate volume (higher in high volatility regimes)
    base_volume = 1000000
    volume = base_volume * (1 + volatility * 50 * np.random.uniform(0.5, 1.5, 500))
    
    # Generate synthetic Wasserstein distances
    # Lower distance in low volatility regimes, higher in high volatility regimes
    w_distance = 0.5 + volatility * 50
    w_distance += np.random.normal(0, 0.1, 500)  # Add some noise
    
    # Create imbalance signal (normalize Wasserstein distances)
    # Use a rolling window of 20 days for normalization
    rolling_mean = pd.Series(w_distance).rolling(window=20, min_periods=1).mean()
    rolling_std = pd.Series(w_distance).rolling(window=20, min_periods=1).std()
    imbalance_signal = (w_distance - rolling_mean) / rolling_std
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'w_distance': w_distance,
        'imbalance_signal': imbalance_signal
    }, index=dates)
    
    print(f"Generated {len(df)} bars of synthetic data with multiple regimes")
    return df


def demonstrate_regime_detection(df):
    """
    Demonstrate regime detection using the RegimeManager.
    
    Args:
        df: DataFrame with OHLCV data and Wasserstein distances
        
    Returns:
        DataFrame with regime labels added
    """
    print("\n" + "="*50)
    print("REGIME DETECTION")
    print("="*50)
    
    # Check if required columns exist
    if 'w_distance' not in df.columns:
        print("Wasserstein distance column not found. Processing data with optimal transport...")
        df = process_ohlcv_with_transport(df)
    
    # Create RegimeManager with 3 regimes
    regime_manager = RegimeManager(num_regimes=3, lookback_window=50)
    
    # Detect regimes
    start_time = time.time()
    regime_series = regime_manager.detect_regime(df)
    detection_time = time.time() - start_time
    
    print(f"Detected regimes in {detection_time:.2f} seconds")
    
    # Add regime to DataFrame
    df['regime'] = regime_series
    
    # Print regime statistics
    regime_counts = df['regime'].value_counts()
    
    print("\nRegime distribution:")
    for regime, count in regime_counts.items():
        percent = count / len(df) * 100
        print(f"  Regime {regime}: {count} bars ({percent:.1f}%)")
    
    # Calculate regime statistics
    stats = regime_manager.get_regime_statistics()
    
    print("\nRegime transition matrix:")
    transition_matrix = np.array(stats['transition_matrix'])
    for i in range(len(transition_matrix)):
        probs = [f"{p:.2f}" for p in transition_matrix[i]]
        print(f"  From regime {i}: {' '.join(probs)}")
    
    if 'mean_durations' in stats:
        print("\nAverage regime duration (bars):")
        for regime, duration in stats['mean_durations'].items():
            print(f"  Regime {regime}: {duration:.1f}")
    
    # Plot regimes along with price
    if input("\nVisualize regimes? (y/n, default: y): ").strip().lower() != 'n':
        plt.figure(figsize=(15, 10))
        
        # Create a colormap for regimes
        cmap = plt.cm.get_cmap('viridis', len(regime_counts))
        
        # Plot price
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df.index, df['close'], 'k-', linewidth=1)
        ax1.set_ylabel('Price')
        ax1.set_title('Price Chart with Regime Classification')
        
        # Color background by regime
        for i, regime in enumerate(regime_counts.index):
            mask = (df['regime'] == regime)
            if not mask.any():
                continue
                
            # Find contiguous segments
            regime_changes = mask.astype(int).diff().fillna(0)
            start_idx = df.index[regime_changes == 1].tolist()
            end_idx = df.index[regime_changes == -1].tolist()
            
            # Handle edge cases
            if mask.iloc[0]:
                start_idx.insert(0, df.index[0])
            if mask.iloc[-1]:
                end_idx.append(df.index[-1])
            
            # Plot each segment
            for start, end in zip(start_idx, end_idx):
                ax1.axvspan(start, end, alpha=0.3, color=cmap(i), label=f"Regime {regime}" if start == start_idx[0] else "")
        
        ax1.legend()
        
        # Plot regime statistic
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        
        # Calculate regime statistic if needed
        if 'regime_stat' not in df.columns:
            df['regime_stat'] = calculate_regime_statistic(df, window=50)
        
        # Plot the statistic
        ax2.plot(df.index, df['regime_stat'], 'b-', label='Regime Statistic')
        
        # Add colored background for regimes
        for i, regime in enumerate(regime_counts.index):
            mask = (df['regime'] == regime)
            for start, end in zip(start_idx, end_idx):
                ax2.axvspan(start, end, alpha=0.3, color=cmap(i))
        
        ax2.set_ylabel('Regime Statistic')
        ax2.set_title('Regime Detection Statistic')
        
        plt.tight_layout()
        plt.show()
    
    return df


def demonstrate_regime_calibration(df):
    """
    Demonstrate parameter calibration for different regimes.
    
    Args:
        df: DataFrame with OHLCV data and regime labels
        
    Returns:
        RegimeManager with calibrated parameters
    """
    print("\n" + "="*50)
    print("REGIME-SPECIFIC PARAMETER CALIBRATION")
    print("="*50)
    
    # Check if regime column exists
    if 'regime' not in df.columns:
        print("Regime column not found. Running regime detection first...")
        df = demonstrate_regime_detection(df)
    
    # Create RegimeManager
    regime_manager = RegimeManager(num_regimes=3, lookback_window=50)
    
    # Print initial parameters
    print("\nInitial parameters for each regime:")
    for regime in range(regime_manager.num_regimes):
        params = regime_manager.get_regime_parameters(regime)
        print(f"\nRegime {regime}:")
        print(f"  Alpha (power law exponent): {params.alpha:.2f}")
        print(f"  Beta (signal predictiveness): {params.beta:.2f}")
        print(f"  Gamma (market impact): {params.gamma:.2f}")
        print(f"  Lambda (cost function weight): {params.lambda_param:.2f}")
        print(f"  Eta (regularization): {params.eta:.4f}")
        print(f"  Signal threshold: {params.signal_threshold:.2f}")
        print(f"  Min holding period: {params.min_holding_period}")
        print(f"  Max holding period: {params.max_holding_period}")
        print(f"  Position size factor: {params.position_size_factor:.2f}")
    
    # Calibrate parameters
    print("\nCalibrating parameters for each regime...")
    
    start_time = time.time()
    regime_manager.calibrate_parameters(df, df['regime'])
    calibration_time = time.time() - start_time
    
    print(f"Calibration completed in {calibration_time:.2f} seconds")
    
    # Print calibrated parameters and changes
    print("\nCalibrated parameters for each regime:")
    for regime in range(regime_manager.num_regimes):
        params = regime_manager.get_regime_parameters(regime)
        print(f"\nRegime {regime}:")
        print(f"  Alpha (power law exponent): {params.alpha:.2f}")
        print(f"  Beta (signal predictiveness): {params.beta:.2f}")
        print(f"  Gamma (market impact): {params.gamma:.2f}")
        print(f"  Lambda (cost function weight): {params.lambda_param:.2f}")
        print(f"  Eta (regularization): {params.eta:.4f}")
        print(f"  Signal threshold: {params.signal_threshold:.2f}")
        print(f"  Min holding period: {params.min_holding_period}")
        print(f"  Max holding period: {params.max_holding_period}")
        print(f"  Position size factor: {params.position_size_factor:.2f}")
    
    # Save calibrated parameters
    regime_manager.save_parameters()
    print("\nParameters saved to disk")
    
    return regime_manager


def apply_regime_specific_strategy(df, regime_manager):
    """
    Apply regime-specific parameters to the trading strategy.
    
    Args:
        df: DataFrame with OHLCV data and regime labels
        regime_manager: RegimeManager with calibrated parameters
        
    Returns:
        DataFrame with trading signals and performance metrics
    """
    print("\n" + "="*50)
    print("REGIME-ADAPTIVE TRADING STRATEGY")
    print("="*50)
    
    # Check if regime column exists
    if 'regime' not in df.columns:
        print("Regime column not found. Running regime detection first...")
        df = demonstrate_regime_detection(df)
    
    # Create a copy of the DataFrame for results
    result_df = df.copy()
    
    # Apply both regime-specific and static trading strategies for comparison
    
    # 1. Static strategy (no regime adaptation)
    static_params = regime_manager.get_regime_parameters(1)  # Use normal regime parameters
    
    print("\nApplying static trading strategy...")
    
    # Apply signal processing pipeline with static parameters
    if 'imbalance_signal' not in result_df.columns and 'w_distance' in result_df.columns:
        # Calculate imbalance signal if not already present
        result_df = calculate_imbalance_signal(result_df, result_df['w_distance'])
    
    # Generate trading signals with static parameters
    result_df = generate_trading_signal(
        result_df,
        signal_col='imbalance_signal',
        threshold_long=static_params.signal_threshold,
        threshold_short=-static_params.signal_threshold,
        position_col='static_position'
    )
    
    # 2. Regime-adaptive strategy
    print("\nApplying regime-adaptive trading strategy...")
    
    # Initialize adaptive position column
    result_df['adaptive_position'] = 0
    
    # Apply regime-specific parameters for each regime
    for regime in range(regime_manager.num_regimes):
        # Get regime-specific parameters
        params = regime_manager.get_regime_parameters(regime)
        
        # Apply to bars with this regime
        regime_mask = (result_df['regime'] == regime)
        
        # Skip if no bars with this regime
        if not regime_mask.any():
            continue
        
        # Generate trading signals with regime-specific parameters
        regime_df = generate_trading_signal(
            result_df[regime_mask],
            signal_col='imbalance_signal',
            threshold_long=params.signal_threshold,
            threshold_short=-params.signal_threshold,
            position_col='temp_position'
        )
        
        # Update adaptive position for this regime
        result_df.loc[regime_mask, 'adaptive_position'] = regime_df['temp_position']
    
    # Calculate returns and performance for both strategies
    if 'close' in result_df.columns:
        # Calculate price returns
        price_returns = result_df['close'].pct_change()
        
        # Calculate strategy returns
        result_df['static_returns'] = result_df['static_position'].shift(1) * price_returns
        result_df['adaptive_returns'] = result_df['adaptive_position'].shift(1) * price_returns
        
        # Calculate cumulative returns
        result_df['static_cumulative'] = (1 + result_df['static_returns']).cumprod()
        result_df['adaptive_cumulative'] = (1 + result_df['adaptive_returns']).cumprod()
        
        # Calculate regime-specific performance
        regime_performance = {}
        
        for regime in range(regime_manager.num_regimes):
            regime_mask = (result_df['regime'] == regime)
            
            # Skip if no bars with this regime
            if not regime_mask.any():
                continue
            
            # Calculate metrics for this regime
            static_metrics = calculate_trading_metrics(
                result_df[regime_mask], 
                position_col='static_position',
                price_col='close'
            )
            
            adaptive_metrics = calculate_trading_metrics(
                result_df[regime_mask], 
                position_col='adaptive_position',
                price_col='close'
            )
            
            regime_performance[regime] = {
                'static': static_metrics,
                'adaptive': adaptive_metrics
            }
        
        # Calculate overall performance
        static_metrics = calculate_trading_metrics(
            result_df, 
            position_col='static_position',
            price_col='close'
        )
        
        adaptive_metrics = calculate_trading_metrics(
            result_df, 
            position_col='adaptive_position',
            price_col='close'
        )
        
        # Print performance comparison
        print("\nPerformance by regime:")
        for regime, perf in regime_performance.items():
            print(f"\nRegime {regime}:")
            
            if perf['static']:
                static_return = perf['static'].get('total_return', 0) * 100
                static_sharpe = perf['static'].get('sharpe_ratio', 0)
                static_drawdown = perf['static'].get('max_drawdown', 0) * 100
                
                print(f"  Static strategy:")
                print(f"    Total return: {static_return:.2f}%")
                print(f"    Sharpe ratio: {static_sharpe:.2f}")
                print(f"    Max drawdown: {static_drawdown:.2f}%")
            
            if perf['adaptive']:
                adaptive_return = perf['adaptive'].get('total_return', 0) * 100
                adaptive_sharpe = perf['adaptive'].get('sharpe_ratio', 0)
                adaptive_drawdown = perf['adaptive'].get('max_drawdown', 0) * 100
                
                print(f"  Adaptive strategy:")
                print(f"    Total return: {adaptive_return:.2f}%")
                print(f"    Sharpe ratio: {adaptive_sharpe:.2f}")
                print(f"    Max drawdown: {adaptive_drawdown:.2f}%")
            
            # Calculate improvement
            if perf['static'] and perf['adaptive']:
                return_improvement = adaptive_return - static_return
                sharpe_improvement = adaptive_sharpe - static_sharpe
                drawdown_improvement = static_drawdown - adaptive_drawdown
                
                print(f"  Improvement:")
                print(f"    Return: {return_improvement:.2f}%")
                print(f"    Sharpe: {sharpe_improvement:.2f}")
                print(f"    Drawdown: {drawdown_improvement:.2f}%")
        
        print("\nOverall performance:")
        if static_metrics:
            static_total_return = static_metrics.get('total_return', 0) * 100
            static_sharpe = static_metrics.get('sharpe_ratio', 0)
            static_drawdown = static_metrics.get('max_drawdown', 0) * 100
            
            print(f"  Static strategy:")
            print(f"    Total return: {static_total_return:.2f}%")
            print(f"    Sharpe ratio: {static_sharpe:.2f}")
            print(f"    Max drawdown: {static_drawdown:.2f}%")
        
        if adaptive_metrics:
            adaptive_total_return = adaptive_metrics.get('total_return', 0) * 100
            adaptive_sharpe = adaptive_metrics.get('sharpe_ratio', 0)
            adaptive_drawdown = adaptive_metrics.get('max_drawdown', 0) * 100
            
            print(f"  Adaptive strategy:")
            print(f"    Total return: {adaptive_total_return:.2f}%")
            print(f"    Sharpe ratio: {adaptive_sharpe:.2f}")
            print(f"    Max drawdown: {adaptive_drawdown:.2f}%")
            
            # Calculate improvement
            return_improvement = adaptive_total_return - static_total_return
            sharpe_improvement = adaptive_sharpe - static_sharpe
            drawdown_improvement = static_drawdown - adaptive_drawdown
            
            print(f"  Improvement:")
            print(f"    Return: {return_improvement:.2f}%")
            print(f"    Sharpe: {sharpe_improvement:.2f}")
            print(f"    Drawdown: {drawdown_improvement:.2f}%")
        
        # Visualize performance
        if input("\nVisualize strategy performance? (y/n, default: y): ").strip().lower() != 'n':
            plt.figure(figsize=(15, 10))
            
            # Create a colormap for regimes
            regime_counts = result_df['regime'].value_counts()
            cmap = plt.cm.get_cmap('viridis', len(regime_counts))
            
            # Plot price and regimes
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(result_df.index, result_df['close'], 'k-', linewidth=1)
            ax1.set_ylabel('Price')
            ax1.set_title('Price Chart with Regime Classification')
            
            # Color background by regime
            for i, regime in enumerate(regime_counts.index):
                mask = (result_df['regime'] == regime)
                
                # Find contiguous segments
                regime_changes = mask.astype(int).diff().fillna(0)
                start_idx = result_df.index[regime_changes == 1].tolist()
                end_idx = result_df.index[regime_changes == -1].tolist()
                
                # Handle edge cases
                if mask.iloc[0]:
                    start_idx.insert(0, result_df.index[0])
                if mask.iloc[-1]:
                    end_idx.append(result_df.index[-1])
                
                # Plot each segment
                for start, end in zip(start_idx, end_idx):
                    ax1.axvspan(start, end, alpha=0.3, color=cmap(i), label=f"Regime {regime}" if start == start_idx[0] else "")
            
            ax1.legend()
            
            # Plot positions
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(result_df.index, result_df['static_position'], 'b-', alpha=0.7, label='Static')
            ax2.plot(result_df.index, result_df['adaptive_position'], 'r-', alpha=0.7, label='Adaptive')
            ax2.set_ylabel('Position')
            ax2.set_yticks([-1, 0, 1])
            ax2.set_title('Trading Positions: Static vs Adaptive')
            ax2.legend()
            
            # Plot cumulative returns
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(result_df.index, result_df['static_cumulative'], 'b-', label='Static')
            ax3.plot(result_df.index, result_df['adaptive_cumulative'], 'r-', label='Adaptive')
            ax3.set_ylabel('Cumulative Return')
            ax3.set_title('Performance Comparison: Static vs Adaptive')
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
    
    return result_df


def main():
    """Main function to run the demonstration."""
    print("\nRegime Detection and Calibration Demonstration")
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
    
    if df is None or df.empty:
        print("No data available. Exiting.")
        return
    
    # Menu loop
    while True:
        print("\nOptions:")
        print("1. Demonstrate regime detection")
        print("2. Demonstrate regime-specific parameter calibration")
        print("3. Apply regime-adaptive trading strategy")
        print("4. Run complete demonstration")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        regime_manager = None
        
        if choice == '1':
            df = demonstrate_regime_detection(df)
        elif choice == '2':
            regime_manager = demonstrate_regime_calibration(df)
        elif choice == '3':
            if regime_manager is None:
                regime_manager = RegimeManager()
            
            if 'regime' not in df.columns:
                print("Regime column not found. Running regime detection first...")
                df = demonstrate_regime_detection(df)
            
            apply_regime_specific_strategy(df, regime_manager)
        elif choice == '4':
            # Run complete demonstration
            df = demonstrate_regime_detection(df)
            regime_manager = demonstrate_regime_calibration(df)
            apply_regime_specific_strategy(df, regime_manager)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()