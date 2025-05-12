"""
Example script demonstrating the OHLCV Optimal Transport framework.

This script shows how to:
1. Retrieve and preprocess OHLCV data
2. Create empirical measures from OHLCV data
3. Compute optimal transport distances
4. Calculate and visualize market imbalance signals
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
    denoise_ohlcv,
    calculate_derived_features
)
from ohlcv_transport.models import (
    create_empirical_measures,
    visualize_empirical_measures,
    compute_transport,
    compute_transport_batch,
    calculate_imbalance_signal,
    process_ohlcv_with_transport,
    visualize_transport_plan,
    visualize_imbalance_signal
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transport_example")


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


def get_sample_data():
    """Get sample data for demonstration purposes."""
    # Option 1: Use real data from API
    if config_manager.validate_api_key():
        print("Retrieving sample data from API...")
        try:
            df = get_complete_dataset(
                symbol="SPY",
                interval="1day",  # Use daily data for more reliability
                days_back=60,
                preprocess=True,
                use_cache=True
            )
            
            print(f"Retrieved {len(df)} bars for SPY")
            return df
        except Exception as e:
            print(f"Error retrieving data: {e}")
            print("Falling back to synthetic data")
    
    # Option 2: Generate synthetic data
    print("Generating synthetic sample data...")
    
    # Date range
    dates = pd.date_range(start='2024-04-01 09:30:00', periods=100, freq='1min')
    
    # Base price series with random walk
    np.random.seed(42)  # For reproducibility
    close = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
    
    # Add a trend
    close = close + np.linspace(0, 5, 100)
    
    # Create OHLC data with realistic relationships
    high = close + np.random.uniform(0, 0.5, 100)
    low = close - np.random.uniform(0, 0.5, 100)
    open_price = low + np.random.uniform(0, 1, 100) * (high - low)
    
    # Volume with some randomness
    volume = 1000 + np.random.uniform(0, 500, 100)
    
    # Add volume spikes at certain points
    volume[30:35] *= 3
    volume[70:75] *= 2
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Preprocess the data
    df = process_sample_data(df)
    
    print(f"Generated {len(df)} synthetic bars")
    return df


def process_sample_data(df):
    """Process the sample data."""
    print("Processing sample data...")
    
    # Clean data
    df = clean_ohlcv(df)
    
    # Denoise data
    df = denoise_ohlcv(df)
    
    # Calculate derived features
    df = calculate_derived_features(df)
    
    return df


def demonstrate_empirical_measures(df):
    """Demonstrate creating and visualizing empirical measures."""
    print("\n" + "="*50)
    print("EMPIRICAL MEASURES DEMONSTRATION")
    print("="*50)
    
    # Select a sample bar
    bar_idx = len(df) // 2  # Middle bar
    bar = df.iloc[bar_idx]
    
    print(f"\nSelected bar at {df.index[bar_idx]}:")
    print(f"  Open: {bar['open']:.2f}")
    print(f"  High: {bar['high']:.2f}")
    print(f"  Low: {bar['low']:.2f}")
    print(f"  Close: {bar['close']:.2f}")
    print(f"  Volume: {bar['volume']:.2f}")
    
    # Create empirical measures for this bar
    measures = create_empirical_measures(df.iloc[bar_idx:bar_idx+1])
    
    if not measures['supply'] or not measures['demand']:
        print("Failed to create measures for the selected bar.")
        return None, None
    
    supply_measure = measures['supply'][0]
    demand_measure = measures['demand'][0]
    
    print("\nMeasure statistics:")
    print(f"  Supply mean: {supply_measure.mean:.4f}")
    print(f"  Supply variance: {supply_measure.variance:.6f}")
    print(f"  Supply total mass: {supply_measure.total_mass:.2f}")
    print(f"  Demand mean: {demand_measure.mean:.4f}")
    print(f"  Demand variance: {demand_measure.variance:.6f}")
    print(f"  Demand total mass: {demand_measure.total_mass:.2f}")
    
    # Visualize the measures
    print("\nVisualizing empirical measures...")
    fig = visualize_empirical_measures(
        supply_measure, 
        demand_measure,
        title=f"Empirical Measures for {df.index[bar_idx]}"
    )
    
    return supply_measure, demand_measure


def demonstrate_transport_calculation(supply_measure, demand_measure):
    """Demonstrate optimal transport calculation."""
    if supply_measure is None or demand_measure is None:
        print("No measures available for transport calculation.")
        return
        
    print("\n" + "="*50)
    print("OPTIMAL TRANSPORT CALCULATION")
    print("="*50)
    
    # Compute transport with different regularization parameters
    reg_params = [0.001, 0.01, 0.1, 1.0]
    
    print("\nTransport calculation with different regularization parameters:")
    for reg_param in reg_params:
        start_time = time.time()
        result = compute_transport(
            supply_measure,
            demand_measure,
            reg_param=reg_param
        )
        elapsed = time.time() - start_time
        
        print(f"  reg_param={reg_param}:")
        print(f"    Distance: {result['distance']:.6f}")
        print(f"    Iterations: {result['iterations']}")
        print(f"    Converged: {result['converged']}")
        print(f"    Time: {result['time_ms']:.2f} ms (Python timer: {elapsed*1000:.2f} ms)")
    
    # Compute transport with plan for visualization
    print("\nComputing transport plan for visualization...")
    result_with_plan = compute_transport(
        supply_measure,
        demand_measure,
        reg_param=0.01,
        return_plan=True
    )
    
    # Visualize the transport plan
    print("Visualizing transport plan...")
    visualize_transport_plan(
        supply_measure,
        demand_measure,
        result_with_plan['transport_plan'],
        title="Optimal Transport Plan"
    )


def demonstrate_batch_transport(df):
    """Demonstrate batch transport calculation."""
    print("\n" + "="*50)
    print("BATCH TRANSPORT CALCULATION")
    print("="*50)
    
    # Process a subset of the data
    window_size = min(30, len(df))
    subset_df = df.iloc[:window_size]
    
    print(f"\nProcessing {window_size} bars...")
    
    # Create empirical measures
    measures = create_empirical_measures(subset_df)
    
    if not measures['supply'] or not measures['demand']:
        print("Failed to create measures.")
        return
    
    # Time batch transport calculation
    start_time = time.time()
    distances = compute_transport_batch(
        measures['supply'],
        measures['demand']
    )
    elapsed = time.time() - start_time
    
    print(f"Computed {len(distances)} distances in {elapsed:.2f} seconds")
    print(f"Average time per distance: {elapsed*1000/len(distances):.2f} ms")
    
    # Calculate imbalance signal
    signal = calculate_imbalance_signal(distances, lookback=10)
    
    # Create a DataFrame for visualization
    result_df = subset_df.copy()
    result_df['w_distance'] = distances
    result_df['imbalance_signal'] = signal
    
    # Visualize the signal
    print("\nVisualizing imbalance signal...")
    visualize_imbalance_signal(
        result_df,
        title="Market Imbalance Signal"
    )


def demonstrate_full_pipeline(df):
    """Demonstrate the full pipeline processing."""
    print("\n" + "="*50)
    print("FULL PIPELINE DEMONSTRATION")
    print("="*50)
    
    print(f"\nProcessing {len(df)} bars with the full pipeline...")
    
    # Time the full pipeline
    start_time = time.time()
    result_df = process_ohlcv_with_transport(
        df,
        num_price_points=100,
        alpha=1.0,
        lambda_param=0.5,
        reg_param=0.01,
        lookback=20
    )
    elapsed = time.time() - start_time
    
    print(f"Processed {len(df)} bars in {elapsed:.2f} seconds")
    print(f"Average time per bar: {elapsed*1000/len(df):.2f} ms")
    
    # Print some statistics
    if 'w_distance' in result_df.columns:
        print("\nWasserstein distance statistics:")
        print(f"  Mean: {result_df['w_distance'].mean():.6f}")
        print(f"  Std: {result_df['w_distance'].std():.6f}")
        print(f"  Min: {result_df['w_distance'].min():.6f}")
        print(f"  Max: {result_df['w_distance'].max():.6f}")
    
    if 'imbalance_signal' in result_df.columns:
        print("\nImbalance signal statistics:")
        print(f"  Mean: {result_df['imbalance_signal'].mean():.6f}")
        print(f"  Std: {result_df['imbalance_signal'].std():.6f}")
        print(f"  Min: {result_df['imbalance_signal'].min():.6f}")
        print(f"  Max: {result_df['imbalance_signal'].max():.6f}")
        
        # Count significant signals (|signal| > 1)
        sig_pos = np.sum(result_df['imbalance_signal'] > 1)
        sig_neg = np.sum(result_df['imbalance_signal'] < -1)
        print(f"  Significant positive signals: {sig_pos} ({sig_pos/len(result_df)*100:.1f}%)")
        print(f"  Significant negative signals: {sig_neg} ({sig_neg/len(result_df)*100:.1f}%)")
    
    # Visualize the full pipeline results
    print("\nVisualizing full pipeline results...")
    visualize_imbalance_signal(
        result_df,
        title="Full Pipeline Market Imbalance Signal"
    )
    
    # Return the results for further analysis
    return result_df


def analyze_signal_predictiveness(df):
    """Analyze the predictiveness of the imbalance signal."""
    if 'imbalance_signal' not in df.columns:
        print("No imbalance signal found in the data.")
        return
        
    print("\n" + "="*50)
    print("SIGNAL PREDICTIVENESS ANALYSIS")
    print("="*50)
    
    # Calculate forward returns
    print("\nCalculating forward returns...")
    for horizon in [1, 3, 5, 10]:
        df[f'fwd_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
    
    # Analyze correlation with forward returns
    print("\nCorrelation with forward returns:")
    for horizon in [1, 3, 5, 10]:
        corr = df['imbalance_signal'].corr(df[f'fwd_return_{horizon}'])
        print(f"  {horizon}-period forward return: {corr:.6f}")
    
    # Analyze performance by signal quintiles
    print("\nForward returns by signal quintiles:")
    
    # Create signal quintiles
    df['signal_quintile'] = pd.qcut(df['imbalance_signal'], 5, labels=False)
    
    # Calculate average return by quintile
    quintile_returns = {}
    for horizon in [1, 3, 5, 10]:
        quintile_returns[horizon] = df.groupby('signal_quintile')[f'fwd_return_{horizon}'].mean()
    
    # Print results
    for horizon in [1, 3, 5, 10]:
        print(f"\n  {horizon}-period forward returns by quintile:")
        for quintile, ret in quintile_returns[horizon].items():
            print(f"    Quintile {quintile+1}: {ret*100:.4f}%")
    
    # Visualize returns by signal strength
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 6))
        
        # Plot average return by signal bin
        df['signal_bin'] = pd.cut(df['imbalance_signal'], 10)
        horizon = 5  # Use 5-period forward return
        bin_returns = df.groupby('signal_bin')[f'fwd_return_{horizon}'].mean() * 100
        
        ax = bin_returns.plot(kind='bar', color='skyblue')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f"Average {horizon}-Period Forward Return by Signal Bin")
        plt.xlabel("Signal Bin")
        plt.ylabel(f"Average Return (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Simple trading strategy based on signal
    print("\nSimple trading strategy backtest:")
    
    # Define strategy: Long when signal > 1, Short when signal < -1
    df['position'] = 0
    df.loc[df['imbalance_signal'] > 1, 'position'] = 1
    df.loc[df['imbalance_signal'] < -1, 'position'] = -1
    
    # Calculate strategy returns
    df['strat_return'] = df['position'].shift(1) * df['close'].pct_change()
    
    # Calculate performance metrics
    total_return = df['strat_return'].sum() * 100
    sharpe = df['strat_return'].mean() / df['strat_return'].std() * np.sqrt(252)
    win_rate = np.sum(df['strat_return'] > 0) / np.sum(df['strat_return'] != 0) * 100
    
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Win Rate: {win_rate:.2f}%")
    
    # Visualize strategy performance
    try:
        # Calculate cumulative returns
        df['cum_return'] = (1 + df['strat_return']).cumprod() - 1
        df['cum_bh'] = (1 + df['close'].pct_change()).cumprod() - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['cum_return'] * 100, label='Strategy')
        plt.plot(df.index, df['cum_bh'] * 100, label='Buy & Hold')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title("Strategy Performance")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating performance visualization: {e}")


def main():
    """Main function to run the examples."""
    print("\nOHLCV Optimal Transport Framework Demonstration")
    print("="*50)
    
    # Setup API key if needed
    setup_api_key()
    
    # Get sample data
    df = get_sample_data()
    
    if df is None or df.empty:
        print("No data available. Exiting.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Demonstrate empirical measures")
        print("2. Demonstrate transport calculation")
        print("3. Demonstrate batch transport")
        print("4. Demonstrate full pipeline")
        print("5. Analyze signal predictiveness")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        supply_measure = None
        demand_measure = None
        result_df = None
        
        if choice == '1':
            supply_measure, demand_measure = demonstrate_empirical_measures(df)
        elif choice == '2':
            if supply_measure is None or demand_measure is None:
                supply_measure, demand_measure = demonstrate_empirical_measures(df)
            demonstrate_transport_calculation(supply_measure, demand_measure)
        elif choice == '3':
            demonstrate_batch_transport(df)
        elif choice == '4':
            result_df = demonstrate_full_pipeline(df)
        elif choice == '5':
            if result_df is None:
                result_df = demonstrate_full_pipeline(df)
            analyze_signal_predictiveness(result_df)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()