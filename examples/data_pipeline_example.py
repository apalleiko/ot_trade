"""
Example script demonstrating the OHLCV data acquisition and preprocessing pipeline.

This script shows how to:
1. Retrieve OHLCV data from the Twelve Data API
2. Preprocess the data for use in the optimal transport framework
3. Cache the data for faster subsequent access
4. Visualize the data and preprocessing results
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
    get_symbol_ohlcv,
    get_complete_dataset,
    clean_ohlcv,
    denoise_ohlcv,
    calculate_derived_features,
    dataset_manager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_pipeline_example")


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


def get_data_with_caching():
    """
    Demonstrate data retrieval with caching.
    """
    print("\n" + "="*50)
    print("DATA RETRIEVAL WITH CACHING")
    print("="*50)
    
    symbol = input("Enter a symbol (default: SPY): ").strip() or "SPY"
    interval = input("Enter an interval (default: 1min): ").strip() or "1min"
    days_back = int(input("Number of days to retrieve (default: 10): ").strip() or "10")
    
    # For intraday data, we need to be careful about weekends and market hours
    # Let's adjust the date range to ensure we get some data
    current_time = datetime.now()
    today_weekday = current_time.weekday()  # 0-6 (Monday to Sunday)
    
    # If today is Sunday (6), add 2 days to go back to Friday
    # If today is Saturday (5), add 1 day to go back to Friday
    if today_weekday == 6:  # Sunday
        days_back += 2
    elif today_weekday == 5:  # Saturday
        days_back += 1
    
    # If using 1min data, ensure we're requesting during market hours
    # US markets are typically open 9:30am-4:00pm ET
    if interval == '1min' or interval == '5min':
        # If current time is outside market hours, adjust end_date to last market close
        if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30) or current_time.hour > 16:
            # Use 4pm of the last trading day as the end time
            if today_weekday < 5:  # Weekday
                # Same day, but at 4pm
                end_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
                # If we're before market open, use previous day's close
                if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30):
                    if today_weekday == 0:  # Monday, go back to Friday
                        end_date = end_date - timedelta(days=3)
                    else:
                        end_date = end_date - timedelta(days=1)
            elif today_weekday == 5:  # Saturday
                # Use Friday at 4pm
                end_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=1)
            else:  # Sunday
                # Use Friday at 4pm
                end_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=2)
        else:
            end_date = current_time
    else:
        end_date = current_time
    
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\nRetrieving {interval} data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    # Time the first retrieval (from API)
    start_time = time.time()
    df = dataset_manager.get_dataset(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        preprocess=False,  # Don't preprocess yet
        force_refresh=True  # Force API call
    )
    api_time = time.time() - start_time
    
    if df is None or df.empty:
        print("No data retrieved. This could be due to the market being closed during the requested period.")
        print("Try a different symbol, a longer date range, or a different time interval.")
        return None
    
    print(f"Retrieved {len(df)} bars in {api_time:.2f} seconds")
    
    # Time the second retrieval (from cache)
    print("\nRetrieving the same data from cache...")
    start_time = time.time()
    cached_df = dataset_manager.get_dataset(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        preprocess=False
    )
    cache_time = time.time() - start_time
    
    if cached_df is not None and not cached_df.empty:
        print(f"Retrieved {len(cached_df)} bars in {cache_time:.2f} seconds")
        print(f"Cache is {api_time / cache_time:.1f}x faster")
    else:
        print("Failed to retrieve data from cache.")
    
    return df


def demonstrate_preprocessing(df):
    """
    Demonstrate data preprocessing steps.
    """
    if df is None or df.empty:
        print("No data to preprocess")
        return
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # 1. Data cleaning
    print("\nStep 1: Cleaning data...")
    start_time = time.time()
    cleaned_df = clean_ohlcv(df)
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    
    # 2. Wavelet denoising
    print("\nStep 2: Applying wavelet denoising...")
    start_time = time.time()
    denoised_df = denoise_ohlcv(cleaned_df)
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    
    # 3. Feature calculation
    print("\nStep 3: Calculating derived features...")
    start_time = time.time()
    features_df = calculate_derived_features(denoised_df)
    print(f"Completed in {time.time() - start_time:.2f} seconds")
    
    print(f"\nFeatures added: {', '.join(set(features_df.columns) - set(df.columns))}")
    
    return {
        'raw': df,
        'cleaned': cleaned_df,
        'denoised': denoised_df,
        'features': features_df
    }


def plot_preprocessing_results(data_dict):
    """
    Visualize the results of preprocessing.
    """
    if not data_dict:
        return
    
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    
    # Set the style
    sns.set(style="darkgrid")
    
    # 1. Compare raw and cleaned data
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Raw data with missing values highlighted
    raw_df = data_dict['raw']
    cleaned_df = data_dict['cleaned']
    
    # Plot close prices
    axes[0].plot(raw_df.index, raw_df['close'], label='Raw', alpha=0.7)
    axes[0].plot(cleaned_df.index, cleaned_df['close'], label='Cleaned', alpha=0.7)
    
    # Highlight missing values in raw data
    missing_mask = raw_df['close'].isna()
    if missing_mask.any():
        missing_idx = raw_df.index[missing_mask]
        axes[0].scatter(missing_idx, 
                      [raw_df['close'].median()] * len(missing_idx), 
                      color='red', marker='x', s=100, label='Missing')
    
    axes[0].set_title('Raw vs Cleaned Close Prices', fontsize=14)
    axes[0].legend()
    
    # Plot volume
    axes[1].bar(raw_df.index, raw_df['volume'], label='Raw', alpha=0.5)
    axes[1].bar(cleaned_df.index, cleaned_df['volume'], label='Cleaned', alpha=0.5)
    
    # Highlight zero or missing volume
    zero_mask = (raw_df['volume'] == 0) | raw_df['volume'].isna()
    if zero_mask.any():
        zero_idx = raw_df.index[zero_mask]
        axes[1].scatter(zero_idx, 
                      [raw_df['volume'].median()] * len(zero_idx), 
                      color='red', marker='x', s=100, label='Zero/Missing')
    
    axes[1].set_title('Raw vs Cleaned Volume', fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 2. Compare raw and denoised data
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get recent subset for better visualization
    subset_size = min(100, len(raw_df))
    raw_subset = raw_df['close'].iloc[-subset_size:]
    denoised_subset = data_dict['denoised']['close'].iloc[-subset_size:]
    
    ax.plot(raw_subset.index, raw_subset, label='Raw', alpha=0.7, linewidth=1)
    ax.plot(denoised_subset.index, denoised_subset, label='Denoised', alpha=0.9, linewidth=2)
    
    ax.set_title('Raw vs Denoised Close Prices (Recent Data)', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Visualize some derived features
    features_df = data_dict['features']
    
    # Select a few interesting features
    feature_cols = ['norm_range', 'volatility', 'rel_volume', 'norm_day_position']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(12, 3*len(feature_cols)), sharex=True)
    
    # Plot each feature
    for i, feature in enumerate(feature_cols):
        if feature in features_df.columns:
            axes[i].plot(features_df.index, features_df[feature], label=feature)
            axes[i].set_title(f'{feature}', fontsize=12)
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization completed. Close plots to continue.")


def demonstrate_complete_pipeline():
    """
    Demonstrate the complete data pipeline with a single function call.
    """
    print("\n" + "="*50)
    print("COMPLETE DATA PIPELINE")
    print("="*50)
    
    symbol = input("Enter a symbol for complete pipeline (default: AAPL): ").strip() or "AAPL"
    interval = input("Enter an interval (default: 1day): ").strip() or "1day"
    days_back = int(input("Number of days to retrieve (default: 30): ").strip() or "30")
    
    # Note: For intraday data like 1min, suggest using daily data for reliability
    if interval == '1min' or interval == '5min':
        print("\nNote: For intraday data, markets might be closed or data might be limited.")
        print("Consider using '1day' for more reliable results during testing.")
        if input("Do you want to continue with intraday data? (y/n, default: n): ").strip().lower() != 'y':
            interval = '1day'
            print(f"Switched to {interval} interval for better reliability.")
    
    print(f"\nRetrieving and processing {days_back} days of {interval} data for {symbol}...")
    
    # Time the complete pipeline
    start_time = time.time()
    df = get_complete_dataset(
        symbol=symbol,
        interval=interval,
        days_back=days_back,
        preprocess=True,
        use_cache=True
    )
    total_time = time.time() - start_time
    
    print(f"Complete pipeline finished in {total_time:.2f} seconds")
    print(f"Retrieved and processed {len(df)} bars")
    
    if not df.empty:
        print("\nDataFrame info:")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {', '.join(df.columns)}")
        print(f"- Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        print(f"- Date range: {df.index.min()} to {df.index.max()}")
        
        # Show some summary statistics
        print("\nSummary statistics for close prices:")
        print(df['close'].describe())
        
        # Visualize the result
        if input("\nVisualize the result? (y/n, default: y): ").strip().lower() != 'n':
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['close'])
            plt.title(f"{symbol} Close Prices ({interval})")
            plt.tight_layout()
            plt.show()
    
    return df


def demonstrate_multi_timeframe():
    """
    Demonstrate working with multiple timeframes.
    """
    print("\n" + "="*50)
    print("MULTI-TIMEFRAME DATA")
    print("="*50)
    
    from ohlcv_transport.data import get_multi_timeframe_data, resample_ohlcv
    
    symbol = input("Enter a symbol (default: QQQ): ").strip() or "QQQ"
    days_back = int(input("Number of days to retrieve (default: 10): ").strip() or "10")
    
    # Calculate a suitable date range (accounting for weekends and market hours)
    current_time = datetime.now()
    today_weekday = current_time.weekday()  # 0-6 (Monday to Sunday)
    
    # If today is Sunday (6), add 2 days to go back to Friday
    # If today is Saturday (5), add 1 day to go back to Friday
    if today_weekday == 6:  # Sunday
        days_back += 2
    elif today_weekday == 5:  # Saturday
        days_back += 1
    
    # Use 4pm of the last trading day as the end time if outside market hours
    if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30) or current_time.hour > 16:
        # Use 4pm of the last trading day as the end time
        if today_weekday < 5:  # Weekday
            # Same day, but at 4pm
            end_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            # If we're before market open, use previous day's close
            if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30):
                if today_weekday == 0:  # Monday, go back to Friday
                    end_date = end_date - timedelta(days=3)
                else:
                    end_date = end_date - timedelta(days=1)
        elif today_weekday == 5:  # Saturday
            # Use Friday at 4pm
            end_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=1)
        else:  # Sunday
            # Use Friday at 4pm
            end_date = current_time.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=2)
    else:
        end_date = current_time
    
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\nRetrieving multiple timeframes for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    # Get all available timeframes from config, making sure they're in the right format
    default_timeframes = ['1min', '5min', '15min', '1h']  # Changed '1hour' to '1h' to match API expected format
    timeframes = config_manager.get('assets.timeframes', default_timeframes)
    
    # Ensure timeframes are in the correct format expected by the Twelve Data API
    normalized_timeframes = []
    for tf in timeframes:
        # Convert '1hour' to '1h' if needed (Twelve Data API uses 'h' not 'hour')
        if 'hour' in tf:
            try:
                hours = int(tf.replace('hour', ''))
                normalized_tf = f"{hours}h"
            except ValueError:
                normalized_tf = '1h'  # Default to 1h if parsing fails
            normalized_timeframes.append(normalized_tf)
        else:
            normalized_timeframes.append(tf)
    
    timeframes = normalized_timeframes
    print(f"Timeframes to retrieve: {', '.join(timeframes)}")
    
    # Time the retrieval
    start_time = time.time()
    try:
        multi_tf_data = get_multi_timeframe_data(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date
        )
        retrieval_time = time.time() - start_time
        
        print(f"Retrieved multiple timeframes in {retrieval_time:.2f} seconds")
        
        # Check if we received any data
        if all(df.empty for df in multi_tf_data.values()):
            print("No data was retrieved for any timeframe. The market might be closed during the requested period.")
            print("Try a different symbol, a longer date range, or different time intervals.")
            return {}
        
        # Show data counts for each timeframe
        for tf in multi_tf_data:
            print(f"- {tf}: {len(multi_tf_data[tf])} bars")
            
        # Demonstrate resampling
        print("\nDemonstrating resampling from 1min to other timeframes...")
        if '1min' in multi_tf_data and not multi_tf_data['1min'].empty:
            base_df = multi_tf_data['1min']
            
            # Try resampling to different timeframes
            resample_timeframes = ['5min', '15min', '1h']
            for tf in resample_timeframes:
                start_time = time.time()
                resampled = resample_ohlcv(base_df, tf)
                resample_time = time.time() - start_time
                
                print(f"- Resampled to {tf}: {len(resampled)} bars in {resample_time:.2f} seconds")
                
                # Compare with API data if available
                api_tf = tf
                
                if api_tf in multi_tf_data:
                    api_df = multi_tf_data[api_tf]
                    overlap_dates = set(resampled.index) & set(api_df.index)
                    if overlap_dates:
                        # Calculate some comparison metrics for overlapping dates
                        resampled_subset = resampled.loc[overlap_dates]
                        api_subset = api_df.loc[overlap_dates]
                        
                        # Calculate mean absolute difference
                        close_diff = np.abs(resampled_subset['close'] - api_subset['close']).mean()
                        rel_diff = close_diff / api_subset['close'].mean() * 100
                        
                        print(f"  - Comparison with API data for {tf}:")
                        print(f"    - Overlapping bars: {len(overlap_dates)}")
                        print(f"    - Mean absolute difference in close: {close_diff:.4f} ({rel_diff:.2f}%)")
        
        return multi_tf_data
    
    except Exception as e:
        print(f"Error retrieving multiple timeframes: {e}")
        print("Please try different parameters or check your API configuration.")
        return {}
    
    # Show data counts for each timeframe
    for tf in multi_tf_data:
        print(f"- {tf}: {len(multi_tf_data[tf])} bars")
    
    # Demonstrate resampling
    print("\nDemonstrating resampling from 1min to other timeframes...")
    if '1min' in multi_tf_data and not multi_tf_data['1min'].empty:
        base_df = multi_tf_data['1min']
        
        # Try resampling to different timeframes
        resample_timeframes = ['5min', '15min', '1h']
        for tf in resample_timeframes:
            start_time = time.time()
            resampled = resample_ohlcv(base_df, tf)
            resample_time = time.time() - start_time
            
            print(f"- Resampled to {tf}: {len(resampled)} bars in {resample_time:.2f} seconds")
            
            # Compare with API data if available
            api_tf = tf
            if tf == '1h':
                api_tf = '1hour'  # Adjust for API naming
                
            if api_tf in multi_tf_data:
                api_df = multi_tf_data[api_tf]
                overlap_dates = set(resampled.index) & set(api_df.index)
                if overlap_dates:
                    # Calculate some comparison metrics for overlapping dates
                    resampled_subset = resampled.loc[overlap_dates]
                    api_subset = api_df.loc[overlap_dates]
                    
                    # Calculate mean absolute difference
                    close_diff = np.abs(resampled_subset['close'] - api_subset['close']).mean()
                    rel_diff = close_diff / api_subset['close'].mean() * 100
                    
                    print(f"  - Comparison with API data for {tf}:")
                    print(f"    - Overlapping bars: {len(overlap_dates)}")
                    print(f"    - Mean absolute difference in close: {close_diff:.4f} ({rel_diff:.2f}%)")
    
    return multi_tf_data


def main():
    """Main function to run the examples."""
    print("\nOHLCV Data Acquisition and Preprocessing Example")
    print("="*50)
    
    # Setup API key if needed
    setup_api_key()
    
    while True:
        print("\nOptions:")
        print("1. Demonstrate data retrieval with caching")
        print("2. Demonstrate data preprocessing steps")
        print("3. Demonstrate complete data pipeline")
        print("4. Demonstrate multi-timeframe data")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            df = get_data_with_caching()
        elif choice == '2':
            # First get some data
            if 'df' not in locals() or df is None or df.empty:
                print("First retrieving some data...")
                df = get_data_with_caching()
            
            # Then demonstrate preprocessing
            data_dict = demonstrate_preprocessing(df)
            
            # Visualize results
            if data_dict and input("\nVisualize preprocessing results? (y/n, default: y): ").strip().lower() != 'n':
                plot_preprocessing_results(data_dict)
        elif choice == '3':
            demonstrate_complete_pipeline()
        elif choice == '4':
            demonstrate_multi_timeframe()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()