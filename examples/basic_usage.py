"""
Basic usage example for the OHLCV Optimal Transport trading system.

This script demonstrates how to use the system to:
1. Configure API access
2. Retrieve OHLCV data
3. Visualize the data
"""
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to Python path to make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from ohlcv_transport.config import config_manager
from ohlcv_transport.api_client import twelve_data_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("basic_usage")


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


def get_ohlcv_data(symbol='SPY', interval='1min', limit=100):
    """
    Retrieve OHLCV data for the given symbol and interval.
    
    Args:
        symbol: Trading symbol (default: 'SPY')
        interval: Data interval (default: '1min')
        limit: Number of data points to retrieve (default: 100)
    
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Retrieving {limit} {interval} bars for {symbol}...")
    
    try:
        df = twelve_data_client.get_time_series(
            symbol=symbol,
            interval=interval,
            outputsize=limit
        )
        
        logger.info(f"Retrieved {len(df)} bars.")
        return df
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        return None


def plot_ohlcv(df, symbol):
    """
    Plot OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol for the title
    """
    if df is None or df.empty:
        logger.error("No data to plot.")
        return
    
    # Set the style
    sns.set(style="darkgrid")
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price data on the first subplot
    ax1.plot(df.index, df['close'], label='Close', linewidth=2)
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.2, label='Range')
    
    # Add title and labels
    ax1.set_title(f'{symbol} Price Chart', fontsize=16)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    
    # Plot volume on the second subplot
    ax2.bar(df.index, df['volume'], width=0.8, alpha=0.6, label='Volume')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Format the x-axis to show readable dates
    fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def get_current_price(symbol='SPY'):
    """
    Get the current price for a symbol.
    
    Args:
        symbol: Trading symbol (default: 'SPY')
    
    Returns:
        Current price as float
    """
    try:
        price = twelve_data_client.get_price(symbol, use_cache=False)
        logger.info(f"Current price of {symbol}: ${price}")
        return price
    except Exception as e:
        logger.error(f"Error retrieving price: {e}")
        return None


def main():
    """Main function."""
    print("\nOHLCV Optimal Transport Trading System - Basic Usage Example")
    print("="*60 + "\n")
    
    # Setup API key if needed
    setup_api_key()
    
    # Test the API connection
    success, message = twelve_data_client.test_connection()
    print(f"API connection test: {message}")
    
    if not success:
        print("API connection failed. Please check your API key and internet connection.")
        sys.exit(1)
    
    # Get current price for a symbol
    symbol = input("\nEnter a symbol (default: SPY): ").strip() or "SPY"
    price = get_current_price(symbol)
    
    if price is None:
        print(f"Could not retrieve price for {symbol}. Please check the symbol and try again.")
        sys.exit(1)
    
    # Get OHLCV data
    print("\nRetrieving OHLCV data...")
    intervals = ['1min', '5min', '15min', '1h', '1day']
    
    print("\nAvailable intervals:")
    for i, interval in enumerate(intervals, 1):
        print(f"{i}. {interval}")
    
    interval_choice = input("\nSelect interval (1-5, default: 1): ").strip() or "1"
    try:
        interval_index = int(interval_choice) - 1
        if interval_index < 0 or interval_index >= len(intervals):
            raise ValueError("Invalid choice")
        interval = intervals[interval_index]
    except ValueError:
        print("Invalid choice. Using default interval (1min).")
        interval = '1min'
    
    limit = input("\nNumber of data points to retrieve (default: 100): ").strip() or "100"
    try:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("Number must be positive")
    except ValueError:
        print("Invalid number. Using default (100).")
        limit = 100
    
    df = get_ohlcv_data(symbol, interval, limit)
    
    if df is not None and not df.empty:
        print("\nOHLCV data summary:")
        print(f"Start date: {df.index.min()}")
        print(f"End date: {df.index.max()}")
        print(f"Number of bars: {len(df)}")
        print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"Average volume: {df['volume'].mean():.0f}")
        
        # Print the first few rows
        print("\nMost recent data:")
        print(df.head())
        
        # Plot the data
        plot_choice = input("\nDo you want to plot the data? (y/n, default: y): ").strip().lower() or "y"
        if plot_choice == "y":
            plot_ohlcv(df, symbol)
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()