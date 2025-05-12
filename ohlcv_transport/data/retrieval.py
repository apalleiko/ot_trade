"""
Data retrieval utilities for OHLCV data.

This module provides functions for retrieving OHLCV data from the Twelve Data API,
handling pagination, rate limiting, and caching.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

from ohlcv_transport.api_client import twelve_data_client
from ohlcv_transport.config import config_manager

# Setup logging
logger = logging.getLogger(__name__)


def get_symbol_ohlcv(symbol: str, 
                    interval: str = '1min', 
                    start_date: Optional[Union[str, datetime]] = None,
                    end_date: Optional[Union[str, datetime]] = None,
                    limit: Optional[int] = None,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    Retrieve OHLCV data for a symbol from Twelve Data API.
    
    Args:
        symbol: The trading symbol (e.g., 'AAPL')
        interval: Data interval ('1min', '5min', '15min', '1h', '1day', etc.)
        start_date: Start date in format 'YYYY-MM-DD' or datetime object
        end_date: End date in same format (defaults to current time)
        limit: Maximum number of data points to retrieve (overrides start_date if provided)
        use_cache: Whether to use cached data if available
    
    Returns:
        DataFrame with OHLCV data
    """
    # Handle date parameters
    if start_date is not None and isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
    
    if end_date is not None and isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Log request details
    log_msg = f"Retrieving {interval} data for {symbol}"
    if limit is not None:
        log_msg += f", limit={limit}"
    if start_date is not None:
        log_msg += f", start={start_date}"
    if end_date is not None:
        log_msg += f", end={end_date}"
    logger.info(log_msg)
    
    try:
        # Call the API client to retrieve the data
        df = twelve_data_client.get_time_series(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            outputsize=limit if limit is not None else 5000,  # Default to 5000 if limit not specified
            use_cache=use_cache
        )
        
        # Sort by datetime just to be sure
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        
        logger.info(f"Retrieved {len(df)} {interval} bars for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving data for {symbol}: {e}")
        # Return empty DataFrame on error
        return pd.DataFrame()


def get_multi_timeframe_data(symbol: str,
                            timeframes: List[str] = None,
                            start_date: Optional[Union[str, datetime]] = None,
                            end_date: Optional[Union[str, datetime]] = None,
                            use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Retrieve OHLCV data for multiple timeframes for a symbol.
    
    Args:
        symbol: The trading symbol (e.g., 'AAPL')
        timeframes: List of timeframes to retrieve (e.g., ['1min', '5min', '15min'])
        start_date: Start date in format 'YYYY-MM-DD' or datetime object
        end_date: End date in same format (defaults to current time)
        use_cache: Whether to use cached data if available
    
    Returns:
        Dictionary mapping timeframes to DataFrames with OHLCV data
    """
    if timeframes is None:
        timeframes = config_manager.get('assets.timeframes', ['1min', '5min', '15min', '1hour'])
    
    # Initialize result dictionary
    result = {}
    
    # Retrieve data for each timeframe
    for tf in timeframes:
        logger.info(f"Retrieving {tf} data for {symbol}")
        
        # Calculate appropriate bar limit for the timeframe
        # The API has a maximum limit of 5000 bars
        limit = calculate_appropriate_limit(tf, start_date, end_date)
        
        # Retrieve the data
        df = get_symbol_ohlcv(
            symbol=symbol,
            interval=tf,
            start_date=start_date,
            end_date=end_date,
            limit=min(limit, 5000),  # Ensure we don't exceed API limit
            use_cache=use_cache
        )
        
        # Add to result
        result[tf] = df
    
    return result


def get_batch_symbols_data(symbols: List[str],
                          interval: str = '1min',
                          start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None,
                          use_cache: bool = True,
                          delay_between_requests: float = 0.1) -> Dict[str, pd.DataFrame]:
    """
    Retrieve OHLCV data for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        interval: Data interval
        start_date: Start date
        end_date: End date
        use_cache: Whether to use cached data if available
        delay_between_requests: Time to wait between API requests (seconds)
    
    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV data
    """
    # Initialize result dictionary
    result = {}
    
    # Retrieve data for each symbol
    for symbol in symbols:
        logger.info(f"Retrieving {interval} data for {symbol}")
        
        # Retrieve the data
        df = get_symbol_ohlcv(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        # Add to result
        result[symbol] = df
        
        # Add delay between requests to be nice to the API
        if delay_between_requests > 0 and symbol != symbols[-1]:
            time.sleep(delay_between_requests)
    
    return result


def download_historical_data(symbol: str,
                            interval: str = '1min',
                            days_back: int = 10,
                            use_cache: bool = True,
                            max_requests: int = 10) -> pd.DataFrame:
    """
    Download historical data for a longer period by making multiple requests.
    
    Args:
        symbol: The trading symbol
        interval: Data interval
        days_back: Number of days to go back
        use_cache: Whether to use cached data if available
        max_requests: Maximum number of API requests to make
    
    Returns:
        DataFrame with concatenated historical OHLCV data
    """
    # Calculate end date (current time)
    end_date = datetime.now()
    
    # Initialize result DataFrame
    all_data = pd.DataFrame()
    
    # Estimate how many days we can retrieve per request based on interval
    days_per_request = estimate_days_per_request(interval)
    
    # Calculate number of requests needed
    num_requests = min(max_requests, int(np.ceil(days_back / days_per_request)))
    
    # Make requests
    for i in range(num_requests):
        # Calculate start and end dates for this request
        req_end_date = end_date - timedelta(days=i * days_per_request)
        req_start_date = req_end_date - timedelta(days=days_per_request)
        
        # Adjust the start date for the last request to cover the full period
        if i == num_requests - 1:
            req_start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Request {i+1}/{num_requests}: {req_start_date.date()} to {req_end_date.date()}")
        
        # Retrieve data
        df = get_symbol_ohlcv(
            symbol=symbol,
            interval=interval,
            start_date=req_start_date,
            end_date=req_end_date,
            use_cache=use_cache
        )
        
        # Add to result
        if not df.empty:
            all_data = pd.concat([all_data, df])
        
        # Add a short delay between requests
        if i < num_requests - 1:
            time.sleep(0.5)
    
    # Remove duplicates and sort
    if not all_data.empty:
        all_data = all_data[~all_data.index.duplicated(keep='first')]
        all_data = all_data.sort_index()
    
    logger.info(f"Retrieved a total of {len(all_data)} bars over {days_back} days")
    return all_data


def calculate_appropriate_limit(interval: str, 
                               start_date: Optional[Union[str, datetime]] = None,
                               end_date: Optional[Union[str, datetime]] = None) -> int:
    """
    Calculate an appropriate limit for the number of bars based on timeframe and date range.
    
    Args:
        interval: Data interval
        start_date: Start date
        end_date: End date
    
    Returns:
        Estimated number of bars needed
    """
    # Default to maximum API limit
    default_limit = 5000
    
    # If either date is not provided, return default
    if start_date is None or end_date is None:
        return default_limit
    
    # Convert to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate date range in seconds
    date_range_seconds = (end_date - start_date).total_seconds()
    
    # Map interval to seconds
    interval_seconds = interval_to_seconds(interval)
    
    # Calculate number of bars
    if interval_seconds > 0:
        estimated_bars = int(date_range_seconds / interval_seconds)
        # Add 10% margin
        estimated_bars = int(estimated_bars * 1.1)
        return min(estimated_bars, default_limit)
    else:
        return default_limit


def interval_to_seconds(interval: str) -> int:
    """
    Convert an interval string to seconds.
    
    Args:
        interval: Interval string (e.g., '1min', '5min', '1h', '1hour', '1day')
    
    Returns:
        Number of seconds in the interval
    """
    # Extract number and unit
    if not interval:
        return 60  # Default to 1 minute
    
    # Handle different formats
    if 'min' in interval:
        try:
            minutes = int(interval.replace('min', ''))
            return minutes * 60
        except ValueError:
            logger.warning(f"Invalid minute format: {interval}, defaulting to 1 minute")
            return 60
    elif 'hour' in interval:
        try:
            hours = int(interval.replace('hour', ''))
            return hours * 3600
        except ValueError:
            logger.warning(f"Invalid hour format: {interval}, defaulting to 1 hour")
            return 3600
    elif 'day' in interval:
        try:
            days = int(interval.replace('day', ''))
            return days * 86400
        except ValueError:
            logger.warning(f"Invalid day format: {interval}, defaulting to 1 day")
            return 86400
    elif 'week' in interval:
        try:
            weeks = int(interval.replace('week', ''))
            return weeks * 7 * 86400
        except ValueError:
            logger.warning(f"Invalid week format: {interval}, defaulting to 1 week")
            return 7 * 86400
    elif 'month' in interval:
        try:
            months = int(interval.replace('month', ''))
            # Approximate months as 30 days
            return months * 30 * 86400
        except ValueError:
            logger.warning(f"Invalid month format: {interval}, defaulting to 1 month")
            return 30 * 86400
    elif 'h' in interval:
        try:
            hours = int(interval.replace('h', ''))
            return hours * 3600
        except ValueError:
            logger.warning(f"Invalid hour format: {interval}, defaulting to 1 hour")
            return 3600
    else:
        logger.warning(f"Unknown interval format: {interval}, defaulting to 1 minute")
        return 60  # Default to 1 minute


def estimate_days_per_request(interval: str) -> float:
    """
    Estimate how many days of data can be retrieved in a single request.
    
    Args:
        interval: Data interval
    
    Returns:
        Estimated number of days per request
    """
    # Map intervals to estimated days that would give 5000 bars
    interval_to_days = {
        '1min': 4,       # ~5760 minutes in 4 days
        '5min': 18,      # 5000 * 5 minutes = 25000 minutes = ~17.4 days
        '15min': 50,     # 5000 * 15 minutes = 75000 minutes = ~52.1 days
        '30min': 105,    # 5000 * 30 minutes = 150000 minutes = ~104.2 days
        '1h': 210,       # 5000 hours = ~208.3 days
        '4h': 830,       # 5000 * 4 hours = 20000 hours = ~833.3 days
        '1day': 5000,    # 5000 days = ~13.7 years
    }
    
    # Get the closest matching interval
    for key in interval_to_days:
        if interval.lower() == key.lower():
            return interval_to_days[key]
    
    # Default to a conservative estimate for unknown intervals
    logger.warning(f"Unknown interval format for days estimation: {interval}")
    return 3  # Default to 3 days (conservative)


def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        target_interval: Target interval to resample to (e.g., '5min', '1h')
    
    Returns:
        Resampled DataFrame
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to resample_ohlcv")
        return df
    
    # Ensure the dataframe has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex, cannot resample")
        return df
    
    # Map interval string to pandas frequency string
    freq = interval_to_pandas_freq(target_interval)
    
    if freq is None:
        logger.error(f"Invalid target interval: {target_interval}")
        return df
    
    logger.info(f"Resampling data from {df.index[1] - df.index[0]} to {freq}")
    
    # Resample using pandas
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Drop rows with NaN values (can happen at the edges of the resampling)
    resampled = resampled.dropna()
    
    logger.info(f"Resampled data from {len(df)} to {len(resampled)} bars")
    return resampled


def interval_to_pandas_freq(interval: str) -> Optional[str]:
    """
    Convert an interval string to a pandas frequency string.
    
    Args:
        interval: Interval string (e.g., '1min', '5min', '1h', '1hour', '1day')
    
    Returns:
        Pandas frequency string or None if invalid
    """
    if not interval:
        return None
    
    # Handle different formats
    if 'min' in interval:
        try:
            minutes = int(interval.replace('min', ''))
            return f"{minutes}T"
        except ValueError:
            logger.warning(f"Invalid minute format for pandas frequency: {interval}")
            return None
    elif 'hour' in interval:
        try:
            hours = int(interval.replace('hour', ''))
            return f"{hours}H"
        except ValueError:
            logger.warning(f"Invalid hour format for pandas frequency: {interval}")
            return None
    elif 'day' in interval:
        try:
            days = int(interval.replace('day', ''))
            return f"{days}D"
        except ValueError:
            logger.warning(f"Invalid day format for pandas frequency: {interval}")
            return None
    elif 'week' in interval:
        try:
            weeks = int(interval.replace('week', ''))
            return f"{weeks}W"
        except ValueError:
            logger.warning(f"Invalid week format for pandas frequency: {interval}")
            return None
    elif 'month' in interval:
        try:
            months = int(interval.replace('month', ''))
            return f"{months}M"
        except ValueError:
            logger.warning(f"Invalid month format for pandas frequency: {interval}")
            return None
    elif 'h' in interval:
        try:
            hours = int(interval.replace('h', ''))
            return f"{hours}H"
        except ValueError:
            logger.warning(f"Invalid hour format for pandas frequency: {interval}")
            return None
    else:
        logger.warning(f"Unknown interval format for pandas frequency: {interval}")
        return None


def get_complete_dataset(symbol: str, 
                        interval: str = '1min',
                        days_back: int = 30,
                        preprocess: bool = True,
                        use_cache: bool = True) -> pd.DataFrame:
    """
    Get a complete processed dataset for a symbol.
    
    This function combines data retrieval and preprocessing in a single call.
    
    Args:
        symbol: The trading symbol
        interval: Data interval
        days_back: Number of days to go back
        preprocess: Whether to preprocess the data
        use_cache: Whether to use cached data
    
    Returns:
        Processed DataFrame
    """
    # Import here to avoid circular imports
    from ohlcv_transport.data.preprocessing import process_ohlcv_data
    
    # Download the data
    logger.info(f"Getting complete dataset for {symbol}, {interval}, {days_back} days back")
    df = download_historical_data(
        symbol=symbol,
        interval=interval,
        days_back=days_back,
        use_cache=use_cache
    )
    
    if df.empty:
        logger.warning(f"No data retrieved for {symbol}")
        return df
    
    # Preprocess the data if requested
    if preprocess:
        logger.info(f"Preprocessing {len(df)} bars")
        df = process_ohlcv_data(df)
    
    return df