"""
Caching utilities for OHLCV data.

This module provides functions for caching and retrieving OHLCV data to/from disk,
allowing for faster data access without repeated API calls.
"""
import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
import pytz

from ohlcv_transport.config import config_manager

# Setup logging
logger = logging.getLogger(__name__)


class OHLCVCache:
    """Cache for OHLCV data."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the OHLCV cache.
        
        Args:
            cache_dir: Directory to store cache files (defaults to config value)
        """
        if cache_dir is None:
            self.cache_dir = config_manager.get_cache_dir() / 'ohlcv'
        else:
            self.cache_dir = Path(cache_dir)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default expiry time (can be overridden in get/set methods)
        self.default_expiry_hours = config_manager.get('system.data_cache_expiry_hours', 24)
        
        logger.debug(f"OHLCV cache initialized at {self.cache_dir}")
    
    def _generate_cache_key(self, symbol: str, interval: str, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> str:
        """
        Generate a unique cache key for the data request.
        
        Args:
            symbol: The trading symbol
            interval: Data interval
            start_date: Start date string
            end_date: End date string
        
        Returns:
            Cache key string
        """
        # Normalize inputs
        symbol = symbol.upper()
        interval = interval.lower()
        
        # Create a string with all parameters
        key_str = f"{symbol}_{interval}"
        if start_date:
            key_str += f"_from_{start_date}"
        if end_date:
            key_str += f"_to_{end_date}"
        
        # Hash for filename safety
        hash_obj = hashlib.md5(key_str.encode())
        return f"{symbol}_{interval}_{hash_obj.hexdigest()}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key string
        
        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """
        Get the metadata file path for a cache key.
        
        Args:
            cache_key: Cache key string
        
        Returns:
            Path to the metadata file
        """
        return self.cache_dir / f"{cache_key}_metadata.json"
    
    def save(self, df: pd.DataFrame, symbol: str, interval: str, 
           start_date: Optional[str] = None, end_date: Optional[str] = None,
           expiry_hours: Optional[float] = None) -> str:
        """
        Save OHLCV data to cache.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: The trading symbol
            interval: Data interval
            start_date: Start date string
            end_date: End date string
            expiry_hours: Number of hours before cache expires (None = default)
        
        Returns:
            Cache key string
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to cache.save")
            return ""
        
        # Generate cache key and paths
        cache_key = self._generate_cache_key(symbol, interval, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        try:
            # Store index frequency if it exists
            index_freq = None
            if hasattr(df.index, 'freq') and df.index.freq is not None:
                index_freq = df.index.freq.freqstr
            
            # Save the DataFrame to parquet format
            df.to_parquet(cache_path)
            
            # Create metadata
            metadata = {
                "symbol": symbol,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "num_bars": len(df),
                "timestamp": datetime.now().timestamp(),
                "expiry_hours": expiry_hours if expiry_hours is not None else self.default_expiry_hours,
                "index_freq": index_freq  # Save index frequency
            }
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Saved {len(df)} bars to cache: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return ""
    
    def load(self, symbol: str, interval: str, 
           start_date: Optional[str] = None, end_date: Optional[str] = None,
           check_expiry: bool = True) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from cache.
        
        Args:
            symbol: The trading symbol
            interval: Data interval
            start_date: Start date string
            end_date: End date string
            check_expiry: Whether to check if cache has expired
        
        Returns:
            DataFrame with OHLCV data or None if not found or expired
        """
        # Generate cache key and paths
        cache_key = self._generate_cache_key(symbol, interval, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        # Check if files exist
        if not cache_path.exists() or not metadata_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check expiry if requested
            if check_expiry:
                timestamp = metadata.get("timestamp", 0)
                expiry_hours = metadata.get("expiry_hours", self.default_expiry_hours)
                
                # Calculate age in hours
                age_hours = (datetime.now().timestamp() - timestamp) / 3600
                
                if age_hours > expiry_hours:
                    logger.debug(f"Cache expired: {cache_key} (age: {age_hours:.2f} hours)")
                    return None
            
            # Load the DataFrame
            df = pd.read_parquet(cache_path)
            
            # Restore index frequency if it was saved
            index_freq = metadata.get("index_freq")
            if index_freq and hasattr(df.index, 'freq'):
                try:
                    df.index.freq = pd.tseries.frequencies.to_offset(index_freq)
                except Exception as e:
                    logger.warning(f"Could not restore index frequency: {e}")
            
            logger.debug(f"Cache hit: {cache_key} ({len(df)} bars)")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def delete(self, symbol: str, interval: str, 
             start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """
        Delete cached OHLCV data.
        
        Args:
            symbol: The trading symbol
            interval: Data interval
            start_date: Start date string
            end_date: End date string
        
        Returns:
            True if successfully deleted, False otherwise
        """
        # Generate cache key and paths
        cache_key = self._generate_cache_key(symbol, interval, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)
        
        success = True
        
        # Delete cache file
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file: {e}")
                success = False
        
        # Delete metadata file
        if metadata_path.exists():
            try:
                metadata_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting metadata file: {e}")
                success = False
        
        if success:
            logger.debug(f"Deleted cache: {cache_key}")
        
        return success
    
    def clean_expired(self) -> int:
        """
        Clean expired cache entries.
        
        Returns:
            Number of entries deleted
        """
        logger.info("Cleaning expired cache entries")
        
        # Find all metadata files
        metadata_files = list(self.cache_dir.glob('*_metadata.json'))
        
        deleted_count = 0
        current_time = datetime.now().timestamp()
        
        for metadata_path in metadata_files:
            try:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check expiry
                timestamp = metadata.get("timestamp", 0)
                expiry_hours = metadata.get("expiry_hours", self.default_expiry_hours)
                
                # Calculate age in hours
                age_hours = (current_time - timestamp) / 3600
                
                if age_hours > expiry_hours:
                    # Extract cache key from metadata filename
                    cache_key = metadata_path.stem.replace('_metadata', '')
                    
                    # Delete the cache files
                    cache_path = self._get_cache_path(cache_key)
                    
                    # Delete both files explicitly
                    if cache_path.exists():
                        try:
                            cache_path.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting cache file {cache_path}: {e}")
                            continue
                    
                    if metadata_path.exists():
                        try:
                            metadata_path.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting metadata file {metadata_path}: {e}")
                            continue
                    
                    deleted_count += 1
                    logger.debug(f"Deleted expired cache: {cache_key}")
                    
            except Exception as e:
                logger.error(f"Error processing metadata file {metadata_path}: {e}")
        
        logger.info(f"Cleaned {deleted_count} expired cache entries")
        return deleted_count
    
def delete_by_key(self, cache_key: str) -> bool:
    """
    Delete cached data by cache key.
    
    Args:
        cache_key: Cache key string
    
    Returns:
        True if successfully deleted, False otherwise
    """
    cache_path = self._get_cache_path(cache_key)
    metadata_path = self._get_metadata_path(cache_key)
    
    success = True
    
    # Delete cache file
    if cache_path.exists():
        try:
            cache_path.unlink()
        except Exception as e:
            logger.error(f"Error deleting cache file: {e}")
            success = False
    
    # Delete metadata file
    if metadata_path.exists():
        try:
            metadata_path.unlink()
        except Exception as e:
            logger.error(f"Error deleting metadata file: {e}")
            success = False
    
    if success:
        logger.debug(f"Deleted cache: {cache_key}")
    
    return success
    
    def list_cache_entries(self) -> List[Dict[str, Any]]:
        """
        List all cache entries with their metadata.
        
        Returns:
            List of dictionaries with cache metadata
        """
        # Find all metadata files
        metadata_files = list(self.cache_dir.glob('*_metadata.json'))
        
        entries = []
        
        for metadata_path in metadata_files:
            try:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Calculate age in hours
                timestamp = metadata.get("timestamp", 0)
                age_hours = (datetime.now().timestamp() - timestamp) / 3600
                
                # Add age to metadata
                metadata["age_hours"] = age_hours
                
                # Calculate expiry status
                expiry_hours = metadata.get("expiry_hours", self.default_expiry_hours)
                metadata["expired"] = age_hours > expiry_hours
                
                # Extract cache key from metadata filename
                cache_key = metadata_path.stem.replace('_metadata', '')
                metadata["cache_key"] = cache_key
                
                entries.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing metadata file {metadata_path}: {e}")
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return entries


class DatasetManager:
    """
    Manager for handling cached datasets.
    
    This class combines the OHLCVCache with additional functionality
    for dataset management, including automatic updates and tracking.
    """
    
    def __init__(self):
        """Initialize the dataset manager."""
        self.cache = OHLCVCache()
        self.datasets = {}  # Track active datasets
    
    def get_dataset(self, symbol: str, interval: str,
                   start_date: Optional[Union[str, datetime]] = None,
                   end_date: Optional[Union[str, datetime]] = None,
                   preprocess: bool = True,
                   force_refresh: bool = False) -> pd.DataFrame:
        """
        Get a dataset, either from cache or by downloading.
        
        Args:
            symbol: The trading symbol
            interval: Data interval
            start_date: Start date
            end_date: End date
            preprocess: Whether to preprocess the data
            force_refresh: Whether to force refresh from API
        
        Returns:
            DataFrame with OHLCV data
        """
        # Convert datetime objects to strings
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Try to load from cache if not forcing refresh
        df = None
        if not force_refresh:
            df = self.cache.load(symbol, interval, start_date, end_date)
        
        # If not found in cache or forcing refresh, download
        if df is None or df.empty:
            # Import here to avoid circular imports
            from ohlcv_transport.data.retrieval import get_symbol_ohlcv
            
            logger.info(f"Downloading data for {symbol}, {interval}")
            df = get_symbol_ohlcv(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                use_cache=False  # Don't use API client's cache, we have our own
            )
            
            # Save to cache before preprocessing
            if not df.empty:
                self.cache.save(df, symbol, interval, start_date, end_date)
        
        # Preprocess if requested
        if preprocess and not df.empty:
            # Import here to avoid circular imports
            from ohlcv_transport.data.preprocessing import process_ohlcv_data
            
            logger.info(f"Preprocessing {len(df)} bars")
            df = process_ohlcv_data(df)
    
        # Track this dataset
        dataset_key = f"{symbol}_{interval}"
        self.datasets[dataset_key] = {
            "symbol": symbol,
            "interval": interval,
            "last_refresh": datetime.now(),
            "num_bars": len(df),
            "start_date": df.index.min() if not df.empty else None,
            "end_date": df.index.max() if not df.empty else None
        }
        
        return df
    
    def refresh_dataset(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Refresh a dataset with the most recent data.
        
        Args:
            symbol: The trading symbol
            interval: Data interval
        
        Returns:
            Updated DataFrame or None if the dataset is not tracked
        """
        dataset_key = f"{symbol}_{interval}"
        
        # Check if this dataset is tracked
        if dataset_key not in self.datasets:
            logger.warning(f"Dataset {dataset_key} is not tracked")
            return None
        
        # Get dataset info
        dataset_info = self.datasets[dataset_key]
        
        # Calculate the date range to update
        # Start from the last end date
        start_date = dataset_info.get("end_date")
        
        if start_date is None:
            logger.warning(f"Cannot refresh {dataset_key}: no end date")
            return None
        
        # Adjust start date slightly earlier to ensure overlap
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date - pd.Timedelta(hours=1)
        
        # Get the updated data
        return self.get_dataset(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            force_refresh=True
        )
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all tracked datasets.
        
        Returns:
            List of dataset information dictionaries
        """
        return list(self.datasets.values())


# Create singleton instances for easy import
ohlcv_cache = OHLCVCache()
dataset_manager = DatasetManager()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # List cache entries
    print("Cache entries:")
    for entry in ohlcv_cache.list_cache_entries():
        print(f"{entry['symbol']} {entry['interval']}: {entry['num_bars']} bars, "
              f"{entry['age_hours']:.2f} hours old, "
              f"{'expired' if entry['expired'] else 'valid'}")
    
    # Clean expired entries
    deleted = ohlcv_cache.clean_expired()
    print(f"Cleaned {deleted} expired entries")