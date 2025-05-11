"""
Data handling modules for the OHLCV Optimal Transport trading system.

This package contains modules for retrieving, preprocessing, and caching OHLCV data.
"""

from ohlcv_transport.data.retrieval import (
    get_symbol_ohlcv,
    get_multi_timeframe_data,
    get_batch_symbols_data,
    download_historical_data,
    resample_ohlcv,
    get_complete_dataset
)

from ohlcv_transport.data.preprocessing import (
    clean_ohlcv,
    detect_outliers,
    replace_outliers,
    wavelet_denoise,
    denoise_ohlcv,
    calculate_derived_features,
    prepare_data_for_transport,
    process_ohlcv_data
)

from ohlcv_transport.data.cache import (
    OHLCVCache,
    DatasetManager,
    ohlcv_cache,
    dataset_manager
)

__all__ = [
    # Retrieval functions
    'get_symbol_ohlcv',
    'get_multi_timeframe_data',
    'get_batch_symbols_data',
    'download_historical_data',
    'resample_ohlcv',
    'get_complete_dataset',
    
    # Preprocessing functions
    'clean_ohlcv',
    'detect_outliers',
    'replace_outliers',
    'wavelet_denoise',
    'denoise_ohlcv',
    'calculate_derived_features',
    'prepare_data_for_transport',
    'process_ohlcv_data',
    
    # Cache classes and instances
    'OHLCVCache',
    'DatasetManager',
    'ohlcv_cache',
    'dataset_manager'
]