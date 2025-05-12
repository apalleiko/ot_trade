"""
OHLCV Optimal Transport trading system

This package implements a quantitative trading framework based on optimal transport
theory, using OHLCV (Open, High, Low, Close, Volume) data to generate trading signals.

The main components are:
1. Data acquisition and preprocessing (ohlcv_transport.data)
2. Empirical measure construction (ohlcv_transport.models.empirical_measures)
3. Optimal transport calculation (ohlcv_transport.models.transport)
4. Signal generation and processing (ohlcv_transport.signals)
"""

__version__ = "0.2.0"
__author__ = "Quantitative Trader"

# Import key modules for easier access
from ohlcv_transport.data import (
    get_symbol_ohlcv,
    get_complete_dataset,
    clean_ohlcv,
    denoise_ohlcv,
    calculate_derived_features
)

from ohlcv_transport.models import (
    create_empirical_measures,
    compute_transport,
    process_ohlcv_with_transport,
    calculate_imbalance_signal
)

# Define public API
__all__ = [
    # Data functions
    'get_symbol_ohlcv',
    'get_complete_dataset',
    'clean_ohlcv',
    'denoise_ohlcv',
    'calculate_derived_features',
    
    # Model functions
    'create_empirical_measures',
    'compute_transport',
    'process_ohlcv_with_transport',
    'calculate_imbalance_signal'
]
