"""
Signal processing modules for the OHLCV Optimal Transport framework.

This package provides modules for generating, processing, and enhancing trading
signals based on the Wasserstein distances calculated from OHLCV data.
"""

from ohlcv_transport.signals.generators import (
    calculate_base_imbalance_signal,
    calculate_multi_timeframe_signal,
    add_signal_features,
    apply_kalman_filter,
    apply_wavelet_denoising,
    apply_exponential_smoothing,
    generate_trading_signal,
    apply_minimum_holding_period,
    calculate_trading_metrics,
    process_signal_pipeline
)

from ohlcv_transport.signals.processing import (
    normalize_signal,
    combine_signals,
    detect_signal_regime,
    adaptive_thresholds,
    apply_signal_filter,
    fractional_differentiation,
    detect_divergence,
    atr_bands,
    signal_to_position_sizing,
    calculate_dynamic_lookback,
    ewma_crossover_signal
)

__all__ = [
    # Signal generators
    'calculate_base_imbalance_signal',
    'calculate_multi_timeframe_signal',
    'add_signal_features',
    'apply_kalman_filter',
    'apply_wavelet_denoising',
    'apply_exponential_smoothing',
    'generate_trading_signal',
    'apply_minimum_holding_period',
    'calculate_trading_metrics',
    'process_signal_pipeline',
    
    # Signal processing
    'normalize_signal',
    'combine_signals',
    'detect_signal_regime',
    'adaptive_thresholds',
    'apply_signal_filter',
    'fractional_differentiation',
    'detect_divergence',
    'atr_bands',
    'signal_to_position_sizing',
    'calculate_dynamic_lookback',
    'ewma_crossover_signal'
]