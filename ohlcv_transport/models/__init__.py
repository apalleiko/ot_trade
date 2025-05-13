"""
Trading models for the OHLCV Optimal Transport framework.

This package contains modules for constructing empirical measures from OHLCV data,
performing optimal transport calculations, and detecting market regimes.
"""

from ohlcv_transport.models.empirical_measures import (
    generate_price_grid,
    normal_density,
    calculate_supply_weights,
    calculate_demand_weights,
    EmpiricalMeasure,
    create_empirical_measures,
    calculate_measure_statistics,
    visualize_empirical_measures,
    visualize_measure_evolution
)

from ohlcv_transport.models.transport import (
    create_cost_matrix,
    sinkhorn_algorithm,
    wasserstein_distance,
    compute_transport,
    compute_transport_batch,
    calculate_imbalance_signal,
    process_ohlcv_with_transport,
    visualize_transport_plan,
    visualize_imbalance_signal
)

from ohlcv_transport.models.regime import (
    RegimeParameters,
    RegimeManager,
    calculate_regime_statistic,
    detect_regime_changes,
    get_regime_specific_parameters
)

__all__ = [
    # Empirical measures
    'generate_price_grid',
    'normal_density',
    'calculate_supply_weights',
    'calculate_demand_weights',
    'EmpiricalMeasure',
    'create_empirical_measures',
    'calculate_measure_statistics',
    'visualize_empirical_measures',
    'visualize_measure_evolution',
    
    # Optimal transport
    'create_cost_matrix',
    'sinkhorn_algorithm',
    'wasserstein_distance',
    'compute_transport',
    'compute_transport_batch',
    'calculate_imbalance_signal',
    'process_ohlcv_with_transport',
    'visualize_transport_plan',
    'visualize_imbalance_signal',
    
    # Regime detection
    'RegimeParameters',
    'RegimeManager',
    'calculate_regime_statistic',
    'detect_regime_changes',
    'get_regime_specific_parameters'
]