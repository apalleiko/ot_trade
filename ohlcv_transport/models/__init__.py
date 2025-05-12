"""
Trading models for the OHLCV Optimal Transport framework.

This package contains modules for constructing empirical measures from OHLCV data
and performing optimal transport calculations.
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
    'visualize_imbalance_signal'
]