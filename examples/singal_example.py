"""
Example script demonstrating the signal generation and processing functionality.

This script shows how to:
1. Generate signals from OHLCV data using optimal transport
2. Process and enhance signals for better trading decisions
3. Apply multi-timeframe signal combination
4. Evaluate trading performance
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
    get_multi_timeframe_data
)
from ohlcv_transport.models import (
    process_ohlcv_with_transport
)
from ohlcv_transport.signals import (
    calculate_base_imbalance_signal,
    calculate_multi_timeframe_signal,
    add_signal_features,
    apply_kalman_filter,
    apply_wavelet_denoising,
    apply_exponential_smoothing,
    generate_trading_signal,
    apply_minimum_holding_period,
    calculate_trading_metrics,
    normalize_signal,
    combine_signals,
    detect_signal_regime,
    adaptive_thresholds,
    detect_divergence,
    signal_to_position_sizing,
    calculate_dynamic_lookback,
    ewma_crossover_signal
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("signal_example")


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