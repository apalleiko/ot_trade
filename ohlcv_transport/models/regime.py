"""
Regime detection and calibration for the OHLCV Optimal Transport framework.

This module provides functions for identifying market regimes and adapting
the model parameters based on the detected regime.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ohlcv_transport.config import config_manager

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RegimeParameters:
    """Class for storing regime-specific parameters."""
    
    # General parameters
    alpha: float = 1.0  # Power law exponent for weighting functions
    beta: float = 0.5   # Signal predictiveness factor
    gamma: float = 1.0  # Market impact factor
    lambda_param: float = 0.5  # Weight for normalized price component in cost
    eta: float = 0.01  # Regularization parameter for Sinkhorn algorithm
    
    # Trading parameters
    signal_threshold: float = 1.5  # Threshold for trade entry
    min_holding_period: int = 5  # Minimum holding period in bars
    max_holding_period: int = 180  # Maximum holding period in bars
    position_size_factor: float = 1.0  # Scaling for position sizing
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for serialization."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lambda_param": self.lambda_param,
            "eta": self.eta,
            "signal_threshold": self.signal_threshold,
            "min_holding_period": self.min_holding_period,
            "max_holding_period": self.max_holding_period,
            "position_size_factor": self.position_size_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeParameters':
        """Create parameters from dictionary."""
        return cls(
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 0.5),
            gamma=data.get("gamma", 1.0),
            lambda_param=data.get("lambda_param", 0.5),
            eta=data.get("eta", 0.01),
            signal_threshold=data.get("signal_threshold", 1.5),
            min_holding_period=data.get("min_holding_period", 5),
            max_holding_period=data.get("max_holding_period", 180),
            position_size_factor=data.get("position_size_factor", 1.0)
        )


class RegimeManager:
    """
    Manager for detecting market regimes and calibrating parameters.
    
    This class encapsulates the logic for regime detection, classification,
    and parameter management across different market regimes.
    """
    
    def __init__(self, num_regimes: int = 3, lookback_window: int = 50):
        """
        Initialize the regime manager.
        
        Args:
            num_regimes: Number of regimes to detect
            lookback_window: Window size for regime statistics
        """
        self.num_regimes = num_regimes
        self.lookback_window = lookback_window
        
        # Initialize regime parameters with default values
        self.regime_params = {
            0: RegimeParameters(  # Low volatility regime
                alpha=0.8,
                beta=0.4,
                gamma=0.8,
                lambda_param=0.4,
                eta=0.02,
                signal_threshold=1.2,
                min_holding_period=8,
                max_holding_period=240,
                position_size_factor=0.8
            ),
            1: RegimeParameters(  # Normal regime (baseline)
                alpha=1.0,
                beta=0.5,
                gamma=1.0,
                lambda_param=0.5,
                eta=0.01,
                signal_threshold=1.5,
                min_holding_period=5,
                max_holding_period=180,
                position_size_factor=1.0
            ),
            2: RegimeParameters(  # High volatility regime
                alpha=1.2,
                beta=0.6,
                gamma=1.5,
                lambda_param=0.6,
                eta=0.005,
                signal_threshold=2.0,
                min_holding_period=3,
                max_holding_period=120,
                position_size_factor=0.6
            )
        }
        
        # Initialize regime transition matrix
        self.transition_matrix = np.ones((num_regimes, num_regimes)) / num_regimes
        
        # Initialize regime history
        self.regime_history = []
        
        # Initialize clustering model
        self.cluster_model = None
        self.feature_scaler = StandardScaler()
        
        # Initialize calibration status
        self.is_calibrated = False
        
        # Load saved parameters if available
        try:
            self._load_saved_parameters()
        except Exception as e:
            logger.warning(f"Failed to load saved parameters: {e}")
            # Keep default parameter values
    
    def _load_saved_parameters(self) -> None:
        """Load saved regime parameters if available."""
        try:
            params_dir = config_manager.get_cache_dir() / 'regime_params'
            params_file = params_dir / 'regime_parameters.json'
            
            if not params_file.exists():
                logger.debug("No saved regime parameters found")
                return
                
            with open(params_file, 'r') as f:
                data = json.load(f)
            
            # Load regime parameters
            if 'regime_params' in data:
                for regime_id_str, params_dict in data['regime_params'].items():
                    try:
                        regime_id = int(regime_id_str)  # Convert string key to int
                        if regime_id in self.regime_params:
                            self.regime_params[regime_id] = RegimeParameters.from_dict(params_dict)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid regime ID in saved parameters: {regime_id_str}, error: {e}")
            
            # Load transition matrix
            if 'transition_matrix' in data:
                try:
                    loaded_matrix = np.array(data['transition_matrix'])
                    if loaded_matrix.shape == self.transition_matrix.shape:
                        self.transition_matrix = loaded_matrix
                except Exception as e:
                    logger.warning(f"Error loading transition matrix: {e}")
            
            # Set calibration status
            self.is_calibrated = data.get('is_calibrated', False)
            
            logger.info("Loaded saved regime parameters")
        except Exception as e:
            logger.warning(f"Error loading saved regime parameters: {e}")
            # If anything goes wrong, we just keep the default parameters    
    def save_parameters(self) -> None:
        """Save regime parameters to disk."""
        try:
            params_dir = config_manager.get_cache_dir() / 'regime_params'
            params_dir.mkdir(parents=True, exist_ok=True)
            
            params_file = params_dir / 'regime_parameters.json'
            
            # Prepare data for serialization
            data = {
                'regime_params': {str(r_id): params.as_dict() for r_id, params in self.regime_params.items()},
                'transition_matrix': self.transition_matrix.tolist(),
                'is_calibrated': self.is_calibrated,
                'timestamp': time.time()
            }
            
            # Use a temporary file approach to avoid partial writes
            temp_file = params_dir / f'temp_{int(time.time())}.json'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Rename temp file to target file (atomic operation)
            if params_file.exists():
                try:
                    params_file.unlink()
                except (PermissionError, OSError):
                    # If we can't delete the old file, use a new filename
                    params_file = params_dir / f'regime_parameters_{int(time.time())}.json'
            
            temp_file.rename(params_file)
            
            logger.info(f"Saved regime parameters to {params_file}")
        except Exception as e:
            logger.warning(f"Error saving regime parameters: {e}")
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime based on Wasserstein transport distances and volatility.
        
        Args:
            df: DataFrame with OHLCV data and transport distances
        
        Returns:
            Series containing regime labels for each bar
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for regime detection")
            return pd.Series()
        
        # Calculate regime features
        regime_features = self._calculate_regime_features(df)
        
        if regime_features.empty:
            logger.warning("Failed to calculate regime features")
            return pd.Series(0, index=df.index)  # Default to regime 0
        
        # Scale features
        scaled_features = self._scale_features(regime_features)
        
        # Detect regimes using clustering
        regimes = self._cluster_regimes(scaled_features)
        
        # Create a Series with the regime labels
        regime_series = pd.Series(0, index=df.index)
        regime_series.loc[regime_features.index] = regimes
        
        # Forward fill regime labels
        regime_series = regime_series.fillna(method='ffill').fillna(0).astype(int)
        
        # Update regime history
        self.regime_history.append(regime_series.iloc[-1])
        if len(self.regime_history) > 1000:  # Limit history size
            self.regime_history = self.regime_history[-1000:]
        
        # Update transition matrix based on observed transitions
        self._update_transition_matrix(regime_series)
        
        logger.info(f"Detected regimes: {pd.Series(regimes).value_counts().to_dict()}")
        return regime_series
    
    def _calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for regime detection.
        
        Args:
            df: DataFrame with OHLCV data and transport distances
        
        Returns:
            DataFrame with regime detection features
        """
        # Check for required columns
        required_cols = ['high', 'low', 'close', 'volume']
        w_distance_col = 'w_distance' if 'w_distance' in df.columns else None
        
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found in DataFrame")
                return pd.DataFrame()
        
        # Need at least lookback_window bars
        if len(df) < self.lookback_window:
            logger.warning(f"Not enough data for regime detection (need {self.lookback_window} bars)")
            return pd.DataFrame()
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index[self.lookback_window-1:])
        
        # 1. Volatility (normalized)
        rolling_vol = df['close'].pct_change().rolling(window=self.lookback_window).std()
        features['volatility'] = rolling_vol
        
        # 2. Volume (normalized)
        rolling_volume = df['volume'].rolling(window=self.lookback_window).mean()
        rolling_volume_std = df['volume'].rolling(window=self.lookback_window).std()
        features['volume_z'] = (df['volume'] - rolling_volume) / rolling_volume_std.replace(0, 1)
        
        # 3. Price range (normalized)
        price_range = (df['high'] - df['low']) / df['close']
        features['price_range'] = price_range.rolling(window=self.lookback_window).mean()
        
        # 4. Wasserstein distance features (if available)
        if w_distance_col:
            # Mean distance
            features['w_distance_mean'] = df[w_distance_col].rolling(window=self.lookback_window).mean()
            
            # Distance volatility
            features['w_distance_vol'] = df[w_distance_col].rolling(window=self.lookback_window).std()
            
            # Rate of change of distance
            features['w_distance_roc'] = df[w_distance_col].pct_change().rolling(window=self.lookback_window).mean()
        
        # 5. Distance between consecutive supply and demand measures (if available)
        # This implementation follows Theorem 3.1 from the research paper
        if all(col in df.columns for col in ['w_distance', 'imbalance_signal']):
            # Consecutive measure distance (approximated by signal changes)
            signal_change = df['imbalance_signal'].diff().abs()
            features['measure_diff'] = signal_change.rolling(window=self.lookback_window).mean()
        
        # 6. Directional features
        # Trend strength
        price_sma = df['close'].rolling(window=self.lookback_window).mean()
        price_std = df['close'].rolling(window=self.lookback_window).std()
        features['trend_strength'] = (df['close'] - price_sma) / price_std.replace(0, 1)
        
        # 7. Periodicity features (using autocorrelation)
        returns = df['close'].pct_change()
        
        # Calculate autocorrelation at lag 1
        def rolling_autocorr(x, lag=1):
            return np.array([x.iloc[i:i+self.lookback_window].autocorr(lag=lag) 
                           if i+self.lookback_window <= len(x) else np.nan 
                           for i in range(len(x) - self.lookback_window + 1)])
        
        features['autocorr_1'] = rolling_autocorr(returns, lag=1)
        
        # Drop any remaining NaN values
        features = features.dropna()
        
        return features
    
    def _scale_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Scale regime features for clustering.
        
        Args:
            features: DataFrame with regime detection features
        
        Returns:
            Scaled features as numpy array
        """
        # Replace infinite values with large numbers
        features_clean = features.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaNs with column means
        features_clean = features_clean.fillna(features_clean.mean())
        
        # Fit scaler if not already fit
        if not hasattr(self.feature_scaler, 'mean_') or self.feature_scaler.mean_ is None:
            self.feature_scaler.fit(features_clean.values)
        
        # Transform features
        scaled_features = self.feature_scaler.transform(features_clean.values)
        
        return scaled_features
    
    def _cluster_regimes(self, scaled_features: np.ndarray) -> np.ndarray:
        """
        Cluster features into regime labels.
        
        Args:
            scaled_features: Scaled feature array
        
        Returns:
            Array of regime labels
        """
        # Initialize or update cluster model
        if self.cluster_model is None or not self.is_calibrated:
            # Use Gaussian Mixture Model for more flexible clustering
            self.cluster_model = GaussianMixture(
                n_components=self.num_regimes,
                covariance_type='full',
                random_state=42,
                n_init=10
            )
            self.cluster_model.fit(scaled_features)
            logger.info("Trained new regime clustering model")
        
        # Get cluster assignments
        regimes = self.cluster_model.predict(scaled_features)
        
        # Sort regimes by volatility (0 = low, 1 = medium, 2 = high)
        # This ensures regime numbers are consistent and interpretable
        if len(regimes) > 0:
            unique_regimes = np.unique(regimes)
            
            if len(unique_regimes) > 1 and len(scaled_features) > 0:
                # Calculate average volatility for each regime
                regime_volatilities = {}
                
                for r in unique_regimes:
                    regime_mask = (regimes == r)
                    if np.any(regime_mask):
                        # Use feature 0 (volatility) to sort regimes
                        vol_value = np.mean(scaled_features[regime_mask, 0])
                        regime_volatilities[r] = vol_value
                
                # Create mapping of old regime labels to new sorted ones
                sorted_regimes = sorted(regime_volatilities.items(), key=lambda x: x[1])
                regime_map = {old_r: new_r for new_r, (old_r, _) in enumerate(sorted_regimes)}
                
                # Remap regime labels
                regimes = np.array([regime_map[r] for r in regimes])
        
        return regimes
    
    def _update_transition_matrix(self, regime_series: pd.Series) -> None:
        """
        Update the regime transition probability matrix.
        
        Args:
            regime_series: Series containing regime labels
        """
        if len(regime_series) < 2:
            return
        
        # Count transitions
        transitions = np.zeros((self.num_regimes, self.num_regimes))
        
        # Get non-NaN values
        valid_regimes = regime_series.dropna()
        
        if len(valid_regimes) < 2:
            return
        
        # Count regime transitions
        for i in range(len(valid_regimes) - 1):
            from_regime = int(valid_regimes.iloc[i])
            to_regime = int(valid_regimes.iloc[i + 1])
            
            # Ensure indices are valid
            if 0 <= from_regime < self.num_regimes and 0 <= to_regime < self.num_regimes:
                transitions[from_regime, to_regime] += 1
        
        # Convert to probabilities (rows sum to 1)
        row_sums = transitions.sum(axis=1, keepdims=True)
        
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        
        # Calculate probabilities
        prob_matrix = transitions / row_sums
        
        # Update transition matrix using exponential moving average
        alpha = 0.05  # Update weight
        self.transition_matrix = (1 - alpha) * self.transition_matrix + alpha * prob_matrix
    
    def calibrate_parameters(self, df: pd.DataFrame, regime_labels: pd.Series) -> None:
        """
        Calibrate parameters for each regime based on historical data.
        
        Args:
            df: DataFrame with OHLCV data and signals
            regime_labels: Series containing regime labels
        """
        if df is None or df.empty or regime_labels is None or regime_labels.empty:
            logger.warning("Empty data provided for parameter calibration")
            return
        
        # Keep track of whether we've calibrated any regime
        any_regime_calibrated = False
        
        # Check if we have enough data for each regime
        regime_counts = regime_labels.value_counts()
        
        for regime in range(self.num_regimes):
            if regime not in regime_counts or regime_counts[regime] < 30:
                logger.warning(f"Not enough data for regime {regime}, skipping calibration")
                continue
            
            # Get data for this regime
            regime_data = df[regime_labels == regime]
            
            # Calibrate parameters based on regime characteristics
            self._calibrate_regime_parameters(regime, regime_data)
            any_regime_calibrated = True
        
        # Mark as calibrated only if at least one regime was calibrated
        if any_regime_calibrated:
            self.is_calibrated = True
            # Save parameters
            self.save_parameters()
        else:
            logger.warning("No regimes were calibrated due to insufficient data")
    
    def _calibrate_regime_parameters(self, regime_id: int, regime_data: pd.DataFrame) -> None:
        """
        Calibrate parameters for a specific regime.
        
        Args:
            regime_id: Regime identifier
            regime_data: DataFrame with data for this regime
        """
        params = self.regime_params.get(regime_id, RegimeParameters())
        
        # Skip calibration if not enough data
        if len(regime_data) < 30:
            return
        
        # 1. Calibrate alpha (power law exponent) - based on price range distribution
        price_range = (regime_data['high'] - regime_data['low']) / regime_data['close']
        
        # Alpha controls the skew of the supply/demand distributions
        # Higher price ranges should lead to higher alpha
        mean_range = price_range.mean()
        baseline_range = 0.01  # 1% as baseline
        
        # Scale alpha based on price range
        alpha_scale = mean_range / baseline_range
        params.alpha = min(1.5, max(0.5, 1.0 * alpha_scale))
        
        # 2. Calibrate beta (signal predictiveness) - based on signal-return correlation
        if 'imbalance_signal' in regime_data.columns and 'close' in regime_data.columns:
            future_returns = regime_data['close'].pct_change().shift(-1)
            signal_corr = regime_data['imbalance_signal'].corr(future_returns)
            
            # Higher correlation should lead to higher beta
            # Scale to reasonable range [0.3, 0.7]
            params.beta = min(0.7, max(0.3, 0.5 + signal_corr))
        
        # 3. Calibrate gamma (market impact) - based on volume and volatility
        volatility = regime_data['close'].pct_change().std()
        
        # For higher volatility regimes, increase gamma (more impact)
        vol_ratio = volatility / 0.01  # 1% daily vol as baseline
        params.gamma = min(2.0, max(0.5, 1.0 * vol_ratio))
        
        # 4. Calibrate lambda_param (cost function weight) - based on volatility
        # Higher volatility -> higher lambda to weight the normalized component more
        params.lambda_param = min(0.8, max(0.3, 0.5 * vol_ratio))
        
        # 5. Calibrate eta (regularization) - based on signal quality
        # Lower in high volatility regimes (less smoothing)
        params.eta = max(0.001, min(0.05, 0.01 / vol_ratio))
        
        # 6. Calibrate trading parameters
        
        # Signal threshold - higher in high volatility regimes
        signal_std = 1.0
        if 'imbalance_signal' in regime_data.columns:
            signal_std = regime_data['imbalance_signal'].std()
        
        params.signal_threshold = min(3.0, max(1.0, 1.5 * vol_ratio * signal_std))
        
        # Holding periods - shorter in high volatility regimes
        mean_holding = 5  # baseline
        params.min_holding_period = max(1, int(mean_holding / vol_ratio))
        params.max_holding_period = max(10, int(180 / vol_ratio))
        
        # Position sizing - smaller in high volatility regimes
        params.position_size_factor = min(1.2, max(0.2, 1.0 / vol_ratio))
        
        # Update parameters
        self.regime_params[regime_id] = params
        
        logger.info(f"Calibrated parameters for regime {regime_id}")
    
    def get_regime_parameters(self, regime_id: int) -> RegimeParameters:
        """
        Get parameters for a specific regime.
        
        Args:
            regime_id: Regime identifier
        
        Returns:
            RegimeParameters object
        """
        # Default to normal regime if regime_id not found
        return self.regime_params.get(regime_id, self.regime_params.get(1, RegimeParameters()))
    
    def predict_next_regime(self, current_regime: int) -> int:
        """
        Predict the next regime based on transition probabilities.
        
        Args:
            current_regime: Current regime identifier
        
        Returns:
            Predicted next regime
        """
        if not 0 <= current_regime < self.num_regimes:
            # Default to normal regime
            return 1
        
        # Get transition probabilities from current regime
        probs = self.transition_matrix[current_regime]
        
        # Sample next regime from transition probabilities
        next_regime = np.random.choice(self.num_regimes, p=probs)
        
        return next_regime
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about regimes and transitions.
        
        Returns:
            Dictionary with regime statistics
        """
        # Count occurrences of each regime in history
        regime_counts = pd.Series(self.regime_history).value_counts().to_dict()
        
        # Calculate regime proportions
        total_count = sum(regime_counts.values()) if regime_counts else 1
        regime_props = {r: count / total_count for r, count in regime_counts.items()}
        
        # Calculate average duration of each regime
        durations = {}
        current_regime = None
        current_duration = 0
        
        for regime in self.regime_history:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    durations.setdefault(current_regime, []).append(current_duration)
                current_regime = regime
                current_duration = 1
        
        # Add the last regime duration
        if current_regime is not None:
            durations.setdefault(current_regime, []).append(current_duration)
        
        # Calculate mean durations
        mean_durations = {r: sum(durs) / len(durs) if durs else 0 for r, durs in durations.items()}
        
        # Return statistics
        return {
            "regime_counts": regime_counts,
            "regime_proportions": regime_props,
            "mean_durations": mean_durations,
            "transition_matrix": self.transition_matrix.tolist(),
            "is_calibrated": self.is_calibrated
        }


def calculate_regime_statistic(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Calculate the regime detection statistic R_t as defined in the research paper.
    
    Formula: R_t = (1/h) * sum_{i=1}^h W_1(μ_{t-i+1}, μ_{t-i}) + 
                  λ * (1/h) * sum_{i=1}^h W_1(ν_{t-i+1}, ν_{t-i})
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for the statistic
    
    Returns:
        Series with the regime statistic
    """
    if df is None or df.empty or 'w_distance' not in df.columns:
        logger.warning("Invalid DataFrame or w_distance column not found")
        return pd.Series()
    
    # Calculate Wasserstein distance changes
    w_distance_diff = df['w_distance'].diff().abs()
    
    # Calculate the regime statistic
    regime_stat = w_distance_diff.rolling(window=window).mean()
    
    # Add volatility component
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        avg_vol = volatility.rolling(window=window*2).mean()
        
        # Normalize volatility to its recent history
        # Avoid division by zero
        norm_vol = volatility / avg_vol.replace(0, avg_vol.mean())
        
        # Combine with Wasserstein distance component
        # Using the formula from the implementation steps
        lambda_weight = 0.5
        regime_stat = regime_stat + lambda_weight * norm_vol
    
    # Add volume component
    if 'volume' in df.columns:
        volume = df['volume']
        avg_volume = volume.rolling(window=window*2).mean()
        
        # Normalize volume to its recent history
        # Avoid division by zero
        norm_volume = volume / avg_volume.replace(0, avg_volume.mean())
        
        # Combine with previous components
        volume_weight = 0.3
        regime_stat = regime_stat + volume_weight * (norm_volume - 1.0)
    
    return regime_stat


def detect_regime_changes(regime_stat: pd.Series, 
                         threshold: float = 1.5, 
                         min_duration: int = 30) -> pd.Series:
    """
    Detect regime changes based on the regime statistic.
    
    Args:
        regime_stat: Series with the regime statistic
        threshold: Threshold for regime change detection
        min_duration: Minimum regime duration in bars
    
    Returns:
        Series with regime labels (0 = normal, 1 = high volatility)
    """
    if regime_stat is None or regime_stat.empty:
        logger.warning("Empty regime statistic provided")
        return pd.Series()
    
    # Calculate rolling mean and standard deviation
    roll_mean = regime_stat.rolling(window=100, min_periods=20).mean()
    roll_std = regime_stat.rolling(window=100, min_periods=20).std()
    
    # Calculate standardized statistic
    z_stat = (regime_stat - roll_mean) / roll_std.replace(0, 1)
    
    # Initialize regime series
    regime = pd.Series(0, index=regime_stat.index)
    
    # Detect high volatility regimes (regime = 1)
    regime[z_stat > threshold] = 1
    
    # Apply minimum duration constraint
    # This prevents rapid regime switching
    curr_regime = 0
    curr_duration = 0
    
    for i in range(len(regime)):
        if regime.iloc[i] == curr_regime:
            curr_duration += 1
        else:
            if curr_duration < min_duration:
                # Revert the regime change if duration too short
                regime.iloc[i-curr_duration:i] = 1 - curr_regime
                
                # Continue with the same regime
                curr_duration += 1
            else:
                # Valid regime change
                curr_regime = regime.iloc[i]
                curr_duration = 1
    
    return regime


def get_regime_specific_parameters(regime: int) -> RegimeParameters:
    """
    Get regime-specific parameters.
    
    This is a convenience function that creates a RegimeManager instance
    and returns parameters for the specified regime.
    
    Args:
        regime: Regime identifier
    
    Returns:
        RegimeParameters object for the specified regime
    """
    manager = RegimeManager()
    return manager.get_regime_parameters(regime)