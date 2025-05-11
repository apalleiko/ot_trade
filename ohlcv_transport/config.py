"""
Configuration management for the OHLCV Optimal Transport trading system.

This module handles loading, validating, and accessing configuration settings
for API keys, trading parameters, and asset specifications.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config")

# Default config path
DEFAULT_CONFIG_PATH = Path(os.path.expanduser("~/.ohlcv_transport/config.json"))

# Default configuration settings
DEFAULT_CONFIG = {
    "api": {
        "twelve_data": {
            "api_key": "eae5ad06a7a14217b4429c50dcf27855",
            "base_url": "https://api.twelvedata.com",
            "rate_limit": {
                "calls_per_minute": 8,
                "calls_per_day": 800
            }
        }
    },
    "assets": {
        "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT"],
        "timeframes": ["1min", "5min", "15min", "1hour"],
        "default_timeframe": "1min"
    },
    "trading": {
        "position_sizing": {
            "max_position_size_percent": 5.0,
            "max_position_size_usd": 10000,
            "risk_per_trade_percent": 0.5
        },
        "execution": {
            "signal_threshold": 1.5,
            "min_holding_period_minutes": 5,
            "max_holding_period_minutes": 180
        },
        "risk_management": {
            "max_daily_loss_percent": 2.0,
            "max_drawdown_percent": 10.0,
            "stop_loss_percent": 1.0
        }
    },
    "system": {
        "data_cache_dir": "~/.ohlcv_transport/cache",
        "data_cache_expiry_hours": 24,
        "log_dir": "~/.ohlcv_transport/logs"
    }
}


class ConfigManager:
    """Configuration manager for the OHLCV Optimal Transport trading system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default path.
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default if not exists.
        
        Returns:
            Dict containing configuration settings
        """
        if not self.config_path.exists():
            logger.info(f"Configuration file not found at {self.config_path}. Creating default config.")
            # Create parent directories if they don't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Write default config
            with open(self.config_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            # Merge with default config to ensure all required fields exist
            merged_config = self._merge_configs(DEFAULT_CONFIG, loaded_config)
            return merged_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Falling back to default configuration")
            return DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with default configuration.
        
        Args:
            default: Default configuration dictionary
            user: User configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = default.copy()
        
        for key, value in user.items():
            # If the value is a dictionary and the key exists in default, merge recursively
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            # Otherwise, override the default value
            else:
                result[key] = value
                
        return result
    
    def save_config(self) -> None:
        """Save the current configuration to the config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary containing the configuration updates
        """
        self.config = self._merge_configs(self.config, updates)
        self.save_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation for nested keys.
        
        Args:
            key_path: Dot-separated path to the configuration value, e.g., 'api.twelve_data.api_key'
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value or the default value if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation for nested keys.
        
        Args:
            key_path: Dot-separated path to the configuration value, e.g., 'api.twelve_data.api_key'
            value: The value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the innermost dict
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
        
        # Save the updated config
        self.save_config()
    
    def validate_api_key(self) -> bool:
        """
        Check if the Twelve Data API key is configured.
        
        Returns:
            True if the API key is set, False otherwise
        """
        api_key = self.get('api.twelve_data.api_key')
        if not api_key:
            logger.warning("Twelve Data API key is not configured. Please set it using set_api_key().")
            return False
        return True
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the Twelve Data API key.
        
        Args:
            api_key: The API key string
        """
        self.set('api.twelve_data.api_key', api_key)
        logger.info("API key has been set successfully.")
    
    def get_symbols(self) -> List[str]:
        """
        Get the list of configured trading symbols.
        
        Returns:
            List of symbol strings
        """
        return self.get('assets.symbols', [])
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the configured trading symbols.
        
        Args:
            symbol: The symbol to add (e.g., 'AAPL')
        """
        symbols = self.get_symbols()
        if symbol not in symbols:
            symbols.append(symbol)
            self.set('assets.symbols', symbols)
            logger.info(f"Added symbol: {symbol}")
        else:
            logger.info(f"Symbol {symbol} already exists in configuration")
    
    def get_cache_dir(self) -> Path:
        """
        Get the data cache directory path.
        
        Returns:
            Path object for the cache directory
        """
        cache_dir_str = self.get('system.data_cache_dir', '~/.ohlcv_transport/cache')
        cache_dir = Path(os.path.expanduser(cache_dir_str))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_log_dir(self) -> Path:
        """
        Get the log directory path.
        
        Returns:
            Path object for the log directory
        """
        log_dir_str = self.get('system.log_dir', '~/.ohlcv_transport/logs')
        log_dir = Path(os.path.expanduser(log_dir_str))
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir


# Create a singleton instance for easy import
config_manager = ConfigManager()


if __name__ == "__main__":
    # Example usage
    print("Configuration manager test:")
    print(f"API key configured: {config_manager.validate_api_key()}")
    print(f"Trading symbols: {config_manager.get_symbols()}")
    print(f"Cache directory: {config_manager.get_cache_dir()}")