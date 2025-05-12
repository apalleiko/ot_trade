"""
Tests for the configuration management module.
"""
import os
import json
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path

# Import module to test
from ohlcv_transport.config import ConfigManager, DEFAULT_CONFIG


class TestConfigManager(unittest.TestCase):
    """Tests for the ConfigManager class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a config manager with a test configuration path
        self.test_config_path = Path('test_config.json')
        self.config_manager = ConfigManager(self.test_config_path)
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"api": {"twelve_data": {"api_key": "test_key"}}}')
    def test_load_config_existing_file(self, mock_file, mock_exists):
        """Test loading configuration from an existing file."""
        # Mock that the file exists
        mock_exists.return_value = True
        
        # Load the configuration
        config = self.config_manager._load_config()
        
        # Verify the file was opened for reading
        mock_file.assert_called_once_with(self.test_config_path, 'r')
        
        # Verify the configuration was loaded and merged with defaults
        self.assertEqual(config['api']['twelve_data']['api_key'], 'test_key')
        self.assertIn('assets', config)  # Should contain default sections
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_config_new_file(self, mock_file, mock_exists):
        """Test creating a new configuration file if it doesn't exist."""
        # Mock that the file doesn't exist
        mock_exists.return_value = False
        
        # Mock the mkdir method directly on config_path's parent
        self.config_manager.config_path.parent.mkdir = unittest.mock.MagicMock()
        
        # Load the configuration (should create a new file)
        config = self.config_manager._load_config()
        
        # Verify the parent directory was created
        self.config_manager.config_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify the file was opened for writing
        mock_file.assert_called_once_with(self.test_config_path, 'w')
        
        # Verify the configuration is the default
        self.assertEqual(config, DEFAULT_CONFIG)
    
    def test_merge_configs(self):
        """Test merging user and default configurations."""
        default_config = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            },
            'e': [4, 5]
        }
        
        user_config = {
            'b': {
                'c': 6,
                'f': 7
            },
            'g': 8
        }
        
        expected_result = {
            'a': 1,
            'b': {
                'c': 6,
                'd': 3,
                'f': 7
            },
            'e': [4, 5],
            'g': 8
        }
        
        result = self.config_manager._merge_configs(default_config, user_config)
        self.assertEqual(result, expected_result)
    
    def test_save_config(self):
        """Test saving configuration to file."""
        # Set some configuration values
        self.config_manager.config = {
            'test': 'value',
            'nested': {
                'key': 'another_value'
            }
        }
        
        # Mock json.dump instead of open to avoid write count issues with json.dump's implementation
        with patch('json.dump') as mock_dump:
            # Save the configuration
            self.config_manager.save_config()
            
            # Verify json.dump was called with the correct parameters
            mock_dump.assert_called_once()
            # First argument should be the config dict
            args, kwargs = mock_dump.call_args
            self.assertEqual(args[0], self.config_manager.config)
            # indent parameter should be set to 4
            self.assertEqual(kwargs.get('indent'), 4)
    
    def test_get_with_existing_key(self):
        """Test getting a configuration value with an existing key."""
        # Set some configuration values
        self.config_manager.config = {
            'a': {
                'b': {
                    'c': 'test_value'
                }
            }
        }
        
        # Get the value using dot notation
        value = self.config_manager.get('a.b.c')
        
        # Verify the value
        self.assertEqual(value, 'test_value')
    
    def test_get_with_missing_key(self):
        """Test getting a configuration value with a missing key."""
        # Set some configuration values
        self.config_manager.config = {
            'a': {
                'b': {
                    'c': 'test_value'
                }
            }
        }
        
        # Get a value with a missing key
        value = self.config_manager.get('a.b.d')
        
        # Verify the value is None
        self.assertIsNone(value)
        
        # Try with a default value
        value = self.config_manager.get('a.b.d', 'default_value')
        
        # Verify the default value is returned
        self.assertEqual(value, 'default_value')
    
    @patch('ohlcv_transport.config.ConfigManager.save_config')
    def test_set(self, mock_save_config):
        """Test setting a configuration value."""
        # Set some initial configuration values
        self.config_manager.config = {
            'a': {
                'b': {
                    'c': 'old_value'
                }
            }
        }
        
        # Set a value using dot notation
        self.config_manager.set('a.b.c', 'new_value')
        
        # Verify the value was updated
        self.assertEqual(self.config_manager.config['a']['b']['c'], 'new_value')
        
        # Set a value for a new key
        self.config_manager.set('a.b.d', 'another_value')
        
        # Verify the value was added
        self.assertEqual(self.config_manager.config['a']['b']['d'], 'another_value')
        
        # Set a value for a completely new path
        self.config_manager.set('x.y.z', 'new_path_value')
        
        # Verify the path was created and the value was set
        self.assertEqual(self.config_manager.config['x']['y']['z'], 'new_path_value')
        
        # Verify save_config was called for each set operation
        self.assertEqual(mock_save_config.call_count, 3)
    
    def test_validate_api_key(self):
        """Test validating the API key."""
        # Case 1: API key is not set
        self.config_manager.config = {
            'api': {
                'twelve_data': {
                    'api_key': ''
                }
            }
        }
        
        # Validate the API key (should return False)
        result = self.config_manager.validate_api_key()
        self.assertFalse(result)
        
        # Case 2: API key is set
        self.config_manager.config = {
            'api': {
                'twelve_data': {
                    'api_key': 'test_key'
                }
            }
        }
        
        # Validate the API key (should return True)
        result = self.config_manager.validate_api_key()
        self.assertTrue(result)
    
    @patch('ohlcv_transport.config.ConfigManager.set')
    def test_set_api_key(self, mock_set):
        """Test setting the API key."""
        # Set the API key
        self.config_manager.set_api_key('new_api_key')
        
        # Verify the set method was called with the correct parameters
        mock_set.assert_called_once_with('api.twelve_data.api_key', 'new_api_key')
    
    def test_get_symbols(self):
        """Test getting the list of symbols."""
        # Set some configuration values
        self.config_manager.config = {
            'assets': {
                'symbols': ['AAPL', 'MSFT', 'GOOG']
            }
        }
        
        # Get the symbols
        symbols = self.config_manager.get_symbols()
        
        # Verify the symbols
        self.assertEqual(symbols, ['AAPL', 'MSFT', 'GOOG'])
        
        # Case 2: Symbols not in configuration
        self.config_manager.config = {}
        
        # Get the symbols (should return an empty list)
        symbols = self.config_manager.get_symbols()
        self.assertEqual(symbols, [])
    
    @patch('ohlcv_transport.config.ConfigManager.set')
    def test_add_symbol(self, mock_set):
        """Test adding a symbol."""
        # Set some configuration values
        self.config_manager.config = {
            'assets': {
                'symbols': ['AAPL', 'MSFT']
            }
        }
        
        # Add a new symbol
        self.config_manager.add_symbol('GOOG')
        
        # Verify the set method was called with the correct parameters
        mock_set.assert_called_once_with('assets.symbols', ['AAPL', 'MSFT', 'GOOG'])
        
        # Try adding a symbol that already exists
        mock_set.reset_mock()
        self.config_manager.add_symbol('MSFT')
        
        # Verify the set method was not called
        mock_set.assert_not_called()
    
    @patch('os.path.expanduser')
    def test_get_cache_dir(self, mock_expanduser):
        """Test getting the cache directory."""
        # Mock expanduser to return a predictable path
        # Use os.path.normpath to make the test platform-independent
        mock_expanduser.return_value = os.path.normpath('/home/user/test/cache')
        
        # Set some configuration values
        self.config_manager.config = {
            'system': {
                'data_cache_dir': '~/test/cache'
            }
        }
        
        # Mock Path.mkdir
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            # Get the cache directory
            cache_dir = self.config_manager.get_cache_dir()
            
            # Verify the path (using normpath to make platform-independent)
            expected_path = os.path.normpath('/home/user/test/cache')
            self.assertEqual(os.path.normpath(str(cache_dir)), expected_path)
            
            # Verify mkdir was called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('os.path.expanduser')
    def test_get_log_dir(self, mock_expanduser):
        """Test getting the log directory."""
        # Mock expanduser to return a predictable path
        # Use os.path.normpath to make the test platform-independent
        mock_expanduser.return_value = os.path.normpath('/home/user/test/logs')
        
        # Set some configuration values
        self.config_manager.config = {
            'system': {
                'log_dir': '~/test/logs'
            }
        }
        
        # Mock Path.mkdir
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            # Get the log directory
            log_dir = self.config_manager.get_log_dir()
            
            # Verify the path (using normpath to make platform-independent)
            expected_path = os.path.normpath('/home/user/test/logs')
            self.assertEqual(os.path.normpath(str(log_dir)), expected_path)
            
            # Verify mkdir was called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


if __name__ == '__main__':
    unittest.main()