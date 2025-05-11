# ohlcv_transport

OHLCV Optimal Transport trading system

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ohlcv_transport.git
cd ohlcv_transport

# Create conda environment
conda create -n ohlcv-transport python=3.10 -y
conda activate ohlcv-transport

# Install dependencies
pip install -e .
```

## Configuration

Before using the system, you need to set up your Twelve Data API key:

```python
from ohlcv_transport.config import config_manager

# Set your API key
config_manager.set_api_key("YOUR_API_KEY")
```

## Usage

Basic usage example:

```python
from ohlcv_transport.api_client import twelve_data_client

# Get OHLCV data
df = twelve_data_client.get_time_series(
    symbol="AAPL",
    interval="1min",
    outputsize=100
)

print(df.head())
```

## Development

Run tests:

```bash
pytest tests/
```
