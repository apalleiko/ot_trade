"""
Project initialization script.

This script creates the necessary directory structure and files
to set up the OHLCV Optimal Transport trading system.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("init_project")

# Directory structure
DIRECTORY_STRUCTURE = [
    "ohlcv_transport",
    "ohlcv_transport/data",
    "ohlcv_transport/models",
    "ohlcv_transport/signals",
    "ohlcv_transport/execution",
    "ohlcv_transport/utils",
    "tests",
    "examples",
    "notebooks"
]

# Init files to create
INIT_FILES = [
    "ohlcv_transport/__init__.py",
    "ohlcv_transport/data/__init__.py",
    "ohlcv_transport/models/__init__.py",
    "ohlcv_transport/signals/__init__.py",
    "ohlcv_transport/execution/__init__.py",
    "ohlcv_transport/utils/__init__.py",
    "tests/__init__.py"
]

# Package version and metadata
PACKAGE_VERSION = "0.1.0"
PACKAGE_METADATA = {
    "name": "ohlcv_transport",
    "description": "OHLCV Optimal Transport trading system",
    "author": "Quantitative Trader"
}

# Requirements list
REQUIREMENTS = [
    "numpy>=1.22.0",
    "pandas>=1.4.0",
    "scipy>=1.8.0",
    "requests>=2.27.0",
    "pywavelets>=1.2.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "joblib>=1.1.0",
    "tqdm>=4.62.0",
    "pytest>=7.0.0",
    "jupyter>=1.0.0"
]


class ProjectInitializer:
    """Project initialization tool."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the project initializer.
        
        Args:
            base_dir: Base directory for the project (defaults to current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.project_name = PACKAGE_METADATA["name"]
    
    def create_directory_structure(self) -> None:
        """Create the project directory structure."""
        logger.info(f"Creating directory structure in {self.base_dir}")
        
        for directory in DIRECTORY_STRUCTURE:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def create_init_files(self) -> None:
        """Create __init__.py files in all package directories."""
        logger.info("Creating __init__.py files")
        
        for init_file in INIT_FILES:
            file_path = self.base_dir / init_file
            
            # Add version to main __init__.py
            if init_file == "ohlcv_transport/__init__.py":
                content = f'''"""
{PACKAGE_METADATA["description"]}
"""

__version__ = "{PACKAGE_VERSION}"
__author__ = "{PACKAGE_METADATA["author"]}"
'''
            else:
                content = '"""Module initialization."""\n'
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Created file: {file_path}")
    
    def create_requirements_file(self) -> None:
        """Create requirements.txt file."""
        requirements_path = self.base_dir / "requirements.txt"
        
        with open(requirements_path, 'w') as f:
            for requirement in REQUIREMENTS:
                f.write(f"{requirement}\n")
        
        logger.info(f"Created requirements.txt: {requirements_path}")
    
    def create_setup_py(self) -> None:
        """Create setup.py file."""
        setup_path = self.base_dir / "setup.py"
        
        content = f'''"""
Project setup script for the {PACKAGE_METADATA["description"]}.
"""
from setuptools import setup, find_packages

setup(
    name="{PACKAGE_METADATA["name"]}",
    version="{PACKAGE_VERSION}",
    packages=find_packages(),
    install_requires={repr(REQUIREMENTS)},
    python_requires=">=3.8",
    author="{PACKAGE_METADATA["author"]}",
    description="{PACKAGE_METADATA["description"]}",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
'''
        
        with open(setup_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created setup.py: {setup_path}")
    
    def create_readme(self) -> None:
        """Create README.md file."""
        readme_path = self.base_dir / "README.md"
        
        content = f'''# {PACKAGE_METADATA["name"]}

{PACKAGE_METADATA["description"]}

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/{PACKAGE_METADATA["name"]}.git
cd {PACKAGE_METADATA["name"]}

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
'''
        
        with open(readme_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created README.md: {readme_path}")
    
    def initialize_project(self) -> None:
        """Initialize the project by creating all necessary files and directories."""
        try:
            self.create_directory_structure()
            self.create_init_files()
            self.create_requirements_file()
            self.create_setup_py()
            self.create_readme()
            
            logger.info(f"Project {self.project_name} initialized successfully!")
            logger.info(f"Next steps:")
            logger.info(f"1. Install the package in development mode: pip install -e .")
            logger.info(f"2. Configure your Twelve Data API key")
            logger.info(f"3. Start developing!")
            
        except Exception as e:
            logger.error(f"Error initializing project: {e}")
            raise


if __name__ == "__main__":
    # Parse command line argument for base directory
    base_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Initialize project
    initializer = ProjectInitializer(base_dir)
    initializer.initialize_project()