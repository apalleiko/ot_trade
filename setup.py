"""
Project setup script for the OHLCV Optimal Transport trading system.
"""
from setuptools import setup, find_packages

setup(
    name="ohlcv_transport",
    version="0.1.0",
    packages=find_packages(),
    install_requires=['numpy>=1.22.0', 'pandas>=1.4.0', 'scipy>=1.8.0', 'requests>=2.27.0', 'pywavelets>=1.2.0', 'scikit-learn>=1.0.0', 'matplotlib>=3.5.0', 'seaborn>=0.11.0', 'joblib>=1.1.0', 'tqdm>=4.62.0', 'pytest>=7.0.0', 'jupyter>=1.0.0'],
    python_requires=">=3.8",
    author="Quantitative Trader",
    description="OHLCV Optimal Transport trading system",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
