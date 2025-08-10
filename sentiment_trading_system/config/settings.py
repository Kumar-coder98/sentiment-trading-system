"""
Configuration settings for the sentiment trading system
"""

import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class APIConfig:
    """API configuration settings"""
    TWITTER_BEARER_TOKEN: str = "AAAAAAAAAAAAAAAAAAAAAMkm3AEAAAAA3e30tENeqCyMouhmrM8JIXE9E98%3DHJh0apfFYWMNr6RGkw8lCrgsRmM0et5aKoRHCp7m9ayXFm6lB5"
    REDDIT_CLIENT_ID: str = "KoqRf7zaqYMpMSKiG3iTeA"
    REDDIT_SECRET: str = "ltLPXaU-J9aOpy6lFAwH3xKDsJCzvg"
    REDDIT_USER_AGENT: str = "goquant-analysis"
    NEWS_API_KEY: str = "1472cf26198c46a68a180c4a214f36a1"

@dataclass
class TradingConfig:
    """Trading signal configuration"""
    BUY_THRESHOLD: float = 0.7
    SELL_THRESHOLD: float = -0.7
    CONFIDENCE_THRESHOLD: float = 0.6
    SIGNAL_COOLDOWN: int = 300  # seconds
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio

@dataclass
class DataConfig:
    """Data processing configuration"""
    BATCH_SIZE: int = 100
    QUEUE_SIZE: int = 10000
    UPDATE_INTERVAL: int = 1  # seconds
    MAX_THREADS: int = 8

    # Target symbols for analysis
    CRYPTO_SYMBOLS: List[str] = None
    STOCK_SYMBOLS: List[str] = None

    def __post_init__(self):
        if self.CRYPTO_SYMBOLS is None:
            self.CRYPTO_SYMBOLS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'DOT', 'LINK', 'UNI']
        if self.STOCK_SYMBOLS is None:
            self.STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'SPY']

@dataclass
class NLPConfig:
    """NLP processing configuration"""
    FINBERT_MODEL: str = "ProsusAI/finbert"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 32
    DEVICE: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"

    # Sentiment aggregation weights
    FINBERT_WEIGHT: float = 0.6
    VADER_WEIGHT: float = 0.3
    TEXTBLOB_WEIGHT: float = 0.1

    # Time decay factor for sentiment
    TIME_DECAY_FACTOR: float = 0.95  # per hour

# Global configuration instances
API_CONFIG = APIConfig()
TRADING_CONFIG = TradingConfig()
DATA_CONFIG = DataConfig()
NLP_CONFIG = NLPConfig()
