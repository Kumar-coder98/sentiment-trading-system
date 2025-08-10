"""
Core data structures for the sentiment trading system
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class DataType(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    MARKET = "market"

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class DataPoint:
    """Normalized data point from any source"""
    timestamp: datetime
    source: DataType
    data_type: str
    symbol: Optional[str]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'data_type': self.data_type,
            'symbol': self.symbol,
            'content': self.content,
            'metadata': self.metadata
        }

@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    timestamp: datetime
    symbol: str
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    source_count: int = 1

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral,
            'compound': self.compound,
            'confidence': self.confidence,
            'source_count': self.source_count
        }

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    reasoning: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_to_live: int = 3600  # seconds

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'time_to_live': self.time_to_live
        }

@dataclass
class MarketData:
    """Market data point"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    change_24h: float
    market_cap: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'change_24h': self.change_24h,
            'market_cap': self.market_cap
        }
