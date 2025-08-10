"""
Trading signal generation engine
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time

from .fear_greed_index import FearGreedIndex
from ..nlp.sentiment_processor import EnsembleSentimentProcessor
from ..config.settings import TRADING_CONFIG, DATA_CONFIG
from ..utils.data_structures import TradingSignal, SignalType, SentimentScore, DataPoint

class SignalGenerator:
    """Generate trading signals based on sentiment and market data"""

    def __init__(self):
        self.sentiment_processor = EnsembleSentimentProcessor()
        self.fear_greed_index = FearGreedIndex()

        # Signal tracking
        self.last_signals = {}  # symbol -> (timestamp, signal_type)
        self.signal_lock = threading.Lock()

        # Configuration
        self.buy_threshold = TRADING_CONFIG.BUY_THRESHOLD
        self.sell_threshold = TRADING_CONFIG.SELL_THRESHOLD
        self.confidence_threshold = TRADING_CONFIG.CONFIDENCE_THRESHOLD
        self.signal_cooldown = TRADING_CONFIG.SIGNAL_COOLDOWN

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Signal generator initialized")

    def _check_signal_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal for symbol"""
        with self.signal_lock:
            if symbol not in self.last_signals:
                return True

            last_signal_time, _ = self.last_signals[symbol]
            time_since_last = (datetime.now() - last_signal_time).total_seconds()

            return time_since_last >= self.signal_cooldown

    def _update_last_signal(self, symbol: str, signal_type: SignalType):
        """Update the last signal timestamp for a symbol"""
        with self.signal_lock:
            self.last_signals[symbol] = (datetime.now(), signal_type)

    def _calculate_signal_confidence(self, sentiment_score: SentimentScore,
                                   sentiment_momentum: float,
                                   fear_greed_score: float) -> float:
        """Calculate overall signal confidence"""
        # Base confidence from sentiment confidence
        base_confidence = sentiment_score.confidence

        # Boost confidence if sentiment and momentum align
        momentum_boost = 0.0
        if sentiment_score.compound > 0 and sentiment_momentum > 0:
            momentum_boost = 0.2
        elif sentiment_score.compound < 0 and sentiment_momentum < 0:
            momentum_boost = 0.2

        # Adjust based on fear/greed extremes
        fear_greed_boost = 0.0
        if fear_greed_score > 75 or fear_greed_score < 25:
            fear_greed_boost = 0.1  # Extreme readings increase confidence

        # Source count boost (more sources = higher confidence)
        source_boost = min(0.2, sentiment_score.source_count * 0.02)

        total_confidence = base_confidence + momentum_boost + fear_greed_boost + source_boost
        return min(1.0, total_confidence)

    def _generate_signal_reasoning(self, sentiment_score: SentimentScore,
                                 sentiment_momentum: float,
                                 fear_greed_score: float,
                                 signal_type: SignalType) -> str:
        """Generate human-readable reasoning for the signal"""
        reasons = []

        # Sentiment reasoning
        if signal_type == SignalType.BUY:
            reasons.append(f"Strong positive sentiment (compound: {sentiment_score.compound:.3f})")
        elif signal_type == SignalType.SELL:
            reasons.append(f"Strong negative sentiment (compound: {sentiment_score.compound:.3f})")

        # Momentum reasoning
        if sentiment_momentum > 0.1:
            reasons.append("Positive sentiment momentum")
        elif sentiment_momentum < -0.1:
            reasons.append("Negative sentiment momentum")

        # Fear/Greed reasoning
        if fear_greed_score > 75:
            reasons.append("Extreme greed in market")
        elif fear_greed_score < 25:
            reasons.append("Extreme fear in market")

        # Confidence reasoning
        reasons.append(f"High confidence ({sentiment_score.confidence:.3f}) from {sentiment_score.source_count} sources")

        return "; ".join(reasons)

    def generate_signal(self, symbol: str, market_data: Optional[Dict] = None) -> Optional[TradingSignal]:
        """Generate a trading signal for a specific symbol"""
        try:
            # Check cooldown
            if not self._check_signal_cooldown(symbol):
                return None

            # Get current sentiment
            sentiment_score = self.sentiment_processor.aggregate_sentiment(symbol, time_window_hours=1)
            if not sentiment_score:
                return None

            # Check minimum confidence
            if sentiment_score.confidence < self.confidence_threshold:
                return None

            # Get sentiment momentum
            sentiment_momentum = self.sentiment_processor.get_sentiment_momentum(symbol)

            # Get fear/greed index
            fear_greed_data = self.fear_greed_index.calculate_index()
            fear_greed_score = fear_greed_data['score']

            # Determine signal type
            signal_type = SignalType.HOLD

            # Buy signal conditions
            if (sentiment_score.compound > self.buy_threshold and
                sentiment_momentum > 0 and
                sentiment_score.confidence > self.confidence_threshold):
                signal_type = SignalType.BUY

            # Sell signal conditions
            elif (sentiment_score.compound < self.sell_threshold and
                  sentiment_momentum < 0 and
                  sentiment_score.confidence > self.confidence_threshold):
                signal_type = SignalType.SELL

            # Only generate actionable signals (not HOLD)
            if signal_type == SignalType.HOLD:
                return None

            # Calculate signal confidence
            signal_confidence = self._calculate_signal_confidence(
                sentiment_score, sentiment_momentum, fear_greed_score
            )

            # Generate reasoning
            reasoning = self._generate_signal_reasoning(
                sentiment_score, sentiment_momentum, fear_greed_score, signal_type
            )

            # Create signal
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=signal_confidence,
                reasoning=reasoning,
                entry_price=market_data.get('price') if market_data else None,
                time_to_live=3600  # 1 hour
            )

            # Set stop loss and take profit levels
            if market_data and 'price' in market_data:
                entry_price = market_data['price']

                if signal_type == SignalType.BUY:
                    signal.stop_loss = entry_price * 0.95  # 5% stop loss
                    signal.take_profit = entry_price * 1.10  # 10% take profit
                elif signal_type == SignalType.SELL:
                    signal.stop_loss = entry_price * 1.05  # 5% stop loss (for short)
                    signal.take_profit = entry_price * 0.90  # 10% take profit (for short)

            # Update last signal tracking
            self._update_last_signal(symbol, signal_type)

            self.logger.info(f"Generated {signal_type.value} signal for {symbol} with confidence {signal_confidence:.3f}")

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def generate_signals(self, symbols: List[str] = None) -> List[TradingSignal]:
        """Generate signals for multiple symbols"""
        if symbols is None:
            symbols = DATA_CONFIG.CRYPTO_SYMBOLS + DATA_CONFIG.STOCK_SYMBOLS

        signals = []

        for symbol in symbols:
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)

        return signals

    def process_data_batch(self, data_points: List[DataPoint]):
        """Process a batch of data points and update sentiment"""
        # Get sentiment scores
        sentiment_scores = self.sentiment_processor.process_data_points(data_points)

        # Update sentiment history
        self.sentiment_processor.update_sentiment_history(sentiment_scores)

        self.logger.debug(f"Processed {len(data_points)} data points, generated {len(sentiment_scores)} sentiment scores")

    def get_signal_statistics(self) -> Dict:
        """Get statistics about generated signals"""
        with self.signal_lock:
            total_signals = len(self.last_signals)

            # Count signal types (would need to track this separately in a real implementation)
            recent_signals = {
                'total': total_signals,
                'last_hour': 0,  # Would need to implement tracking
                'cooldown_active': total_signals  # Simplified
            }

            return recent_signals

# Test function
def test_signal_generator():
    """Test the signal generation system"""
    generator = SignalGenerator()

    # Create test data points
    test_data = [
        DataPoint(
            timestamp=datetime.now(),
            source=DataType.TWITTER,
            data_type="tweet",
            symbol="BTC",
            content="Bitcoin is showing incredible bullish momentum! To the moon!",
            metadata={}
        ),
        DataPoint(
            timestamp=datetime.now(),
            source=DataType.REDDIT,
            data_type="post",
            symbol="BTC",
            content="BTC breaking resistance levels, very bullish setup",
            metadata={}
        ),
        DataPoint(
            timestamp=datetime.now(),
            source=DataType.NEWS,
            data_type="news_article",
            symbol="BTC",
            content="Bitcoin adoption continues to grow as institutional investors show strong interest",
            metadata={}
        )
    ]

    print("Testing signal generation...")

    # Process test data
    generator.process_data_batch(test_data)

    # Wait a moment for processing
    time.sleep(2)

    # Generate signals
    signals = generator.generate_signals(['BTC', 'ETH', 'AAPL'])

    print(f"\nGenerated {len(signals)} signals:")

    for signal in signals:
        print(f"\nSymbol: {signal.symbol}")
        print(f"Signal: {signal.signal_type.value}")
        print(f"Confidence: {signal.confidence:.3f}")
        print(f"Reasoning: {signal.reasoning}")
        if signal.entry_price:
            print(f"Entry Price: ${signal.entry_price:.2f}")
            print(f"Stop Loss: ${signal.stop_loss:.2f}")
            print(f"Take Profit: ${signal.take_profit:.2f}")

    # Get statistics
    stats = generator.get_signal_statistics()
    print(f"\nSignal Statistics: {stats}")

if __name__ == "__main__":
    test_signal_generator()
