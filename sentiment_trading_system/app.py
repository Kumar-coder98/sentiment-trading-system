"""
Main application runner for the sentiment trading system
"""

import logging
import time
import signal
import sys
from datetime import datetime
from typing import List

from data_sources.data_ingestion import DataIngestionEngine
from signals.signal_engine import SignalGenerator
from nlp.sentiment_processor import EnsembleSentimentProcessor
from config.settings import DATA_CONFIG, TRADING_CONFIG
from utils.data_structures import TradingSignal

class SentimentTradingSystem:
    """Main sentiment trading system coordinator"""

    def __init__(self):
        self.data_engine = DataIngestionEngine()
        self.signal_generator = SignalGenerator()
        self.is_running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('sentiment_trading.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the sentiment trading system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return

        self.logger.info("Starting Sentiment Trading System...")
        self.is_running = True

        try:
            # Start data ingestion
            self.data_engine.start()
            self.logger.info("Data ingestion engine started")

            # Start signal generation loop
            self._run_signal_loop()

        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.stop()

    def stop(self):
        """Stop the sentiment trading system"""
        if not self.is_running:
            return

        self.logger.info("Stopping Sentiment Trading System...")
        self.is_running = False

        # Stop data ingestion
        try:
            self.data_engine.stop()
            self.logger.info("Data ingestion engine stopped")
        except Exception as e:
            self.logger.error(f"Error stopping data engine: {e}")

    def _run_signal_loop(self):
        """Main signal generation loop"""
        self.logger.info("Starting signal generation loop...")

        last_signal_time = time.time()
        signal_interval = 60  # Generate signals every minute

        while self.is_running:
            try:
                current_time = time.time()

                if current_time - last_signal_time >= signal_interval:
                    # Generate signals for all symbols
                    signals = self.signal_generator.generate_signals()

                    if signals:
                        self._process_signals(signals)

                    # Get system statistics
                    queue_status = self.data_engine.get_queue_status()
                    signal_stats = self.signal_generator.get_signal_statistics()

                    self.logger.info(
                        f"System Status - Queue: {queue_status['queue_size']}/{queue_status['max_size']} "
                        f"({queue_status['utilization']:.1f}%), Signals: {signal_stats['total']}"
                    )

                    last_signal_time = current_time

                time.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error(f"Error in signal loop: {e}")
                time.sleep(5)  # Wait before retrying

    def _process_signals(self, signals: List[TradingSignal]):
        """Process generated trading signals"""
        for signal in signals:
            self._handle_signal(signal)

    def _handle_signal(self, signal: TradingSignal):
        """Handle an individual trading signal"""
        # Log the signal
        self.logger.info(
            f"SIGNAL: {signal.signal_type.value} {signal.symbol} "
            f"(Confidence: {signal.confidence:.3f}) - {signal.reasoning}"
        )

        # In a real implementation, this would:
        # 1. Send signal to trading platform
        # 2. Update position management system
        # 3. Log to trading database
        # 4. Send notifications to users

        # For demo, just print detailed signal info
        print(f"\n{'='*60}")
        print(f"TRADING SIGNAL GENERATED")
        print(f"{'='*60}")
        print(f"Timestamp: {signal.timestamp}")
        print(f"Symbol: {signal.symbol}")
        print(f"Action: {signal.signal_type.value}")
        print(f"Confidence: {signal.confidence:.3f}")
        print(f"Reasoning: {signal.reasoning}")

        if signal.entry_price:
            print(f"Entry Price: ${signal.entry_price:.2f}")
            print(f"Stop Loss: ${signal.stop_loss:.2f}")
            print(f"Take Profit: ${signal.take_profit:.2f}")

        print(f"Time to Live: {signal.time_to_live} seconds")
        print(f"{'='*60}\n")

    def get_system_status(self) -> dict:
        """Get current system status"""
        queue_status = self.data_engine.get_queue_status()
        signal_stats = self.signal_generator.get_signal_statistics()

        return {
            'is_running': self.is_running,
            'timestamp': datetime.now().isoformat(),
            'data_queue': queue_status,
            'signals': signal_stats
        }

def main():
    """Main function"""
    print("Sentiment Trading System")
    print("=" * 50)
    print("Real-time sentiment analysis and trade signal generation")
    print("Data sources: Twitter, Reddit, News, Market Data")
    print("Analysis: FinBERT + VADER + TextBlob ensemble")
    print("=" * 50)

    system = SentimentTradingSystem()

    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"\nSystem error: {e}")
    finally:
        system.stop()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
