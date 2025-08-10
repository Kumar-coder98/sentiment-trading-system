"""
Main data ingestion engine that coordinates all data sources
"""

import logging
import time
from queue import Queue, Empty
from threading import Thread
from typing import List

from .twitter_source import TwitterDataSource
from .reddit_source import RedditDataSource
from .news_source import NewsDataSource
from .market_source import MarketDataSource
from ..config.settings import DATA_CONFIG
from ..utils.data_structures import DataPoint

class DataIngestionEngine:
    """Coordinates all data sources and manages the data pipeline"""

    def __init__(self):
        self.data_queue = Queue(maxsize=DATA_CONFIG.QUEUE_SIZE)
        self.is_running = False
        self.data_sources = []
        self.processor_thread = None

        # Initialize data sources
        self._initialize_sources()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_sources(self):
        """Initialize all data sources"""
        try:
            self.twitter_source = TwitterDataSource(self.data_queue)
            self.data_sources.append(self.twitter_source)
            self.logger.info("Twitter source initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Twitter source: {e}")

        try:
            self.reddit_source = RedditDataSource(self.data_queue)
            self.data_sources.append(self.reddit_source)
            self.logger.info("Reddit source initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit source: {e}")

        try:
            self.news_source = NewsDataSource(self.data_queue)
            self.data_sources.append(self.news_source)
            self.logger.info("News source initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize News source: {e}")

        try:
            self.market_source = MarketDataSource(self.data_queue)
            self.data_sources.append(self.market_source)
            self.logger.info("Market source initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Market source: {e}")

    def _process_data_batch(self, batch: List[DataPoint]):
        """Process a batch of data points"""
        if not batch:
            return

        # Group by symbol for efficient processing
        symbol_groups = {}
        for data_point in batch:
            if data_point.symbol:
                if data_point.symbol not in symbol_groups:
                    symbol_groups[data_point.symbol] = []
                symbol_groups[data_point.symbol].append(data_point)

        # Log batch statistics
        self.logger.info(f"Processed batch: {len(batch)} points, {len(symbol_groups)} symbols")

        for symbol, points in symbol_groups.items():
            source_counts = {}
            for point in points:
                source = point.source.value
                source_counts[source] = source_counts.get(source, 0) + 1

            self.logger.debug(f"Symbol {symbol}: {source_counts}")

    def _data_processor(self):
        """Process data from the queue in batches"""
        batch = []
        last_process_time = time.time()

        while self.is_running:
            try:
                # Try to get data with timeout
                try:
                    data_point = self.data_queue.get(timeout=1.0)
                    batch.append(data_point)
                except Empty:
                    pass

                # Process batch if it's full or enough time has passed
                current_time = time.time()
                should_process = (
                    len(batch) >= DATA_CONFIG.BATCH_SIZE or
                    (batch and current_time - last_process_time >= DATA_CONFIG.UPDATE_INTERVAL)
                )

                if should_process:
                    self._process_data_batch(batch)
                    batch = []
                    last_process_time = current_time

            except Exception as e:
                self.logger.error(f"Error in data processor: {e}")
                time.sleep(1)

        # Process remaining batch
        if batch:
            self._process_data_batch(batch)

    def start(self):
        """Start the data ingestion engine"""
        if self.is_running:
            self.logger.warning("Data ingestion engine is already running")
            return

        self.logger.info("Starting data ingestion engine...")
        self.is_running = True

        # Start data processor thread
        self.processor_thread = Thread(target=self._data_processor, daemon=True)
        self.processor_thread.start()

        # Start all data sources
        for source in self.data_sources:
            try:
                source.start()
            except Exception as e:
                self.logger.error(f"Failed to start data source {type(source).__name__}: {e}")

        self.logger.info("Data ingestion engine started successfully")

    def stop(self):
        """Stop the data ingestion engine"""
        if not self.is_running:
            return

        self.logger.info("Stopping data ingestion engine...")
        self.is_running = False

        # Stop all data sources
        for source in self.data_sources:
            try:
                source.stop()
            except Exception as e:
                self.logger.error(f"Error stopping data source {type(source).__name__}: {e}")

        # Wait for processor thread
        if self.processor_thread:
            self.processor_thread.join()

        self.logger.info("Data ingestion engine stopped")

    def get_queue_status(self) -> dict:
        """Get current queue status"""
        return {
            'queue_size': self.data_queue.qsize(),
            'max_size': DATA_CONFIG.QUEUE_SIZE,
            'utilization': self.data_queue.qsize() / DATA_CONFIG.QUEUE_SIZE * 100
        }

def main():
    """Main function for testing the data ingestion engine"""
    engine = DataIngestionEngine()

    try:
        engine.start()

        # Run for 2 minutes for demonstration
        start_time = time.time()
        while time.time() - start_time < 120:  # 2 minutes
            status = engine.get_queue_status()
            print(f"Queue status: {status['queue_size']}/{status['max_size']} ({status['utilization']:.1f}%)")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        engine.stop()

if __name__ == "__main__":
    main()
