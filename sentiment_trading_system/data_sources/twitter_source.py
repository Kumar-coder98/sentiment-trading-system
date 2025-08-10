"""
Twitter data source for real-time sentiment analysis
"""

import tweepy
import time
import logging
from datetime import datetime
from typing import List, Optional
from queue import Queue
import threading
import re

from ..config.settings import API_CONFIG, DATA_CONFIG
from ..utils.data_structures import DataPoint, DataType

class TwitterDataSource:
    """High-performance Twitter data collector"""

    def __init__(self, data_queue: Queue):
        self.data_queue = data_queue
        self.client = tweepy.Client(bearer_token=API_CONFIG.TWITTER_BEARER_TOKEN)
        self.is_running = False
        self.symbols = DATA_CONFIG.CRYPTO_SYMBOLS + DATA_CONFIG.STOCK_SYMBOLS
        self.rate_limiter = self._setup_rate_limiter()

    def _setup_rate_limiter(self):
        """Setup rate limiting (300 requests per 15 minutes)"""
        return {
            'requests': 0,
            'reset_time': time.time() + 900,  # 15 minutes
            'max_requests': 300
        }

    def _check_rate_limit(self):
        """Check and handle rate limiting"""
        current_time = time.time()

        if current_time > self.rate_limiter['reset_time']:
            self.rate_limiter['requests'] = 0
            self.rate_limiter['reset_time'] = current_time + 900

        if self.rate_limiter['requests'] >= self.rate_limiter['max_requests']:
            sleep_time = self.rate_limiter['reset_time'] - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.rate_limiter['requests'] = 0
                self.rate_limiter['reset_time'] = time.time() + 900

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract financial symbols from tweet text"""
        symbols = []

        # Extract cashtags ($SYMBOL)
        cashtags = re.findall(r'\$([A-Z]{1,5})\b', text.upper())
        symbols.extend(cashtags)

        # Extract known symbols
        for symbol in self.symbols:
            if symbol.upper() in text.upper():
                symbols.append(symbol.upper())

        return list(set(symbols))

    def _build_search_query(self) -> str:
        """Build optimized search query for financial content"""
        symbol_queries = [f"${symbol}" for symbol in self.symbols[:10]]  # Limit to avoid query length

        base_terms = [
            "crypto", "trading", "bullish", "bearish", "pump", "dump",
            "moon", "dip", "buy", "sell", "hodl", "investment"
        ]

        query_parts = symbol_queries + base_terms
        query = " OR ".join(query_parts)

        # Add filters
        query += " -is:retweet lang:en"

        return query

    def collect_data(self):
        """Collect tweets in real-time"""
        logging.info("Starting Twitter data collection...")

        while self.is_running:
            try:
                self._check_rate_limit()

                query = self._build_search_query()

                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics', 'context_annotations']
                ).flatten(limit=500)

                self.rate_limiter['requests'] += 1

                for tweet in tweets:
                    if not self.is_running:
                        break

                    symbols = self._extract_symbols(tweet.text)

                    for symbol in symbols:
                        data_point = DataPoint(
                            timestamp=tweet.created_at or datetime.now(),
                            source=DataType.TWITTER,
                            data_type="tweet",
                            symbol=symbol,
                            content=tweet.text,
                            metadata={
                                'tweet_id': tweet.id,
                                'retweet_count': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                                'like_count': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                                'reply_count': tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0
                            }
                        )

                        if not self.data_queue.full():
                            self.data_queue.put(data_point)

                time.sleep(10)  # Wait before next collection cycle

            except Exception as e:
                logging.error(f"Twitter collection error: {e}")
                time.sleep(30)  # Wait longer on error

    def start(self):
        """Start data collection in separate thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.collect_data, daemon=True)
        self.thread.start()
        logging.info("Twitter data source started")

    def stop(self):
        """Stop data collection"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        logging.info("Twitter data source stopped")
