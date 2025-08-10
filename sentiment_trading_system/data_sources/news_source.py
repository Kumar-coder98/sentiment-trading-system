"""
News data source for financial sentiment analysis
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List
from queue import Queue
import threading
import re

from ..config.settings import API_CONFIG, DATA_CONFIG
from ..utils.data_structures import DataPoint, DataType

class NewsDataSource:
    """News aggregator for financial sentiment"""

    def __init__(self, data_queue: Queue):
        self.data_queue = data_queue
        self.api_key = API_CONFIG.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"
        self.is_running = False
        self.symbols = DATA_CONFIG.CRYPTO_SYMBOLS + DATA_CONFIG.STOCK_SYMBOLS
        self.last_update = datetime.now() - timedelta(hours=1)

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract financial symbols from news text"""
        symbols = []
        text_upper = text.upper()

        # Check for each symbol
        for symbol in self.symbols:
            if symbol.upper() in text_upper:
                symbols.append(symbol.upper())

        # Additional crypto patterns
        crypto_map = {
            'BITCOIN': 'BTC',
            'ETHEREUM': 'ETH',
            'SOLANA': 'SOL',
            'DOGECOIN': 'DOGE',
            'CARDANO': 'ADA',
            'POLKADOT': 'DOT',
            'CHAINLINK': 'LINK',
            'UNISWAP': 'UNI'
        }

        for crypto_name, symbol in crypto_map.items():
            if crypto_name in text_upper:
                symbols.append(symbol)

        return list(set(symbols))

    def _build_queries(self) -> List[str]:
        """Build search queries for financial news"""
        queries = [
            "cryptocurrency trading",
            "bitcoin ethereum",
            "stock market",
            "financial markets",
            "investment analysis",
            "crypto market",
            "trading signals",
            "market sentiment"
        ]

        # Add specific symbol queries
        for symbol in self.symbols[:5]:  # Limit to avoid API quota
            queries.append(f"{symbol} trading")

        return queries

    def _fetch_news(self, query: str, page_size: int = 20) -> List[dict]:
        """Fetch news articles from NewsAPI"""
        try:
            params = {
                'q': query,
                'apiKey': self.api_key,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': page_size,
                'from': self.last_update.isoformat()
            }

            response = requests.get(f"{self.base_url}/everything", params=params)

            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logging.error(f"News API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            return []

    def collect_data(self):
        """Collect news data"""
        logging.info("Starting news data collection...")

        while self.is_running:
            try:
                queries = self._build_queries()

                for query in queries:
                    if not self.is_running:
                        break

                    articles = self._fetch_news(query)

                    for article in articles:
                        if not self.is_running:
                            break

                        title = article.get('title', '')
                        description = article.get('description', '')
                        content = article.get('content', '')

                        full_text = f"{title} {description} {content}"
                        symbols = self._extract_symbols(full_text)

                        if symbols:
                            published_at = article.get('publishedAt')
                            timestamp = datetime.now()

                            if published_at:
                                try:
                                    timestamp = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                                except:
                                    pass

                            for symbol in symbols:
                                data_point = DataPoint(
                                    timestamp=timestamp,
                                    source=DataType.NEWS,
                                    data_type="news_article",
                                    symbol=symbol,
                                    content=full_text,
                                    metadata={
                                        'source_name': article.get('source', {}).get('name'),
                                        'author': article.get('author'),
                                        'url': article.get('url'),
                                        'query': query
                                    }
                                )

                                if not self.data_queue.full():
                                    self.data_queue.put(data_point)

                    time.sleep(2)  # Rate limiting between queries

                self.last_update = datetime.now()
                time.sleep(300)  # Wait 5 minutes between full cycles

            except Exception as e:
                logging.error(f"News collection error: {e}")
                time.sleep(300)

    def start(self):
        """Start data collection in separate thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.collect_data, daemon=True)
        self.thread.start()
        logging.info("News data source started")

    def stop(self):
        """Stop data collection"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        logging.info("News data source stopped")
