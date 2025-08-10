"""
Market data source for price and volume information
"""

import yfinance as yf
import time
import logging
from datetime import datetime
from typing import Dict, List
from queue import Queue
import threading

from ..config.settings import DATA_CONFIG
from ..utils.data_structures import DataPoint, DataType, MarketData

class MarketDataSource:
    """Market data collector using Yahoo Finance"""

    def __init__(self, data_queue: Queue):
        self.data_queue = data_queue
        self.is_running = False
        self.symbols = self._prepare_symbols()

    def _prepare_symbols(self) -> Dict[str, str]:
        """Map internal symbols to Yahoo Finance tickers"""
        symbol_map = {}

        # Crypto symbols (Yahoo Finance format)
        crypto_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'SOL': 'SOL-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'DOT': 'DOT-USD',
            'LINK': 'LINK-USD',
            'UNI': 'UNI-USD'
        }

        # Stock symbols (already in correct format)
        stock_symbols = DATA_CONFIG.STOCK_SYMBOLS

        symbol_map.update(crypto_map)
        for symbol in stock_symbols:
            symbol_map[symbol] = symbol

        return symbol_map

    def _fetch_market_data(self, symbol: str, yf_ticker: str) -> MarketData:
        """Fetch current market data for a symbol"""
        try:
            ticker = yf.Ticker(yf_ticker)
            info = ticker.info

            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            # Get volume
            volume = info.get('volume') or info.get('regularMarketVolume', 0)

            # Get 24h change
            previous_close = info.get('previousClose', current_price)
            change_24h = ((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0

            # Get market cap
            market_cap = info.get('marketCap')

            return MarketData(
                timestamp=datetime.now(),
                symbol=symbol,
                price=current_price,
                volume=volume,
                change_24h=change_24h,
                market_cap=market_cap
            )

        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def collect_data(self):
        """Collect market data for all symbols"""
        logging.info("Starting market data collection...")

        while self.is_running:
            try:
                for symbol, yf_ticker in self.symbols.items():
                    if not self.is_running:
                        break

                    market_data = self._fetch_market_data(symbol, yf_ticker)

                    if market_data:
                        # Create data point
                        data_point = DataPoint(
                            timestamp=market_data.timestamp,
                            source=DataType.MARKET,
                            data_type="price_data",
                            symbol=symbol,
                            content=f"Price: ${market_data.price:.2f}, Volume: {market_data.volume:,}, Change: {market_data.change_24h:.2f}%",
                            metadata={
                                'price': market_data.price,
                                'volume': market_data.volume,
                                'change_24h': market_data.change_24h,
                                'market_cap': market_data.market_cap
                            }
                        )

                        if not self.data_queue.full():
                            self.data_queue.put(data_point)

                time.sleep(60)  # Update every minute

            except Exception as e:
                logging.error(f"Market data collection error: {e}")
                time.sleep(120)

    def start(self):
        """Start data collection in separate thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.collect_data, daemon=True)
        self.thread.start()
        logging.info("Market data source started")

    def stop(self):
        """Stop data collection"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        logging.info("Market data source stopped")
