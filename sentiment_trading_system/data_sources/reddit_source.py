"""
Reddit data source for sentiment analysis
"""

import praw
import time
import logging
from datetime import datetime
from typing import List
from queue import Queue
import threading
import re

from ..config.settings import API_CONFIG, DATA_CONFIG
from ..utils.data_structures import DataPoint, DataType

class RedditDataSource:
    """Reddit data collector for financial discussions"""

    def __init__(self, data_queue: Queue):
        self.data_queue = data_queue
        self.reddit = praw.Reddit(
            client_id=API_CONFIG.REDDIT_CLIENT_ID,
            client_secret=API_CONFIG.REDDIT_SECRET,
            user_agent=API_CONFIG.REDDIT_USER_AGENT
        )
        self.is_running = False
        self.symbols = DATA_CONFIG.CRYPTO_SYMBOLS + DATA_CONFIG.STOCK_SYMBOLS
        self.subreddits = [
            'CryptoCurrency', 'Bitcoin', 'ethereum', 'investing', 'stocks',
            'wallstreetbets', 'SecurityAnalysis', 'ValueInvesting', 'crypto',
            'altcoin', 'CryptoMarkets', 'StockMarket'
        ]

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract financial symbols from Reddit text"""
        symbols = []

        # Extract ticker mentions
        tickers = re.findall(r'\b([A-Z]{1,5})\b', text)
        for ticker in tickers:
            if ticker.upper() in [s.upper() for s in self.symbols]:
                symbols.append(ticker.upper())

        # Extract crypto mentions
        crypto_patterns = [
            r'\b(bitcoin|btc)\b',
            r'\b(ethereum|eth)\b',
            r'\b(solana|sol)\b',
            r'\b(dogecoin|doge)\b'
        ]

        for pattern in crypto_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if 'bitcoin' in match or 'btc' in match:
                    symbols.append('BTC')
                elif 'ethereum' in match or 'eth' in match:
                    symbols.append('ETH')
                elif 'solana' in match or 'sol' in match:
                    symbols.append('SOL')
                elif 'dogecoin' in match or 'doge' in match:
                    symbols.append('DOGE')

        return list(set(symbols))

    def _process_submission(self, submission):
        """Process a Reddit submission"""
        try:
            title_symbols = self._extract_symbols(submission.title)
            selftext_symbols = self._extract_symbols(submission.selftext or "")
            all_symbols = list(set(title_symbols + selftext_symbols))

            if all_symbols:
                full_text = f"{submission.title} {submission.selftext or ''}"

                for symbol in all_symbols:
                    data_point = DataPoint(
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        source=DataType.REDDIT,
                        data_type="post",
                        symbol=symbol,
                        content=full_text,
                        metadata={
                            'submission_id': submission.id,
                            'subreddit': str(submission.subreddit),
                            'score': submission.score,
                            'upvote_ratio': submission.upvote_ratio,
                            'num_comments': submission.num_comments,
                            'author': str(submission.author) if submission.author else None
                        }
                    )

                    if not self.data_queue.full():
                        self.data_queue.put(data_point)

        except Exception as e:
            logging.error(f"Error processing Reddit submission: {e}")

    def _process_comments(self, submission, limit=10):
        """Process comments from a submission"""
        try:
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    symbols = self._extract_symbols(comment.body)

                    for symbol in symbols:
                        data_point = DataPoint(
                            timestamp=datetime.fromtimestamp(comment.created_utc),
                            source=DataType.REDDIT,
                            data_type="comment",
                            symbol=symbol,
                            content=comment.body,
                            metadata={
                                'comment_id': comment.id,
                                'submission_id': comment.submission.id,
                                'subreddit': str(comment.subreddit),
                                'score': comment.score,
                                'author': str(comment.author) if comment.author else None
                            }
                        )

                        if not self.data_queue.full():
                            self.data_queue.put(data_point)

        except Exception as e:
            logging.error(f"Error processing Reddit comments: {e}")

    def collect_data(self):
        """Collect Reddit data in real-time"""
        logging.info("Starting Reddit data collection...")

        while self.is_running:
            try:
                for subreddit_name in self.subreddits:
                    if not self.is_running:
                        break

                    try:
                        subreddit = self.reddit.subreddit(subreddit_name)

                        # Get hot posts
                        for submission in subreddit.hot(limit=10):
                            if not self.is_running:
                                break

                            self._process_submission(submission)
                            self._process_comments(submission, limit=5)

                        # Get new posts
                        for submission in subreddit.new(limit=5):
                            if not self.is_running:
                                break

                            self._process_submission(submission)

                    except Exception as e:
                        logging.error(f"Error accessing subreddit {subreddit_name}: {e}")
                        continue

                time.sleep(60)  # Wait 1 minute between cycles to respect rate limits

            except Exception as e:
                logging.error(f"Reddit collection error: {e}")
                time.sleep(120)  # Wait longer on error

    def start(self):
        """Start data collection in separate thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self.collect_data, daemon=True)
        self.thread.start()
        logging.info("Reddit data source started")

    def stop(self):
        """Stop data collection"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        logging.info("Reddit data source stopped")
