"""
Ensemble sentiment processor combining FinBERT, VADER, and TextBlob
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict
import threading
import numpy as np

# Traditional sentiment analyzers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from .finbert_analyzer import FinBERTAnalyzer
from ..config.settings import NLP_CONFIG
from ..utils.data_structures import DataPoint, SentimentScore

class EnsembleSentimentProcessor:
    """Ensemble sentiment processor using multiple models"""

    def __init__(self):
        self.finbert_weight = NLP_CONFIG.FINBERT_WEIGHT
        self.vader_weight = NLP_CONFIG.VADER_WEIGHT
        self.textblob_weight = NLP_CONFIG.TEXTBLOB_WEIGHT
        self.time_decay_factor = NLP_CONFIG.TIME_DECAY_FACTOR

        # Initialize analyzers
        self.finbert = FinBERTAnalyzer()
        self.vader = SentimentIntensityAnalyzer()

        # Sentiment storage for aggregation
        self.sentiment_history = defaultdict(list)
        self.history_lock = threading.Lock()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Ensemble sentiment processor initialized")

    def _analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader.polarity_scores(text)
            return {
                'positive': max(0, scores['compound']) if scores['compound'] > 0 else 0,
                'negative': abs(min(0, scores['compound'])) if scores['compound'] < 0 else 0,
                'neutral': 1 - abs(scores['compound']),
                'compound': scores['compound']
            }
        except Exception as e:
            self.logger.error(f"VADER analysis error: {e}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'compound': 0.0}

    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            # Convert to positive/negative/neutral format
            if polarity > 0:
                positive = polarity
                negative = 0
                neutral = 1 - polarity
            elif polarity < 0:
                positive = 0
                negative = abs(polarity)
                neutral = 1 - abs(polarity)
            else:
                positive = 0.33
                negative = 0.33
                neutral = 0.34

            return {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'compound': polarity
            }
        except Exception as e:
            self.logger.error(f"TextBlob analysis error: {e}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'compound': 0.0}

    def _ensemble_scores(self, finbert_scores: Dict[str, float],
                        vader_scores: Dict[str, float],
                        textblob_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine scores from all three models"""
        ensemble = {}

        for key in ['positive', 'negative', 'neutral', 'compound']:
            weighted_score = (
                finbert_scores[key] * self.finbert_weight +
                vader_scores[key] * self.vader_weight +
                textblob_scores[key] * self.textblob_weight
            )
            ensemble[key] = weighted_score

        # Normalize positive, negative, neutral to sum to 1
        total = ensemble['positive'] + ensemble['negative'] + ensemble['neutral']
        if total > 0:
            ensemble['positive'] /= total
            ensemble['negative'] /= total
            ensemble['neutral'] /= total

        return ensemble

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze a single text using ensemble method"""
        # Get scores from all models
        finbert_scores = self.finbert.analyze_single(text)
        vader_scores = self._analyze_vader(text)
        textblob_scores = self._analyze_textblob(text)

        # Combine using ensemble weights
        ensemble_scores = self._ensemble_scores(finbert_scores, vader_scores, textblob_scores)

        # Add confidence score based on agreement between models
        agreements = []
        for key in ['positive', 'negative', 'neutral']:
            std_dev = np.std([finbert_scores[key], vader_scores[key], textblob_scores[key]])
            agreements.append(1 - std_dev)  # Lower std_dev = higher agreement

        ensemble_scores['confidence'] = np.mean(agreements)

        return ensemble_scores

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze a batch of texts"""
        if not texts:
            return []

        # Get FinBERT scores for the batch
        finbert_results = self.finbert.analyze_sentiment(texts)

        # Get VADER and TextBlob scores
        vader_results = [self._analyze_vader(text) for text in texts]
        textblob_results = [self._analyze_textblob(text) for text in texts]

        # Combine all results
        ensemble_results = []
        for finbert, vader, textblob in zip(finbert_results, vader_results, textblob_results):
            ensemble = self._ensemble_scores(finbert, vader, textblob)

            # Calculate confidence
            agreements = []
            for key in ['positive', 'negative', 'neutral']:
                std_dev = np.std([finbert[key], vader[key], textblob[key]])
                agreements.append(1 - std_dev)

            ensemble['confidence'] = np.mean(agreements)
            ensemble_results.append(ensemble)

        return ensemble_results

    def process_data_points(self, data_points: List[DataPoint]) -> List[SentimentScore]:
        """Process a list of data points and return sentiment scores"""
        if not data_points:
            return []

        texts = [dp.content for dp in data_points]
        sentiment_results = self.analyze_batch(texts)

        sentiment_scores = []
        for data_point, sentiment in zip(data_points, sentiment_results):
            score = SentimentScore(
                timestamp=data_point.timestamp,
                symbol=data_point.symbol,
                positive=sentiment['positive'],
                negative=sentiment['negative'],
                neutral=sentiment['neutral'],
                compound=sentiment['compound'],
                confidence=sentiment['confidence']
            )
            sentiment_scores.append(score)

        return sentiment_scores

    def _apply_time_decay(self, scores: List[SentimentScore], reference_time: datetime) -> List[SentimentScore]:
        """Apply time decay to sentiment scores"""
        decayed_scores = []

        for score in scores:
            time_diff = (reference_time - score.timestamp).total_seconds() / 3600  # hours
            decay_factor = self.time_decay_factor ** time_diff

            # Apply decay to all sentiment components
            decayed_score = SentimentScore(
                timestamp=score.timestamp,
                symbol=score.symbol,
                positive=score.positive * decay_factor,
                negative=score.negative * decay_factor,
                neutral=score.neutral * decay_factor,
                compound=score.compound * decay_factor,
                confidence=score.confidence * decay_factor,
                source_count=score.source_count
            )
            decayed_scores.append(decayed_score)

        return decayed_scores

    def aggregate_sentiment(self, symbol: str, time_window_hours: int = 1) -> Optional[SentimentScore]:
        """Aggregate sentiment for a symbol over a time window"""
        with self.history_lock:
            if symbol not in self.sentiment_history:
                return None

            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=time_window_hours)

            # Filter recent scores
            recent_scores = [
                score for score in self.sentiment_history[symbol]
                if score.timestamp >= cutoff_time
            ]

            if not recent_scores:
                return None

            # Apply time decay
            decayed_scores = self._apply_time_decay(recent_scores, current_time)

            # Aggregate
            total_weight = sum(score.confidence for score in decayed_scores)

            if total_weight == 0:
                return None

            weighted_positive = sum(score.positive * score.confidence for score in decayed_scores) / total_weight
            weighted_negative = sum(score.negative * score.confidence for score in decayed_scores) / total_weight
            weighted_neutral = sum(score.neutral * score.confidence for score in decayed_scores) / total_weight
            weighted_compound = sum(score.compound * score.confidence for score in decayed_scores) / total_weight

            avg_confidence = np.mean([score.confidence for score in decayed_scores])

            return SentimentScore(
                timestamp=current_time,
                symbol=symbol,
                positive=weighted_positive,
                negative=weighted_negative,
                neutral=weighted_neutral,
                compound=weighted_compound,
                confidence=avg_confidence,
                source_count=len(recent_scores)
            )

    def update_sentiment_history(self, sentiment_scores: List[SentimentScore]):
        """Update sentiment history with new scores"""
        with self.history_lock:
            for score in sentiment_scores:
                if score.symbol:
                    self.sentiment_history[score.symbol].append(score)

                    # Keep only recent history (24 hours)
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.sentiment_history[score.symbol] = [
                        s for s in self.sentiment_history[score.symbol]
                        if s.timestamp >= cutoff_time
                    ]

    def get_sentiment_momentum(self, symbol: str, short_window: int = 1, long_window: int = 4) -> float:
        """Calculate sentiment momentum (short MA - long MA)"""
        short_sentiment = self.aggregate_sentiment(symbol, short_window)
        long_sentiment = self.aggregate_sentiment(symbol, long_window)

        if short_sentiment and long_sentiment:
            return short_sentiment.compound - long_sentiment.compound

        return 0.0

# Test function
def test_ensemble_processor():
    """Test the ensemble sentiment processor"""
    processor = EnsembleSentimentProcessor()

    test_texts = [
        "Bitcoin is showing incredible bullish momentum!",
        "The market crash is devastating for all investors",
        "Steady growth in the tech sector this quarter",
        "HODL! Diamond hands! To the moon!",
        "Bearish sentiment dominates cryptocurrency markets"
    ]

    print("Testing ensemble sentiment processor...")
    start_time = datetime.now()

    results = processor.analyze_batch(test_texts)

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    print(f"\nProcessed {len(test_texts)} texts in {processing_time:.3f} seconds")
    print(f"Average time per text: {processing_time/len(test_texts)*1000:.1f}ms")

    for text, result in zip(test_texts, results):
        print(f"\nText: {text}")
        print(f"Positive: {result['positive']:.3f}, Negative: {result['negative']:.3f}, Neutral: {result['neutral']:.3f}")
        print(f"Compound: {result['compound']:.3f}, Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    test_ensemble_processor()
