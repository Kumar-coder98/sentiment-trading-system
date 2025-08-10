"""
FinBERT-based sentiment analysis for financial text
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime

from ..config.settings import NLP_CONFIG

class FinBERTAnalyzer:
    """FinBERT sentiment analysis with GPU acceleration"""

    def __init__(self):
        self.device = torch.device(NLP_CONFIG.DEVICE)
        self.model_name = NLP_CONFIG.FINBERT_MODEL
        self.max_length = NLP_CONFIG.MAX_LENGTH
        self.batch_size = NLP_CONFIG.BATCH_SIZE

        self.tokenizer = None
        self.model = None
        self.labels = ['positive', 'negative', 'neutral']

        self._load_model()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"FinBERT model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {e}")
            self.logger.info("Falling back to default configuration")
            # Fallback: could implement a simpler model here
            raise e

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for FinBERT"""
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = text.strip()

        # Remove URLs
        import re
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Truncate if too long (leave room for special tokens)
        if len(text) > self.max_length - 10:
            text = text[:self.max_length - 10]

        return text

    def _analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze a batch of texts with FinBERT"""
        if not texts:
            return []

        try:
            # Preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]

            # Tokenize
            encoding = self.tokenizer(
                cleaned_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to numpy and extract results
            predictions_np = predictions.cpu().numpy()

            results = []
            for pred in predictions_np:
                result = {}
                for i, label in enumerate(self.labels):
                    result[label] = float(pred[i])

                # Calculate compound score
                result['compound'] = result['positive'] - result['negative']

                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error in FinBERT batch analysis: {e}")
            # Return neutral sentiment as fallback
            return [{'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'compound': 0.0}] * len(texts)

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for a list of texts"""
        if not texts:
            return []

        all_results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._analyze_batch(batch_texts)
            all_results.extend(batch_results)

        return all_results

    def analyze_single(self, text: str) -> Dict[str, float]:
        """Analyze sentiment for a single text"""
        results = self.analyze_sentiment([text])
        return results[0] if results else {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'compound': 0.0}

    def get_device_info(self) -> Dict[str, str]:
        """Get information about the device being used"""
        return {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }

# Test function
def test_finbert():
    """Test FinBERT analyzer"""
    analyzer = FinBERTAnalyzer()

    test_texts = [
        "The stock market is showing strong bullish momentum today",
        "Cryptocurrency prices are crashing and investors are panicking",
        "The company reported neutral earnings results for this quarter",
        "Bitcoin to the moon! Best investment ever!",
        "Market volatility is concerning for long-term investors"
    ]

    print("FinBERT Device Info:", analyzer.get_device_info())

    start_time = datetime.now()
    results = analyzer.analyze_sentiment(test_texts)
    end_time = datetime.now()

    print(f"\nAnalyzed {len(test_texts)} texts in {(end_time - start_time).total_seconds():.3f} seconds")

    for text, result in zip(test_texts, results):
        print(f"\nText: {text}")
        print(f"Sentiment: {result}")

if __name__ == "__main__":
    test_finbert()
