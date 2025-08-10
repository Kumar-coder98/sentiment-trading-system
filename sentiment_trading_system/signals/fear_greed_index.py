"""
Fear and Greed Index calculation based on market indicators
"""

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class FearGreedIndex:
    """Calculate Fear and Greed Index using multiple market indicators"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Index weights (should sum to 1.0)
        self.weights = {
            'market_momentum': 0.25,
            'stock_price_strength': 0.20,
            'stock_price_breadth': 0.20,
            'safe_haven_demand': 0.15,
            'junk_bond_demand': 0.10,
            'market_volatility': 0.10
        }

    def _get_market_momentum(self) -> float:
        """Calculate market momentum (S&P 500 vs 125-day MA)"""
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="200d")

            if len(hist) < 125:
                return 50.0  # Neutral

            current_price = hist['Close'].iloc[-1]
            ma_125 = hist['Close'].rolling(125).mean().iloc[-1]

            momentum_ratio = current_price / ma_125

            # Convert to 0-100 scale (1.0 = 50, 1.1 = ~75, 0.9 = ~25)
            score = 50 + (momentum_ratio - 1) * 250
            return np.clip(score, 0, 100)

        except Exception as e:
            self.logger.error(f"Error calculating market momentum: {e}")
            return 50.0

    def _get_stock_price_strength(self) -> float:
        """Calculate stock price strength (52-week highs vs lows)"""
        try:
            # Use a broad market ETF as proxy
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1y")

            if len(hist) < 250:
                return 50.0

            current_price = hist['Close'].iloc[-1]
            year_high = hist['High'].max()
            year_low = hist['Low'].min()

            # Position within 52-week range
            position = (current_price - year_low) / (year_high - year_low)

            return position * 100

        except Exception as e:
            self.logger.error(f"Error calculating stock price strength: {e}")
            return 50.0

    def _get_stock_price_breadth(self) -> float:
        """Calculate stock price breadth using advance/decline data"""
        try:
            # Use multiple ETFs as proxy for market breadth
            tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
            advancing = 0
            declining = 0

            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d")

                    if len(hist) >= 2:
                        today_close = hist['Close'].iloc[-1]
                        yesterday_close = hist['Close'].iloc[-2]

                        if today_close > yesterday_close:
                            advancing += 1
                        else:
                            declining += 1
                except:
                    continue

            total = advancing + declining
            if total > 0:
                breadth_ratio = advancing / total
                return breadth_ratio * 100

            return 50.0

        except Exception as e:
            self.logger.error(f"Error calculating stock price breadth: {e}")
            return 50.0

    def _get_safe_haven_demand(self) -> float:
        """Calculate safe haven demand (stocks vs treasuries)"""
        try:
            # Compare SPY (stocks) vs TLT (20+ year treasuries)
            spy = yf.Ticker("SPY")
            tlt = yf.Ticker("TLT")

            spy_hist = spy.history(period="30d")
            tlt_hist = tlt.history(period="30d")

            if len(spy_hist) < 20 or len(tlt_hist) < 20:
                return 50.0

            # Calculate 20-day performance
            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-20] - 1) * 100
            tlt_return = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[-20] - 1) * 100

            # Higher stock performance relative to bonds = less fear
            relative_performance = spy_return - tlt_return

            # Convert to 0-100 scale
            score = 50 + relative_performance * 2
            return np.clip(score, 0, 100)

        except Exception as e:
            self.logger.error(f"Error calculating safe haven demand: {e}")
            return 50.0

    def _get_junk_bond_demand(self) -> float:
        """Calculate junk bond demand using credit spreads"""
        try:
            # Use HYG (high yield bonds) vs TLT (treasuries) as proxy
            hyg = yf.Ticker("HYG")
            tlt = yf.Ticker("TLT")

            hyg_hist = hyg.history(period="30d")
            tlt_hist = tlt.history(period="30d")

            if len(hyg_hist) < 20 or len(tlt_hist) < 20:
                return 50.0

            # Calculate relative performance
            hyg_return = (hyg_hist['Close'].iloc[-1] / hyg_hist['Close'].iloc[-20] - 1) * 100
            tlt_return = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[-20] - 1) * 100

            # Higher junk bond performance relative to treasuries = less fear
            relative_performance = hyg_return - tlt_return

            score = 50 + relative_performance * 3
            return np.clip(score, 0, 100)

        except Exception as e:
            self.logger.error(f"Error calculating junk bond demand: {e}")
            return 50.0

    def _get_market_volatility(self) -> float:
        """Calculate market volatility using VIX"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="30d")

            if len(hist) < 1:
                return 50.0

            current_vix = hist['Close'].iloc[-1]

            # VIX interpretation: <20 = low fear, >30 = high fear
            # Invert VIX for greed scale (lower VIX = higher greed)
            if current_vix <= 15:
                score = 90
            elif current_vix <= 20:
                score = 70
            elif current_vix <= 25:
                score = 50
            elif current_vix <= 30:
                score = 30
            elif current_vix <= 35:
                score = 15
            else:
                score = 5

            return score

        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {e}")
            return 50.0

    def calculate_index(self) -> Dict[str, float]:
        """Calculate the complete Fear and Greed Index"""
        components = {
            'market_momentum': self._get_market_momentum(),
            'stock_price_strength': self._get_stock_price_strength(),
            'stock_price_breadth': self._get_stock_price_breadth(),
            'safe_haven_demand': self._get_safe_haven_demand(),
            'junk_bond_demand': self._get_junk_bond_demand(),
            'market_volatility': self._get_market_volatility()
        }

        # Calculate weighted score
        weighted_score = sum(
            components[component] * self.weights[component]
            for component in components
        )

        # Interpret score
        if weighted_score <= 25:
            interpretation = "Extreme Fear"
        elif weighted_score <= 45:
            interpretation = "Fear"
        elif weighted_score <= 55:
            interpretation = "Neutral"
        elif weighted_score <= 75:
            interpretation = "Greed"
        else:
            interpretation = "Extreme Greed"

        return {
            'score': weighted_score,
            'interpretation': interpretation,
            'components': components,
            'timestamp': datetime.now().isoformat()
        }

    def get_historical_average(self, days: int = 30) -> float:
        """Get historical average (simulated for now)"""
        # In a real implementation, this would calculate historical averages
        # For now, return a reasonable baseline
        return 50.0

    def normalize_score(self, raw_score: float, historical_avg: float) -> float:
        """Normalize score based on historical context"""
        # Simple normalization - could be enhanced with z-scores
        normalized = raw_score + (raw_score - historical_avg) * 0.1
        return np.clip(normalized, 0, 100)

def test_fear_greed_index():
    """Test the Fear and Greed Index calculation"""
    fgi = FearGreedIndex()

    print("Calculating Fear and Greed Index...")
    start_time = datetime.now()

    result = fgi.calculate_index()

    end_time = datetime.now()
    calculation_time = (end_time - start_time).total_seconds()

    print(f"\nFear and Greed Index calculated in {calculation_time:.3f} seconds")
    print(f"\nOverall Score: {result['score']:.1f}/100")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Timestamp: {result['timestamp']}")

    print("\nComponent Breakdown:")
    for component, score in result['components'].items():
        print(f"  {component.replace('_', ' ').title()}: {score:.1f}")

if __name__ == "__main__":
    test_fear_greed_index()
