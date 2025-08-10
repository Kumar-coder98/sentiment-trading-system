
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

class Backtester:
    def __init__(self, data, initial_capital=10000, commission=0.001):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None

    def _generate_signals(self, buy_threshold, sell_threshold):
        self.data['sentiment_momentum'] = self.data['sentiment'].diff(5)
        self.data['signal'] = 0
        buy_conditions = ((self.data['sentiment'] > buy_threshold) & (self.data['confidence'] > 0.6) & (self.data['sentiment_momentum'] > 0))
        self.data.loc[buy_conditions, 'signal'] = 1
        sell_conditions = ((self.data['sentiment'] < sell_threshold) & (self.data['confidence'] > 0.6) & (self.data['sentiment_momentum'] < 0))
        self.data.loc[sell_conditions, 'signal'] = -1
        self.data['signal'] = self.data['signal'].shift(1).fillna(0)

    def run_backtest(self, buy_threshold=0.7, sell_threshold=-0.7):
        self._generate_signals(buy_threshold, sell_threshold)
        signals = self.data['signal'].to_numpy()
        opens = self.data['Open'].to_numpy()
        closes = self.data['Close'].to_numpy()
        n_days = len(self.data)
        holdings, cash, total = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
        cash[0], total[0] = self.initial_capital, self.initial_capital
        position = 0
        for i in range(1, n_days):
            cash[i], holdings[i] = cash[i-1], holdings[i-1]
            signal, current_open, current_close, prev_close = signals[i], opens[i], closes[i], closes[i-1]
            traded_today = False
            if signal == 1 and position == 0 and current_open > 0:
                shares_to_buy = cash[i] / current_open
                cost = shares_to_buy * current_open * (1 + self.commission)
                cash[i] -= cost
                holdings[i] = shares_to_buy * current_close
                position, traded_today = 1, True
            elif signal == -1 and position == 1 and prev_close > 0:
                sale_proceeds = holdings[i-1] * (current_open / prev_close) * (1 - self.commission)
                cash[i] += sale_proceeds
                holdings[i] = 0
                position, traded_today = 0, True
            if not traded_today and position == 1 and prev_close > 0:
                holdings[i] = holdings[i-1] * (current_close / prev_close)
            total[i] = cash[i] + holdings[i]
        self.results = pd.DataFrame({'holdings': holdings, 'cash': cash, 'total': total, 'Date': self.data['Date']}, index=self.data.index)

    def plot_performance(self, ticker_symbol, save_path):
        if self.results is None: raise Exception("Backtest has not been run yet.")
        style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(self.results['Date'], self.results['total'], label='Strategy Equity Curve', color='royalblue', linewidth=2)
        buy_signals = self.data[self.data['signal'] == 1]
        if not buy_signals.empty:
            ax.plot(buy_signals['Date'], self.results.loc[buy_signals.index]['total'], '^', markersize=10, color='green', label='Buy Signal', alpha=0.8, linestyle='None')
        sell_signals = self.data[self.data['signal'] == -1]
        if not sell_signals.empty:
            ax.plot(sell_signals['Date'], self.results.loc[sell_signals.index]['total'], 'v', markersize=10, color='red', label='Sell Signal', alpha=0.8, linestyle='None')
        ax.set_title(f'Sentiment Strategy Performance vs. {ticker_symbol}', fontsize=20)
        ax.set_ylabel('Portfolio Value ($)', fontsize=15)
        ax.set_xlabel('Date', fontsize=15)
        ax.legend(loc='upper left', fontsize=12)
        ax2 = ax.twinx()
        ax2.plot(self.data['Date'], self.data['Close'], label=f'{ticker_symbol} Price', color='grey', linestyle='--', alpha=0.6)
        ax2.set_ylabel(f'{ticker_symbol} Price ($)', fontsize=15)
        ax2.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # Close the plot to free memory
        print(f"\nPlot saved to {save_path}")
