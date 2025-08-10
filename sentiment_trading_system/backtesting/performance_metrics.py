
import numpy as np
import pandas as pd

def calculate_performance(results, data, risk_free_rate=0.02):
    if results is None or results.empty or len(data) < 2: return {"Error": "Not enough data."}
    if results['total'].iloc[0] == 0: return {"Error": "Initial capital is zero."}
    daily_returns = results['total'].pct_change().dropna()
    if daily_returns.empty: return {"Total Return": "0.00%", "Annualized Return": "0.00%", "Max Drawdown": "0.00%"}
    total_return = (results['total'].iloc[-1] / results['total'].iloc[0]) - 1
    days = (results['Date'].iloc[-1] - results['Date'].iloc[0]).days
    annualized_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0
    annualized_volatility = daily_returns.std() * np.sqrt(365)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(365)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
    last_close, first_close = float(data['Close'].iloc[-1]), float(data['Close'].iloc[0])
    buy_hold_return = (last_close / first_close) - 1 if first_close > 0 else 0.0
    return {"Total Return": f"{total_return:.2%}", "Annualized Return": f"{annualized_return:.2%}", "Max Drawdown": f"{max_drawdown:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}", "Sortino Ratio": f"{sortino_ratio:.2f}", "Buy & Hold Return": f"{buy_hold_return:.2%}"}
