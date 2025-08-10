
from data_preparation import prepare_backtest_data
from backtester import Backtester
from performance_metrics import calculate_performance
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

CONFIG = {"ticker": 'BTC-USD', "initial_capital": 10000.0, "buy_threshold": 0.5, "sell_threshold": -0.5, "random_seed": 101}

def main():
    print("--- Starting Backtest with Configuration ---")
    print(CONFIG)
    try:
        data = prepare_backtest_data(CONFIG["ticker"], random_seed=CONFIG["random_seed"])
        print("Data preparation successful.\n")
    except ValueError as e:
        print(e); return
    backtester = Backtester(data, initial_capital=CONFIG["initial_capital"])
    print("Running backtest simulation...")
    backtester.run_backtest(buy_threshold=CONFIG["buy_threshold"], sell_threshold=CONFIG["sell_threshold"])
    print("Backtest simulation complete.")
    
    print("\n--- Performance Results ---")
    performance = calculate_performance(backtester.results, backtester.data)
    for metric, value in performance.items():
        print(f"{metric}: {value}")
        
    print("\n--- Key Trading Activity ---")
    
    # --- DEFINITIVE FIX: Use direct assignment instead of join ---
    # Create a copy to work with
    merged_data = backtester.results.copy()
    # Directly add the columns. This is guaranteed to work because the indices are identical.
    merged_data['sentiment'] = backtester.data['sentiment']
    merged_data['signal'] = backtester.data['signal']
    
    trade_activity = merged_data[merged_data['signal'] != 0]
    
    if not trade_activity.empty:
        # Display all trade rows without truncation, formatting floats for readability
        with pd.option_context('display.max_rows', None, 'display.float_format', '{:,.2f}'.format):
            # Select columns for a cleaner display
            display_cols = ['Date', 'holdings', 'cash', 'total', 'sentiment', 'signal']
            print(trade_activity[display_cols].to_string())
    else:
        print("No trades were executed in this backtest.")
    
    print("\n--- Final Portfolio State (Last 5 Days) ---")
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(backtester.results.tail().to_string())
    
    results_save_path = '/content/sentiment_trading_system/backtesting/backtesting_results.csv'
    backtester.results.to_csv(results_save_path, index=False)
    print(f"\nFull backtesting results saved to {results_save_path}")
    
    print("\nGenerating performance plot...")
    plot_save_path = '/content/sentiment_trading_system/backtesting/backtesting_result.png'
    backtester.plot_performance(CONFIG["ticker"], save_path=plot_save_path)

if __name__ == "__main__":
    main()
