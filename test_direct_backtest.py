#!/usr/bin/env python3
"""Direct test of the strategy without walk-forward analysis."""

import pandas as pd
import yaml
from quantzoo.strategies.mnq_808 import MNQ808, MNQ808Params
from quantzoo.backtest.engine import BacktestEngine, BacktestConfig
from quantzoo.data.loaders import load_csv_ohlcv
from quantzoo.metrics.core import calculate_metrics

def test_direct_backtest():
    """Test the strategy directly with the backtest engine."""
    
    # Load data
    data = load_csv_ohlcv('crossover_test_data.csv', tz=None, timeframe='15m')
    print(f"Loaded {len(data)} bars of data")
    
    # Setup strategy with exact config
    params = MNQ808Params(
        atr_mult=1.5,
        lookback=10,
        use_mfi=True,
        trail_mult_legacy=1.0,
        contracts=1,
        risk_ticks_legacy=150,
        session_start="08:00",
        session_end="16:30",
        tick_size=0.25,
        tick_value=0.5,
        treat_atr_as_ticks=True
    )
    
    strategy = MNQ808(params)
    
    # Setup backtest config
    config = BacktestConfig(
        initial_capital=100000,
        fees_bps=1.0,
        slippage_bps=1.0,
        seed=123
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run(data, strategy)
    
    print(f"\n=== DIRECT BACKTEST RESULTS ===")
    print(f"Total trades: {len(results.get('trades', []))}")
    
    trades = results.get('trades', [])
    if trades:
        print(f"First trade: {trades[0]}")
        
        # Calculate metrics
        equity_curve = results.get('equity_curve', [])
        metrics = calculate_metrics(trades, equity_curve)
        
        print(f"\nMetrics:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.3f}")
    else:
        print("No trades executed in direct backtest!")

if __name__ == "__main__":
    test_direct_backtest()