#!/usr/bin/env python3
"""Debug script to test the MNQ808 strategy with verbose logging."""

import pandas as pd
import numpy as np
from quantzoo.strategies.mnq_808 import MNQ808, MNQ808Params
from quantzoo.backtest.engine import BacktestEngine, BacktestConfig
from quantzoo.data.loaders import load_csv_ohlcv

def debug_strategy():
    """Run the strategy with debug output."""
    
    # Load data
    data = load_csv_ohlcv('/Users/ronniel/quantzoo/crossover_test_data.csv', tz=None, timeframe='15m')
    
    print(f"Loaded data: {len(data)} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    print(f"Columns: {list(data.columns)}")
    print("\nFirst 5 bars:")
    print(data.head())
    
    # Setup strategy
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
    
    # Create strategy instance and add debug
    strategy = MNQ808(params)
    original_on_bar = strategy.on_bar
    
    bar_count = 0
    signal_count = 0
    
    def debug_on_bar(ctx, bar):
        nonlocal bar_count, signal_count
        bar_count += 1
        
        # Call original method
        original_on_bar(ctx, bar)
        
        # Debug output every 5 bars
        if bar_count % 5 == 0:
            print(f"\nBar {bar_count}: {bar.name}")
            print(f"  Price: O={ctx.open:.2f} H={ctx.high:.2f} L={ctx.low:.2f} C={ctx.close:.2f}")
            print(f"  Position: {ctx.position_size()}")
            print(f"  In session: {ctx.in_session(params.session_start, params.session_end)}")
            print(f"  Bar confirmed: {ctx.bar_confirmed()}")
            
            if len(strategy.anchor_history) > 0:
                print(f"  Anchor: {strategy.anchor_history[-1]:.2f}")
                if len(strategy.momentum_history) > 0:
                    print(f"  Momentum: {strategy.momentum_history[-1]:.2f}")
                if len(strategy.upper_band_history) > 0:
                    print(f"  Upper band: {strategy.upper_band_history[-1]:.2f}")
                if len(strategy.lower_band_history) > 0:
                    print(f"  Lower band: {strategy.lower_band_history[-1]:.2f}")
                    
                # Check crossover conditions if we have enough data
                if len(strategy.anchor_history) >= 3:
                    anchor_lag2 = strategy.anchor_history[-3]
                    close_prev = ctx.get_series("close", -1) if ctx.bar_index() > 0 else ctx.close
                    print(f"  Close prev: {close_prev:.2f}, Anchor[2]: {anchor_lag2:.2f}")
                    is_long = close_prev <= anchor_lag2 and ctx.close > anchor_lag2
                    is_short = close_prev >= anchor_lag2 and ctx.close < anchor_lag2
                    print(f"  Long signal: {is_long}, Short signal: {is_short}")
    
    strategy.on_bar = debug_on_bar
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        fees_bps=1.0,
        slippage_bps=1.0,
        seed=123
    )
    
    engine = BacktestEngine(config)
    
    results = engine.run(data, strategy)
    
    print(f"\n=== RESULTS ===")
    print(f"Total bars processed: {bar_count}")
    print(f"Total trades: {len(results.get('trades', []))}")
    print(f"Final equity: {results.get('metrics', {}).get('total_return', 0):.2%}")
    
    trades = results.get('trades', [])
    if len(trades) > 0:
        print(f"First trade: {trades[0]}")
    else:
        print("No trades executed!")
        
        # Additional debugging for no trades
        print(f"\nStrategy state:")
        print(f"  Anchor history length: {len(strategy.anchor_history)}")
        print(f"  Momentum history length: {len(strategy.momentum_history)}")
        
        if len(strategy.anchor_history) >= 3:
            print(f"  Last 3 anchors: {strategy.anchor_history[-3:]}")
        if len(strategy.momentum_history) >= 3:
            print(f"  Last 3 momentum: {strategy.momentum_history[-3:]}")

if __name__ == "__main__":
    debug_strategy()