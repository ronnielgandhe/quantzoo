#!/usr/bin/env python3

import pandas as pd
import sys
sys.path.append('/Users/ronniel/quantzoo')

from quantzoo.backtest.engine import BacktestEngine, BacktestConfig

class DebugStrategy:
    def on_start(self, ctx):
        self.entered = False
        print("Strategy started")
    
    def on_bar(self, ctx, bar):
        bar_idx = ctx.bar_index()
        print(f"Bar {bar_idx}: close={bar.close}, position={ctx.engine.position.size}")
        
        # Enter at bar 5
        if not self.entered and bar_idx == 5:
            print(f"Entering long at bar {bar_idx}, price {bar.close}")
            ctx.buy(1, "Long")
            self.entered = True
            print(f"After entry: position={ctx.engine.position.size}, entry_bar={ctx.engine.entry_bar}, entry_price={ctx.engine.entry_price}")

# Create test data
dates = pd.date_range('2023-01-01 09:00', periods=10, freq='15min')
prices = [100, 100, 100, 100, 100, 100, 99, 98, 97, 96]  # Drop after bar 5
data = pd.DataFrame({
    'open': prices,
    'high': [p + 0.1 for p in prices],
    'low': [p - 0.1 for p in prices],
    'close': prices,
    'volume': [1000] * 10
}, index=dates)

print("Data:")
print(data[['close']])

config = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
engine = BacktestEngine(config)
strategy = DebugStrategy()
result = engine.run(data, strategy)

print(f"\nResult: {len(result['trades'])} trades")
for i, trade in enumerate(result['trades']):
    print(f"Trade {i}: {trade}")