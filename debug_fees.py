#!/usr/bin/env python3

import pandas as pd
import sys
sys.path.append('/Users/ronniel/quantzoo')

from quantzoo.backtest.engine import BacktestEngine, BacktestConfig

class SimpleBuyHoldStrategy:
    def on_start(self, ctx):
        pass
    
    def on_bar(self, ctx, bar):
        if ctx.bar_index() == 1:  # Buy on second bar
            ctx.buy(1, "Long")
        elif ctx.bar_index() == 18:  # Sell near end
            ctx.close_position("Exit")

# Test data with price movement
dates = pd.date_range('2023-01-01', periods=20, freq='15min')
prices = [100 + i * 0.1 for i in range(20)]  # Slight uptrend

data = pd.DataFrame({
    'open': prices,
    'high': [p + 0.5 for p in prices],
    'low': [p - 0.5 for p in prices],
    'close': prices,
    'volume': [1000] * 20
}, index=dates)

print("Data sample:")
print(data[['close']].head(5))
print("...")
print(data[['close']].tail(5))

config = BacktestConfig(
    initial_capital=10000,
    fees_bps=10,  # 0.1% fees  
    slippage_bps=5,   # 0.05% slippage
    seed=42
)

engine = BacktestEngine(config)
strategy = SimpleBuyHoldStrategy()
result = engine.run(data, strategy)

print(f"\nTrades: {len(result['trades'])}")
for trade in result['trades']:
    print(f"Trade: entry_price={trade.entry_price}, exit_price={trade.exit_price}")
    print(f"       quantity={trade.quantity}, side={trade.side}")
    print(f"       pnl={trade.pnl}, fees={trade.fees}, slippage={trade.slippage}")
    
    # Calculate what test expects
    gross_pnl = trade.quantity * (trade.exit_price - trade.entry_price)
    transaction_costs = trade.fees + trade.slippage
    expected_net_pnl = gross_pnl - transaction_costs
    
    print(f"       gross_pnl={gross_pnl}")
    print(f"       transaction_costs={transaction_costs}")
    print(f"       expected_net_pnl={expected_net_pnl}")
    print(f"       actual_pnl={trade.pnl}")
    print(f"       difference={trade.pnl - expected_net_pnl}")