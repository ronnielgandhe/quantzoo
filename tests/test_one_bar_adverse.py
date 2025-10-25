"""Test one bar adverse exit rule."""

import pytest
import pandas as pd
import numpy as np
from quantzoo.backtest.engine import BacktestEngine, BacktestConfig, StrategyContext


class OneBarAdverseTestStrategy:
    """Strategy to test one bar adverse exit rule."""
    
    def on_start(self, ctx):
        self.entered = False
        self.entry_bar = None
    
    def on_bar(self, ctx, bar):
        # Enter long position at bar 5
        if not self.entered and ctx.bar_index() == 5:
            ctx.buy(1, "Long")
            self.entered = True
            self.entry_bar = ctx.bar_index()


class ManualExitStrategy:
    """Strategy that manually implements one bar adverse exit."""
    
    def on_start(self, ctx):
        self.entered = False
        self.entry_bar = None
        self.entry_price = None
    
    def on_bar(self, ctx, bar):
        # Enter position
        if not self.entered and ctx.bar_index() == 5:
            ctx.buy(1, "Long")
            self.entered = True
            self.entry_bar = ctx.bar_index()
            self.entry_price = ctx.close
        
        # Check one bar adverse exit manually
        elif (self.entered and 
              ctx.bar_index() == self.entry_bar + 1 and 
              ctx.close < self.entry_price):
            ctx.close_position("OneBarAdverse")
            self.entered = False


def test_long_one_bar_adverse_exit():
    """Test one bar adverse exit for long position."""
    # Create test data where price drops on the bar after entry
    dates = pd.date_range('2023-01-01 09:00', periods=10, freq='15T')
    prices = [100, 100, 100, 100, 100, 100, 99, 98, 97, 96]  # Drop after bar 5 (entry)
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'close': prices,
        'volume': [1000] * 10
    }, index=dates)
    
    # Run backtest
    config = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
    engine = BacktestEngine(config)
    strategy = OneBarAdverseTestStrategy()
    result = engine.run(data, strategy)
    
    # Should have one trade that was closed by adverse exit
    assert len(result["trades"]) == 1, f"Expected 1 trade, got {len(result['trades'])}"
    
    trade = result["trades"][0]
    
    # Trade should be closed at bar 6 (one bar after entry at bar 5)
    entry_time = data.index[5]
    exit_time = data.index[6]
    
    assert trade.entry_time == entry_time, f"Entry time mismatch: {trade.entry_time} vs {entry_time}"
    assert trade.exit_time == exit_time, f"Exit time mismatch: {trade.exit_time} vs {exit_time}"
    assert trade.exit_reason == "OneBarAdverse", f"Exit reason should be OneBarAdverse, got {trade.exit_reason}"
    
    # Trade should be unprofitable (price dropped from 100 to 99)
    assert trade.pnl < 0, f"Trade should be unprofitable, got PnL: {trade.pnl}"


def test_long_one_bar_favorable_no_exit():
    """Test that one bar adverse exit doesn't trigger when price goes up."""
    # Create test data where price rises on the bar after entry
    dates = pd.date_range('2023-01-01 09:00', periods=10, freq='15T')
    prices = [100, 100, 100, 100, 100, 100, 101, 102, 103, 104]  # Rise after bar 5
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'close': prices,
        'volume': [1000] * 10
    }, index=dates)
    
    # Run backtest
    config = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
    engine = BacktestEngine(config)
    strategy = OneBarAdverseTestStrategy()
    result = engine.run(data, strategy)
    
    # Should have no completed trades (position remains open)
    assert len(result["trades"]) == 0, f"Expected 0 completed trades, got {len(result['trades'])}"
    
    # Position should still be open
    assert not engine.position.is_flat(), "Position should still be open"
    assert engine.position.size == 1, f"Position size should be 1, got {engine.position.size}"


def test_short_one_bar_adverse_exit():
    """Test one bar adverse exit for short position."""
    
    class ShortTestStrategy:
        def on_start(self, ctx):
            self.entered = False
        
        def on_bar(self, ctx, bar):
            if not self.entered and ctx.bar_index() == 5:
                ctx.sell(1, "Short")
                self.entered = True
    
    # Create test data where price rises on the bar after entry (adverse for short)
    dates = pd.date_range('2023-01-01 09:00', periods=10, freq='15T')
    prices = [100, 100, 100, 100, 100, 100, 101, 102, 103, 104]  # Rise after bar 5
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'close': prices,
        'volume': [1000] * 10
    }, index=dates)
    
    # Run backtest
    config = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
    engine = BacktestEngine(config)
    strategy = ShortTestStrategy()
    result = engine.run(data, strategy)
    
    # Should have one trade closed by adverse exit
    assert len(result["trades"]) == 1, f"Expected 1 trade, got {len(result['trades'])}"
    
    trade = result["trades"][0]
    assert trade.exit_reason == "OneBarAdverse", f"Exit reason should be OneBarAdverse, got {trade.exit_reason}"
    assert trade.side == "short", f"Trade side should be short, got {trade.side}"
    assert trade.pnl < 0, f"Short trade should be unprofitable when price rises, got PnL: {trade.pnl}"


def test_one_bar_adverse_timing():
    """Test that one bar adverse exit only triggers on the next bar."""
    # Create test data with adverse move on bar after entry
    dates = pd.date_range('2023-01-01 09:00', periods=15, freq='15T')
    prices = [100] * 6 + [99] + [98] * 8  # Drop at bar 6, continue dropping
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'close': prices,
        'volume': [1000] * 15
    }, index=dates)
    
    config = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
    engine = BacktestEngine(config)
    strategy = OneBarAdverseTestStrategy()
    result = engine.run(data, strategy)
    
    # Should exit exactly at bar 6 (one bar after entry at bar 5)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    
    # Exit should be at bar 6, not later
    expected_exit_time = data.index[6]
    assert trade.exit_time == expected_exit_time, \
        f"Exit should be at bar 6: {expected_exit_time}, got {trade.exit_time}"


def test_comparison_with_manual_implementation():
    """Compare engine's one bar adverse exit with manual implementation."""
    # Test data
    dates = pd.date_range('2023-01-01 09:00', periods=10, freq='15T')
    prices = [100, 100, 100, 100, 100, 100, 99, 98, 97, 96]
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + 0.1 for p in prices],
        'low': [p - 0.1 for p in prices],
        'close': prices,
        'volume': [1000] * 10
    }, index=dates)
    
    # Test with engine's built-in one bar adverse exit
    config1 = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
    engine1 = BacktestEngine(config1)
    result1 = engine1.run(data, OneBarAdverseTestStrategy())
    
    # Test with manual implementation
    config2 = BacktestConfig(seed=42, fees_bps=0, slippage_bps=0)
    engine2 = BacktestEngine(config2)
    result2 = engine2.run(data, ManualExitStrategy())
    
    # Both should produce the same result
    assert len(result1["trades"]) == len(result2["trades"])
    
    if result1["trades"]:
        trade1 = result1["trades"][0]
        trade2 = result2["trades"][0]
        
        assert trade1.exit_time == trade2.exit_time, "Exit times should match"
        assert abs(trade1.pnl - trade2.pnl) < 0.01, "PnL should match"


if __name__ == "__main__":
    pytest.main([__file__])