"""Test fees and slippage application."""

import pytest
import pandas as pd
import numpy as np
from quantzoo.backtest.engine import BacktestEngine, BacktestConfig, StrategyContext


class SimpleBuyHoldStrategy:
    """Simple strategy for testing fees and slippage."""
    
    def on_start(self, ctx):
        self.entered = False
    
    def on_bar(self, ctx, bar):
        if not self.entered and ctx.bar_index() == 5:
            ctx.buy(1, "Entry")
            self.entered = True
        elif self.entered and ctx.bar_index() == 10:
            ctx.close_position("Exit")


def test_fees_application():
    """Test that trading fees are correctly applied."""
    # Create test data
    dates = pd.date_range('2023-01-01', periods=20, freq='15T')
    data = pd.DataFrame({
        'open': [100] * 20,
        'high': [100.5] * 20,
        'low': [99.5] * 20,
        'close': [100] * 20,
        'volume': [1000] * 20
    }, index=dates)
    
    # Test with known fees
    config = BacktestConfig(
        initial_capital=10000,
        fees_bps=100,  # 1% fees for easy calculation
        slippage_bps=0,  # No slippage for this test
        seed=42
    )
    
    engine = BacktestEngine(config)
    strategy = SimpleBuyHoldStrategy()
    result = engine.run(data, strategy)
    
    # Should have exactly 2 trades (entry and exit)
    assert len(result["trades"]) == 1, f"Expected 1 completed trade, got {len(result['trades'])}"
    
    trade = result["trades"][0]
    
    # Check fees calculation
    # Entry: 1 share * $100 * 1% = $1
    # Exit: 1 share * $100 * 1% = $1
    # Total fees should be $2
    expected_fees = 2.0  # $1 entry + $1 exit
    assert abs(trade.fees - expected_fees) < 0.1, f"Expected fees {expected_fees}, got {trade.fees}"


def test_slippage_application():
    """Test that slippage is randomly applied."""
    # Create test data
    dates = pd.date_range('2023-01-01', periods=20, freq='15T')
    data = pd.DataFrame({
        'open': [100] * 20,
        'high': [101] * 20,
        'low': [99] * 20,
        'close': [100] * 20,
        'volume': [1000] * 20
    }, index=dates)
    
    # Test with known slippage
    config = BacktestConfig(
        initial_capital=10000,
        fees_bps=0,  # No fees for this test
        slippage_bps=100,  # 1% slippage
        seed=42
    )
    
    engine = BacktestEngine(config)
    strategy = SimpleBuyHoldStrategy()
    result = engine.run(data, strategy)
    
    # Should have trades with slippage applied
    assert len(result["trades"]) == 1, f"Expected 1 completed trade, got {len(result['trades'])}"
    
    trade = result["trades"][0]
    
    # Slippage should be non-zero (unless extremely unlucky)
    assert trade.slippage > 0, f"Expected positive slippage, got {trade.slippage}"
    
    # Slippage should be reasonable (not more than 2% of notional)
    notional = abs(trade.quantity * trade.entry_price)
    max_expected_slippage = notional * 0.02  # 2% max
    assert trade.slippage <= max_expected_slippage, f"Slippage too high: {trade.slippage}"


def test_combined_fees_and_slippage():
    """Test fees and slippage working together."""
    # Create test data with price movement
    dates = pd.date_range('2023-01-01', periods=20, freq='15T')
    prices = [100 + i * 0.1 for i in range(20)]  # Slight uptrend
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [1000] * 20
    }, index=dates)
    
    # Test with both fees and slippage
    config = BacktestConfig(
        initial_capital=10000,
        fees_bps=10,  # 0.1% fees
        slippage_bps=5,   # 0.05% slippage
        seed=42
    )
    
    engine = BacktestEngine(config)
    strategy = SimpleBuyHoldStrategy()
    result = engine.run(data, strategy)
    
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    
    # Both fees and slippage should be applied
    assert trade.fees > 0, "Fees should be positive"
    assert trade.slippage > 0, "Slippage should be positive"
    
    # Total transaction costs should reduce PnL
    gross_pnl = trade.quantity * (trade.exit_price - trade.entry_price)
    net_pnl = trade.pnl
    transaction_costs = trade.fees + trade.slippage
    
    # Net PnL should equal gross PnL minus transaction costs
    expected_net_pnl = gross_pnl - transaction_costs
    assert abs(net_pnl - expected_net_pnl) < 0.01, \
        f"PnL calculation error: expected {expected_net_pnl}, got {net_pnl}"


def test_deterministic_with_seed():
    """Test that results are deterministic with same seed."""
    # Create test data
    dates = pd.date_range('2023-01-01', periods=20, freq='15T')
    data = pd.DataFrame({
        'open': [100] * 20,
        'high': [101] * 20,
        'low': [99] * 20,
        'close': [100] * 20,
        'volume': [1000] * 20
    }, index=dates)
    
    # Run twice with same seed
    config1 = BacktestConfig(seed=123, slippage_bps=50)
    engine1 = BacktestEngine(config1)
    result1 = engine1.run(data, SimpleBuyHoldStrategy())
    
    config2 = BacktestConfig(seed=123, slippage_bps=50)
    engine2 = BacktestEngine(config2)
    result2 = engine2.run(data, SimpleBuyHoldStrategy())
    
    # Results should be identical
    assert len(result1["trades"]) == len(result2["trades"])
    
    if result1["trades"]:
        trade1 = result1["trades"][0]
        trade2 = result2["trades"][0]
        
        assert abs(trade1.slippage - trade2.slippage) < 1e-10, \
            "Slippage should be identical with same seed"
        assert abs(trade1.pnl - trade2.pnl) < 1e-10, \
            "PnL should be identical with same seed"


def test_different_seeds_different_results():
    """Test that different seeds produce different slippage."""
    # Create test data
    dates = pd.date_range('2023-01-01', periods=20, freq='15T')
    data = pd.DataFrame({
        'open': [100] * 20,
        'high': [101] * 20,
        'low': [99] * 20,
        'close': [100] * 20,
        'volume': [1000] * 20
    }, index=dates)
    
    # Run with different seeds
    slippages = []
    for seed in [42, 123, 456]:
        config = BacktestConfig(seed=seed, slippage_bps=100)
        engine = BacktestEngine(config)
        result = engine.run(data, SimpleBuyHoldStrategy())
        
        if result["trades"]:
            slippages.append(result["trades"][0].slippage)
    
    # Should have different slippage values
    assert len(set(slippages)) > 1, "Different seeds should produce different slippage"


if __name__ == "__main__":
    pytest.main([__file__])