"""Test for look-ahead bias prevention."""

import pytest
import pandas as pd
import numpy as np
from quantzoo.strategies.mnq_808 import MNQ808, MNQ808Params
from quantzoo.backtest.engine import BacktestEngine, BacktestConfig, StrategyContext
from quantzoo.data.loaders import calculate_rsi, calculate_mfi, calculate_atr


def test_no_future_data_access():
    """Test that strategy cannot access future data."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='15T')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Generate realistic OHLC
    for i in range(len(data)):
        open_price = data.loc[data.index[i], 'open']
        noise = np.random.randn(3) * 0.05
        high = open_price + abs(noise[0])
        low = open_price - abs(noise[1])
        close = open_price + noise[2]
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.loc[data.index[i], 'high'] = high
        data.loc[data.index[i], 'low'] = low
        data.loc[data.index[i], 'close'] = close
    
    # Test that context cannot access future bars
    config = BacktestConfig(seed=42)
    engine = BacktestEngine(config)
    engine.data = data
    ctx = StrategyContext(engine)
    
    # Test at middle of dataset
    ctx._bar_index = 50
    ctx._current_bar = data.iloc[50]
    
    # Should be able to access current and past data
    assert not np.isnan(ctx.get_series("close", 0))  # current
    assert not np.isnan(ctx.get_series("close", -1))  # previous
    assert not np.isnan(ctx.get_series("close", -10))  # 10 bars ago
    
    # Should return NaN for future data
    assert np.isnan(ctx.get_series("close", 1))  # next bar
    assert np.isnan(ctx.get_series("close", 10))  # 10 bars ahead


def test_indicator_lookback_only():
    """Test that indicators only use historical data."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=50, freq='15T')
    prices = 100 + np.cumsum(np.random.randn(50) * 0.1)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 1, 50),
        'low': prices - np.random.uniform(0, 1, 50),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # Calculate indicators with full dataset
    rsi_full = calculate_rsi(data['close'], 14)
    atr_full = calculate_atr(data, 14)
    
    # Calculate indicators with truncated dataset
    data_truncated = data.iloc[:-5]  # Remove last 5 bars
    rsi_truncated = calculate_rsi(data_truncated['close'], 14)
    atr_truncated = calculate_atr(data_truncated, 14)
    
    # Compare overlapping values - they should be identical
    overlap_len = min(len(rsi_full), len(rsi_truncated))
    if overlap_len > 14:  # Only compare after warmup period
        rsi_diff = np.abs(rsi_full.iloc[14:overlap_len] - rsi_truncated.iloc[14:])
        atr_diff = np.abs(atr_full.iloc[14:overlap_len] - atr_truncated.iloc[14:])
        
        # Should be identical (within numerical precision)
        assert np.all(rsi_diff < 1e-10), "RSI shows look-ahead bias"
        assert np.all(atr_diff < 1e-10), "ATR shows look-ahead bias"


def test_strategy_indicator_consistency():
    """Test that strategy indicators are calculated consistently."""
    params = MNQ808Params(lookback=10, use_mfi=False)  # Use RSI for simpler testing
    strategy = MNQ808(params)
    
    # Create simple test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=30, freq='15T')
    data = pd.DataFrame({
        'open': np.arange(100, 130),
        'high': np.arange(101, 131),
        'low': np.arange(99, 129),
        'close': np.arange(100, 130),
        'volume': [1000] * 30
    }, index=dates)
    
    # Run strategy and capture indicator values
    config = BacktestConfig(seed=42)
    engine = BacktestEngine(config)
    ctx = StrategyContext(engine)
    engine.data = data
    
    strategy.on_start(ctx)
    
    indicator_values = []
    for i in range(len(data)):
        ctx._bar_index = i
        ctx._current_bar = data.iloc[i]
        strategy.on_bar(ctx, data.iloc[i])
        
        if i >= params.lookback:
            # Calculate expected RSI manually for comparison
            recent_closes = [data.iloc[j]['close'] for j in range(max(0, i-params.lookback), i+1)]
            if len(recent_closes) > 1:
                gains = []
                losses = []
                for k in range(1, len(recent_closes)):
                    diff = recent_closes[k] - recent_closes[k-1]
                    gains.append(max(0, diff))
                    losses.append(max(0, -diff))
                
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    expected_rsi = 100 - (100 / (1 + rs))
                else:
                    expected_rsi = 100
                
                # Store for comparison
                if len(strategy.momentum_history) > 0:
                    actual_rsi = strategy.momentum_history[-1]
                    indicator_values.append((expected_rsi, actual_rsi))
    
    # Check that calculated values are reasonable
    # (Exact comparison is difficult due to different calculation methods)
    assert len(indicator_values) > 0, "No indicator values calculated"
    
    for expected, actual in indicator_values:
        assert 0 <= actual <= 100, f"RSI out of range: {actual}"
        # Allow some tolerance for calculation differences
        assert abs(expected - actual) < 10, f"RSI difference too large: {abs(expected - actual)}"


if __name__ == "__main__":
    pytest.main([__file__])