"""Test walk-forward analysis."""

import pytest
import pandas as pd
import numpy as np
from quantzoo.eval.walkforward import WalkForwardAnalysis, validate_no_lookahead


class DummyStrategy:
    """Simple strategy for testing walk-forward analysis."""
    
    def on_start(self, ctx):
        self.trade_count = 0
    
    def on_bar(self, ctx, bar):
        # Make a trade every 50 bars
        if ctx.bar_index() % 50 == 0 and ctx.bar_index() > 0:
            if ctx.position_size() == 0:
                ctx.buy(1, "Long")
                self.trade_count += 1
            else:
                ctx.close_position("Close")


def create_test_data(n_bars: int = 1000) -> pd.DataFrame:
    """Create synthetic test data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01 09:00', periods=n_bars, freq='15T')
    
    # Generate price series with trend and noise
    trend = np.linspace(100, 110, n_bars)
    noise = np.cumsum(np.random.randn(n_bars) * 0.1)
    prices = trend + noise
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 0.5, n_bars),
        'low': prices - np.random.uniform(0, 0.5, n_bars),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)
    
    return data


def test_sliding_window_analysis():
    """Test sliding window walk-forward analysis."""
    data = create_test_data(1000)
    
    wf = WalkForwardAnalysis(
        kind="sliding",
        train_bars=200,
        test_bars=100
    )
    
    strategy = DummyStrategy()
    results = wf.run(data, strategy, fees_bps=1.0, slippage_bps=1.0, seed=42)
    
    # Should have multiple windows
    assert len(results) > 0, "Should have at least one window"
    
    # Each result should have expected structure
    for i, result in enumerate(results):
        assert "window" in result, f"Window {i} missing 'window' field"
        assert "start_date" in result, f"Window {i} missing 'start_date' field"
        assert "end_date" in result, f"Window {i} missing 'end_date' field"
        assert "trades" in result, f"Window {i} missing 'trades' field"
        assert "equity_curve" in result, f"Window {i} missing 'equity_curve' field"
        
        # Test data should have 100 bars
        assert result["test_bars"] == 100, f"Window {i} should have 100 test bars"
        
        # Training data should have 200 bars
        assert result["train_bars"] == 200, f"Window {i} should have 200 train bars"
    
    # Windows should not overlap in test periods
    for i in range(len(results) - 1):
        current_end = results[i]["end_date"]
        next_start = results[i + 1]["test_start"]
        assert current_end <= next_start, f"Windows {i} and {i+1} overlap in test periods"


def test_expanding_window_analysis():
    """Test expanding window walk-forward analysis."""
    data = create_test_data(800)
    
    wf = WalkForwardAnalysis(
        kind="expanding",
        train_bars=200,
        test_bars=100
    )
    
    strategy = DummyStrategy()
    results = wf.run(data, strategy, fees_bps=1.0, slippage_bps=1.0, seed=42)
    
    # Should have multiple windows
    assert len(results) > 0, "Should have at least one window"
    
    # Training data should expand over time
    train_sizes = [result["train_bars"] for result in results]
    
    # First window should have exactly train_bars
    assert train_sizes[0] == 200, f"First window should have 200 train bars, got {train_sizes[0]}"
    
    # Later windows should have more training data
    if len(train_sizes) > 1:
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1], \
                f"Training data should expand: window {i-1} has {train_sizes[i-1]}, window {i} has {train_sizes[i]}"


def test_window_data_integrity():
    """Test that window data doesn't leak between train and test."""
    data = create_test_data(500)
    
    wf = WalkForwardAnalysis(
        kind="sliding",
        train_bars=100,
        test_bars=50
    )
    
    # Generate splits manually to inspect
    splits = wf._generate_splits(len(data))
    
    for train_idx, test_idx in splits:
        # Train and test indices should not overlap
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set.intersection(test_set)
        assert len(overlap) == 0, f"Train and test sets overlap: {overlap}"
        
        # Test should come after train
        max_train_idx = max(train_idx) if len(train_idx) > 0 else -1
        min_test_idx = min(test_idx) if len(test_idx) > 0 else float('inf')
        assert max_train_idx < min_test_idx, \
            f"Test data should come after train data: max_train={max_train_idx}, min_test={min_test_idx}"


def test_deterministic_results():
    """Test that results are deterministic with same seed."""
    data = create_test_data(400)
    
    wf = WalkForwardAnalysis(
        kind="sliding",
        train_bars=100,
        test_bars=50
    )
    
    strategy1 = DummyStrategy()
    results1 = wf.run(data, strategy1, seed=123)
    
    strategy2 = DummyStrategy()
    results2 = wf.run(data, strategy2, seed=123)
    
    # Should have same number of windows
    assert len(results1) == len(results2), "Should have same number of windows"
    
    # Each window should have same results
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        assert len(r1["trades"]) == len(r2["trades"]), f"Window {i} trade count differs"
        assert len(r1["equity_curve"]) == len(r2["equity_curve"]), f"Window {i} equity curve length differs"
        
        # Final equity should be identical
        if r1["equity_curve"] and r2["equity_curve"]:
            assert abs(r1["equity_curve"][-1] - r2["equity_curve"][-1]) < 1e-10, \
                f"Window {i} final equity differs"


def test_insufficient_data_handling():
    """Test handling of insufficient data scenarios."""
    # Very small dataset
    data = create_test_data(50)
    
    wf = WalkForwardAnalysis(
        kind="sliding",
        train_bars=100,  # More than available data
        test_bars=50
    )
    
    strategy = DummyStrategy()
    results = wf.run(data, strategy)
    
    # Should handle gracefully (empty results or appropriate error)
    assert isinstance(results, list), "Should return a list even with insufficient data"


def test_purged_k_fold():
    """Test purged K-fold cross-validation."""
    wf = WalkForwardAnalysis()
    
    n_samples = 1000
    splits = wf.purged_k_fold(n_samples, n_splits=5, embargo_pct=0.01)
    
    # Should have 5 splits
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # No overlap between train and test
        train_set = set(train_idx)
        test_set = set(test_idx)
        overlap = train_set.intersection(test_set)
        assert len(overlap) == 0, f"Split {i}: train and test overlap"
        
        # Test that embargo is respected
        if len(test_idx) > 0 and len(train_idx) > 0:
            min_test = min(test_idx)
            max_test = max(test_idx)
            
            # Check embargo before test set
            train_before_test = [idx for idx in train_idx if idx < min_test]
            if train_before_test:
                max_train_before = max(train_before_test)
                embargo_size = int(n_samples * 0.01)
                assert max_train_before < min_test - embargo_size, \
                    f"Split {i}: embargo not respected before test set"
            
            # Check embargo after test set
            train_after_test = [idx for idx in train_idx if idx > max_test]
            if train_after_test:
                min_train_after = min(train_after_test)
                embargo_size = int(n_samples * 0.01)
                assert min_train_after > max_test + embargo_size, \
                    f"Split {i}: embargo not respected after test set"


def dummy_indicator_function(df):
    """Dummy indicator function for lookahead testing."""
    return df['close'].rolling(10).mean()


def test_no_lookahead_validation():
    """Test the no lookahead validation function."""
    data = create_test_data(100)
    
    # Test with proper indicator (should pass)
    result = validate_no_lookahead(
        data, 
        DummyStrategy(), 
        dummy_indicator_function,
        lookahead_bars=5
    )
    
    # Should not detect lookahead bias in rolling mean
    assert result == True, "Rolling mean should not have lookahead bias"


if __name__ == "__main__":
    pytest.main([__file__])