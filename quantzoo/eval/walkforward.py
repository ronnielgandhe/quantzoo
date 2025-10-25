"""Walk-forward analysis and validation utilities."""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from quantzoo.backtest.engine import BacktestEngine, BacktestConfig
from quantzoo.metrics.core import calculate_metrics


class WalkForwardAnalysis:
    """
    Walk-forward analysis with purged time series splits to prevent look-ahead bias.
    
    Supports both sliding window and expanding window approaches.
    """
    
    def __init__(self, kind: str = "sliding", train_bars: int = 2000, test_bars: int = 500):
        """
        Initialize walk-forward analysis.
        
        Args:
            kind: "sliding" or "expanding" window
            train_bars: Number of bars for training
            test_bars: Number of bars for testing
        """
        self.kind = kind
        self.train_bars = train_bars
        self.test_bars = test_bars
    
    def run(self, df: pd.DataFrame, strategy, fees_bps: float = 1.0, 
            slippage_bps: float = 1.0, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Run walk-forward analysis on the dataset.
        
        Args:
            df: OHLCV DataFrame
            strategy: Strategy instance
            fees_bps: Trading fees in basis points
            slippage_bps: Slippage in basis points
            seed: Random seed for reproducibility
            
        Returns:
            List of results for each window
        """
        results = []
        
        # Generate splits
        splits = self._generate_splits(len(df))
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Extract train and test data
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # Skip if insufficient data
            if len(test_data) < 10:
                continue
            
            # Create backtest engine
            config = BacktestConfig(
                initial_capital=100000.0,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
                seed=seed + i  # Different seed for each window
            )
            engine = BacktestEngine(config)
            
            # Run backtest on test data
            result = engine.run(test_data, strategy)
            
            # Add window metadata
            result.update({
                "window": i,
                "start_date": test_data.index[0],  # Test period start date
                "end_date": test_data.index[-1],   # Test period end date
                "train_start": train_data.index[0] if len(train_data) > 0 else None,
                "train_end": train_data.index[-1] if len(train_data) > 0 else None,
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "train_bars": len(train_data),
                "test_bars": len(test_data),
            })
            
            results.append(result)
        
        return results
    
    def _generate_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series splits with proper temporal ordering.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        
        if self.kind == "sliding":
            # Sliding window approach
            start_idx = self.train_bars
            
            while start_idx + self.test_bars <= n_samples:
                train_start = start_idx - self.train_bars
                train_end = start_idx
                test_start = start_idx
                test_end = start_idx + self.test_bars
                
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                
                splits.append((train_idx, test_idx))
                
                # Move window forward by test_bars to avoid overlap
                start_idx += self.test_bars
        
        elif self.kind == "expanding":
            # Expanding window approach
            start_idx = self.train_bars
            
            while start_idx + self.test_bars <= n_samples:
                train_start = 0  # Always start from beginning
                train_end = start_idx
                test_start = start_idx
                test_end = start_idx + self.test_bars
                
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                
                splits.append((train_idx, test_idx))
                
                # Move window forward by test_bars
                start_idx += self.test_bars
        
        else:
            raise ValueError(f"Unknown kind: {self.kind}. Use 'sliding' or 'expanding'")
        
        return splits
    
    def purged_k_fold(self, n_samples: int, n_splits: int = 5, 
                      embargo_pct: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged K-fold splits to prevent look-ahead bias.
        
        Args:
            n_samples: Total number of samples
            n_splits: Number of folds
            embargo_pct: Percentage of data to embargo around test set
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        fold_size = n_samples // n_splits
        embargo_size = int(n_samples * embargo_pct)
        
        for i in range(n_splits):
            # Test set
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_idx = np.arange(test_start, test_end)
            
            # Training set with embargo
            train_idx = []
            
            # Before test set (with embargo)
            if test_start > embargo_size:
                train_idx.extend(range(0, test_start - embargo_size))
            
            # After test set (with embargo)
            if test_end + embargo_size < n_samples:
                train_idx.extend(range(test_end + embargo_size, n_samples))
            
            train_idx = np.array(train_idx)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits


def validate_no_lookahead(df: pd.DataFrame, strategy, indicator_func, 
                         lookahead_bars: int = 1) -> bool:
    """
    Validate that strategy doesn't use future information.
    
    Args:
        df: OHLCV DataFrame
        strategy: Strategy instance to test
        indicator_func: Function that calculates indicators
        lookahead_bars: Number of future bars to test for
        
    Returns:
        True if no look-ahead bias detected, False otherwise
    """
    # Calculate indicators normally
    normal_indicators = indicator_func(df)
    
    # Calculate indicators with future data removed
    truncated_df = df.iloc[:-lookahead_bars]
    truncated_indicators = indicator_func(truncated_df)
    
    # Compare overlapping portion
    min_length = min(len(normal_indicators), len(truncated_indicators))
    
    if min_length == 0:
        return True
    
    # Check if values match in overlapping region
    normal_subset = normal_indicators.iloc[:min_length]
    truncated_subset = truncated_indicators.iloc[:min_length]
    
    # Allow small numerical differences
    tolerance = 1e-10
    differences = np.abs(normal_subset - truncated_subset)
    
    # Check if any significant differences exist
    has_lookahead = np.any(differences > tolerance)
    
    return not has_lookahead


def calculate_window_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all walk-forward windows.
    
    Args:
        results: List of window results
        
    Returns:
        Aggregate metrics
    """
    if not results:
        return {}
    
    # Collect all trades and equity curves
    all_trades = []
    all_equity = []
    
    for result in results:
        all_trades.extend(result.get("trades", []))
        all_equity.extend(result.get("equity_curve", []))
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_trades, all_equity)
    
    # Calculate per-window statistics
    window_returns = [result.get("total_return", 0) for result in results]
    window_sharpes = []
    window_drawdowns = []
    
    for result in results:
        trades = result.get("trades", [])
        equity = result.get("equity_curve", [])
        metrics = calculate_metrics(trades, equity)
        window_sharpes.append(metrics.get("sharpe_ratio", 0))
        window_drawdowns.append(metrics.get("max_drawdown", 0))
    
    # Add window statistics
    overall_metrics.update({
        "windows_total": len(results),
        "windows_profitable": sum(1 for r in window_returns if r > 0),
        "avg_window_return": np.mean(window_returns) if window_returns else 0,
        "std_window_return": np.std(window_returns) if window_returns else 0,
        "avg_window_sharpe": np.mean(window_sharpes) if window_sharpes else 0,
        "avg_window_drawdown": np.mean(window_drawdowns) if window_drawdowns else 0,
        "worst_window_return": min(window_returns) if window_returns else 0,
        "best_window_return": max(window_returns) if window_returns else 0,
    })
    
    return overall_metrics