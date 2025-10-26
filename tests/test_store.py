"""Tests for DuckDB storage layer."""

import pytest
import tempfile
import os
import shutil
import pandas as pd
from datetime import datetime

from quantzoo.store.duck import DuckStore


class TestDuckStore:
    """Test DuckStore functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.duckdb")
        self.store = DuckStore(self.db_path, self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_store_initialization(self):
        """Test store initialization."""
        assert os.path.exists(self.db_path)
        assert os.path.exists(self.temp_dir)
    
    def test_write_read_trades(self):
        """Test writing and reading trades data."""
        # Create test trades data
        trades_df = pd.DataFrame({
            'symbol': ['MNQ', 'MNQ', 'MNQ'],
            'entry_time': [datetime.now(), datetime.now(), datetime.now()],
            'exit_time': [datetime.now(), datetime.now(), datetime.now()],
            'side': ['long', 'short', 'long'],
            'quantity': [1, 1, 1],
            'entry_price': [15200.0, 15250.0, 15300.0],
            'exit_price': [15225.0, 15225.0, 15350.0],
            'pnl': [25.0, 25.0, 50.0],
            'commission': [2.0, 2.0, 2.0]
        })
        
        run_id = "test_run_001"
        
        # Write trades
        self.store.write_trades(trades_df, run_id)
        
        # Read trades back
        read_trades = self.store.read_trades(run_id)
        
        assert read_trades is not None
        assert len(read_trades) == 3
        assert 'run_id' in read_trades.columns
        assert all(read_trades['run_id'] == run_id)
    
    def test_write_read_equity(self):
        """Test writing and reading equity data."""
        # Create test equity data
        equity_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='15min'),
            'equity': [100000 + i * 100 for i in range(100)],
            'drawdown': [0.0] * 100
        })
        
        run_id = "test_run_002"
        
        # Write equity
        self.store.write_equity(equity_df, run_id)
        
        # Read equity back
        read_equity = self.store.read_equity(run_id)
        
        assert read_equity is not None
        assert len(read_equity) == 100
        assert 'run_id' in read_equity.columns
        assert all(read_equity['run_id'] == run_id)
    
    def test_metadata_tracking(self):
        """Test run metadata tracking."""
        run_id = "test_run_003"
        
        # Write with metadata
        trades_df = pd.DataFrame({
            'symbol': ['MNQ'],
            'pnl': [100.0]
        })
        
        metadata = {
            'seed': 42,
            'strategy_name': 'test_strategy',
            'config_path': 'test_config.yaml',
            'metrics': {
                'total_trades': 1,
                'total_return': 0.001,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.02
            }
        }
        
        self.store.write_trades(trades_df, run_id, metadata)
        
        # Check metadata
        runs = self.store.list_runs()
        assert len(runs) == 1
        assert runs[0]['run_id'] == run_id
        assert runs[0]['seed'] == 42
        assert runs[0]['strategy_name'] == 'test_strategy'
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        run_id = "test_empty"
        
        # Should not crash on empty data
        self.store.write_trades(empty_df, run_id)
        self.store.write_equity(empty_df, run_id)
        
        # Should return None for non-existent data
        assert self.store.read_trades(run_id) is None
        assert self.store.read_equity(run_id) is None
    
    def test_nonexistent_run(self):
        """Test reading non-existent run data."""
        assert self.store.read_trades("nonexistent") is None
        assert self.store.read_equity("nonexistent") is None
    
    def test_list_runs_empty(self):
        """Test listing runs when database is empty."""
        runs = self.store.list_runs()
        assert runs == []
    
    def test_query_functionality(self):
        """Test SQL query functionality."""
        # Create test data
        trades_df = pd.DataFrame({
            'symbol': ['MNQ', 'MES', 'MNQ'],
            'pnl': [100.0, -50.0, 75.0],
            'side': ['long', 'short', 'long']
        })
        
        self.store.write_trades(trades_df, "query_test")
        
        # Test basic query
        all_trades = self.store.query_trades()
        assert len(all_trades) == 3
        
        # Test filtered query
        mnq_trades = self.store.query_trades("symbol = 'MNQ'")
        assert len(mnq_trades) == 2
        
        # Test limit
        limited_trades = self.store.query_trades(limit=2)
        assert len(limited_trades) == 2
    
    def test_run_summary(self):
        """Test run summary generation."""
        # Create test run
        trades_df = pd.DataFrame({
            'symbol': ['MNQ'],
            'pnl': [100.0]
        })
        
        metadata = {
            'strategy_name': 'test_summary',
            'metrics': {
                'total_trades': 1,
                'total_return': 0.001,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.02
            }
        }
        
        self.store.write_trades(trades_df, "summary_test", metadata)
        
        # Get summary
        summary = self.store.get_run_summary()
        assert len(summary) == 1
        assert summary.iloc[0]['strategy_name'] == 'test_summary'
        assert summary.iloc[0]['total_trades'] == 1


if __name__ == "__main__":
    pytest.main([__file__])