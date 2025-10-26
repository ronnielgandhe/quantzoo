"""Tests for position and PnL math functionality."""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add quantzoo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from quantzoo.backtest.engine import StrategyContext, BacktestEngine, BacktestConfig, Position
    QUANTZOO_AVAILABLE = True
except ImportError:
    QUANTZOO_AVAILABLE = False


class MockStrategy:
    """Mock strategy for testing."""
    
    def on_start(self, ctx):
        pass
    
    def on_bar(self, ctx, bar):
        # Simple buy and hold strategy for testing
        if ctx.bar_index() == 0:
            ctx.buy(1, "Entry")
        elif ctx.bar_index() == 5:
            ctx.close_position("Exit")


class TestPositionMath(unittest.TestCase):
    """Test suite for position and PnL calculations."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample price data
        dates = pd.date_range('2025-01-01', periods=10, freq='15min')
        self.sample_data = pd.DataFrame({
            'open': [4200, 4210, 4205, 4215, 4220, 4225, 4230, 4235, 4240, 4245],
            'high': [4210, 4220, 4215, 4225, 4230, 4235, 4240, 4245, 4250, 4255],
            'low': [4195, 4205, 4200, 4210, 4215, 4220, 4225, 4230, 4235, 4240],
            'close': [4205, 4215, 4210, 4220, 4225, 4230, 4235, 4240, 4245, 4250],
            'volume': [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1450]
        }, index=dates)
        
        # Create backtest config
        self.config = BacktestConfig(
            initial_capital=100000,
            fees_bps=1.0,
            slippage_bps=1.0,
            seed=42
        )
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_position_basic_operations(self):
        """Test basic position operations."""
        position = Position()
        
        # Test initial state
        self.assertTrue(position.is_flat())
        self.assertFalse(position.is_long())
        self.assertFalse(position.is_short())
        self.assertEqual(position.size, 0.0)
        self.assertEqual(position.avg_price, 0.0)
        self.assertEqual(position.unrealized_pnl, 0.0)
        
        # Test long position
        position.size = 1.0
        position.avg_price = 4200.0
        self.assertFalse(position.is_flat())
        self.assertTrue(position.is_long())
        self.assertFalse(position.is_short())
        
        # Test short position
        position.size = -1.0
        self.assertFalse(position.is_flat())
        self.assertFalse(position.is_long())
        self.assertTrue(position.is_short())
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_strategy_context_position_methods(self):
        """Test StrategyContext position methods."""
        engine = BacktestEngine(self.config)
        results = engine.run(self.sample_data, MockStrategy())
        
        # Create context for testing
        ctx = StrategyContext(engine)
        ctx._bar_index = 0
        ctx._current_bar = self.sample_data.iloc[0]
        
        # Test current_position method
        position_info = ctx.current_position()
        
        # Should return a dictionary with required fields
        self.assertIsInstance(position_info, dict)
        required_fields = ["side", "qty", "avg_price", "entry_time"]
        for field in required_fields:
            self.assertIn(field, position_info)
        
        # Test that side is one of expected values
        self.assertIn(position_info["side"], ["flat", "long", "short"])
        
        # Test that qty is non-negative
        self.assertGreaterEqual(position_info["qty"], 0)
        
        # Test unrealized_pnl method
        pnl_info = ctx.unrealized_pnl()
        
        # Should return a dictionary with dollars and percent
        self.assertIsInstance(pnl_info, dict)
        self.assertIn("dollars", pnl_info)
        self.assertIn("percent", pnl_info)
        
        # Both should be numeric
        self.assertIsInstance(pnl_info["dollars"], (int, float))
        self.assertIsInstance(pnl_info["percent"], (int, float))
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_position_tracking_during_trade_cycle(self):
        """Test position tracking through a complete trade cycle."""
        
        class TestStrategy:
            def __init__(self):
                self.positions_log = []
                self.pnl_log = []
            
            def on_start(self, ctx):
                pass
            
            def on_bar(self, ctx, bar):
                # Log position and PnL at each bar
                pos_info = ctx.current_position()
                pnl_info = ctx.unrealized_pnl()
                
                self.positions_log.append({
                    'bar_index': ctx.bar_index(),
                    'position': pos_info,
                    'pnl': pnl_info,
                    'price': ctx.close
                })
                
                # Execute trades
                if ctx.bar_index() == 1:  # Enter long position
                    ctx.buy(1, "Entry")
                elif ctx.bar_index() == 5:  # Exit position
                    ctx.close_position("Exit")
        
        strategy = TestStrategy()
        engine = BacktestEngine(self.config)
        results = engine.run(self.sample_data, strategy)
        
        # Verify position tracking
        self.assertGreater(len(strategy.positions_log), 0)
        
        # Check position states
        for i, log_entry in enumerate(strategy.positions_log):
            pos = log_entry['position']
            pnl = log_entry['pnl']
            
            if i <= 1:  # Before entry
                self.assertEqual(pos['side'], 'flat')
                self.assertEqual(pos['qty'], 0)
                self.assertEqual(pnl['dollars'], 0.0)
                self.assertEqual(pnl['percent'], 0.0)
            elif 1 < i <= 5:  # During position
                self.assertEqual(pos['side'], 'long')
                self.assertEqual(pos['qty'], 1)
                self.assertGreater(pos['avg_price'], 0)
                # PnL should be calculated
                self.assertIsInstance(pnl['dollars'], (int, float))
                self.assertIsInstance(pnl['percent'], (int, float))
            else:  # After exit
                self.assertEqual(pos['side'], 'flat')
                self.assertEqual(pos['qty'], 0)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_unrealized_pnl_calculation(self):
        """Test unrealized PnL calculation accuracy."""
        
        class PnLTestStrategy:
            def __init__(self):
                self.entry_price = None
                self.pnl_at_exit = None
            
            def on_start(self, ctx):
                pass
            
            def on_bar(self, ctx, bar):
                if ctx.bar_index() == 1:
                    ctx.buy(1, "Entry")
                    self.entry_price = ctx.close
                elif ctx.bar_index() == 3:
                    # Check PnL before exit
                    pnl_info = ctx.unrealized_pnl()
                    self.pnl_at_exit = pnl_info
                elif ctx.bar_index() == 4:
                    ctx.close_position("Exit")
        
        strategy = PnLTestStrategy()
        engine = BacktestEngine(self.config)
        results = engine.run(self.sample_data, strategy)
        
        # Verify PnL calculation
        if strategy.entry_price and strategy.pnl_at_exit:
            expected_pnl = self.sample_data.iloc[3]['close'] - strategy.entry_price
            
            # Allow for small differences due to fees/slippage
            actual_pnl = strategy.pnl_at_exit['dollars']
            self.assertAlmostEqual(expected_pnl, actual_pnl, delta=10.0)
            
            # Test percentage calculation
            notional = strategy.entry_price * 1  # 1 contract
            expected_pct = (expected_pnl / notional) * 100
            actual_pct = strategy.pnl_at_exit['percent']
            self.assertAlmostEqual(expected_pct, actual_pct, delta=1.0)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_position_entry_time_tracking(self):
        """Test that entry time is correctly tracked."""
        
        class EntryTimeStrategy:
            def __init__(self):
                self.entry_time = None
                self.position_at_entry = None
            
            def on_start(self, ctx):
                pass
            
            def on_bar(self, ctx, bar):
                if ctx.bar_index() == 2:
                    ctx.buy(1, "Entry")
                elif ctx.bar_index() == 3:
                    pos_info = ctx.current_position()
                    self.position_at_entry = pos_info
                    self.entry_time = pos_info.get('entry_time')
        
        strategy = EntryTimeStrategy()
        engine = BacktestEngine(self.config)
        results = engine.run(self.sample_data, strategy)
        
        # Verify entry time is tracked
        self.assertIsNotNone(strategy.entry_time)
        
        # Entry time should be at the correct index
        if strategy.entry_time:
            expected_time = self.sample_data.index[2]  # Entry at bar 2
            # Compare timestamps (allowing for potential timezone differences)
            if hasattr(strategy.entry_time, 'strftime'):
                self.assertEqual(strategy.entry_time, expected_time)
            else:
                # Handle string representation
                self.assertIn(str(expected_time.date()), str(strategy.entry_time))
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_position_methods_with_no_position(self):
        """Test position methods when no position is held."""
        engine = BacktestEngine(self.config)
        
        # Don't run any strategy, just test context methods
        ctx = StrategyContext(engine)
        ctx._bar_index = 0
        ctx._current_bar = self.sample_data.iloc[0]
        
        # Test with flat position
        position_info = ctx.current_position()
        self.assertEqual(position_info['side'], 'flat')
        self.assertEqual(position_info['qty'], 0)
        self.assertEqual(position_info['avg_price'], 0.0)
        self.assertIsNone(position_info['entry_time'])
        
        pnl_info = ctx.unrealized_pnl()
        self.assertEqual(pnl_info['dollars'], 0.0)
        self.assertEqual(pnl_info['percent'], 0.0)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_position_methods_consistency(self):
        """Test consistency between position methods and engine state."""
        
        class ConsistencyStrategy:
            def __init__(self):
                self.consistency_checks = []
            
            def on_start(self, ctx):
                pass
            
            def on_bar(self, ctx, bar):
                # Check consistency at each bar
                pos_info = ctx.current_position()
                pnl_info = ctx.unrealized_pnl()
                
                # Compare with engine state
                engine_pos = ctx.engine.position
                
                # Size consistency
                if engine_pos.is_flat():
                    self.consistency_checks.append(pos_info['side'] == 'flat')
                    self.consistency_checks.append(pos_info['qty'] == 0)
                elif engine_pos.is_long():
                    self.consistency_checks.append(pos_info['side'] == 'long')
                    self.consistency_checks.append(pos_info['qty'] == abs(engine_pos.size))
                elif engine_pos.is_short():
                    self.consistency_checks.append(pos_info['side'] == 'short')
                    self.consistency_checks.append(pos_info['qty'] == abs(engine_pos.size))
                
                # PnL consistency
                self.consistency_checks.append(
                    abs(pnl_info['dollars'] - engine_pos.unrealized_pnl) < 1e-6
                )
                
                # Execute trades for variety
                if ctx.bar_index() == 1:
                    ctx.buy(1, "Long")
                elif ctx.bar_index() == 4:
                    ctx.sell(2, "Short")
                elif ctx.bar_index() == 7:
                    ctx.close_position("Close")
        
        strategy = ConsistencyStrategy()
        engine = BacktestEngine(self.config)
        results = engine.run(self.sample_data, strategy)
        
        # All consistency checks should pass
        self.assertTrue(all(strategy.consistency_checks))
        self.assertGreater(len(strategy.consistency_checks), 0)


if __name__ == "__main__":
    unittest.main()