"""Tests for plot/candles module."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sys
import os

# Add quantzoo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from quantzoo.plot.candles import make_candles, create_mnq_808_overlays
    QUANTZOO_AVAILABLE = True
except ImportError:
    QUANTZOO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestCandlesModule(unittest.TestCase):
    """Test suite for candles plotting module."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range('2025-01-01', periods=100, freq='15min', tz='UTC')
        self.sample_df = pd.DataFrame({
            'open': np.random.uniform(4200, 4300, 100),
            'high': np.random.uniform(4220, 4320, 100),
            'low': np.random.uniform(4180, 4280, 100),
            'close': np.random.uniform(4200, 4300, 100),
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        for i in range(len(self.sample_df)):
            row = self.sample_df.iloc[i]
            max_oc = max(row['open'], row['close'])
            min_oc = min(row['open'], row['close'])
            self.sample_df.iloc[i, self.sample_df.columns.get_loc('high')] = max(row['high'], max_oc)
            self.sample_df.iloc[i, self.sample_df.columns.get_loc('low')] = min(row['low'], min_oc)
        
        # Sample trade data
        self.sample_trades = [
            {
                'time': '2025-01-01T08:15:00-05:00',
                'side': 'buy',
                'qty': 1,
                'price': 4250.0,
                'reason': 'crossover(long)',
                'fees_bps': 1.0,
                'slippage_bps': 0.5
            },
            {
                'time': '2025-01-01T10:30:00-05:00',
                'side': 'sell',
                'qty': 1,
                'price': 4275.0,
                'reason': 'profit_target',
                'exit_time': '2025-01-01T10:30:00-05:00',
                'exit_price': 4275.0,
                'pnl': 25.0
            }
        ]
        
        # Sample overlays
        self.sample_overlays = {
            'anchor': np.random.uniform(4240, 4260, 100),
            'upper_band': np.random.uniform(4280, 4300, 100),
            'lower_band': np.random.uniform(4200, 4220, 100)
        }
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_make_candles_basic(self):
        """Test basic candlestick chart creation."""
        fig = make_candles(self.sample_df)
        
        # Should return a plotly Figure
        self.assertIsInstance(fig, go.Figure)
        
        # Should have at least one trace (candlestick)
        self.assertGreater(len(fig.data), 0)
        
        # First trace should be candlestick
        self.assertEqual(fig.data[0].type, 'candlestick')
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_make_candles_with_trades(self):
        """Test candlestick chart with trade markers."""
        fig = make_candles(self.sample_df, trades=self.sample_trades)
        
        # Should have candlestick + trade markers
        self.assertGreater(len(fig.data), 1)
        
        # Check for trade marker traces
        trace_names = [trace.name for trace in fig.data]
        self.assertIn('Buys', trace_names)
        self.assertIn('Sells', trace_names)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_make_candles_with_overlays(self):
        """Test candlestick chart with overlays."""
        fig = make_candles(self.sample_df, overlays=self.sample_overlays)
        
        # Should have candlestick + overlay traces
        self.assertGreater(len(fig.data), 1)
        
        # Check for overlay traces
        trace_names = [trace.name for trace in fig.data]
        self.assertIn('Anchor', trace_names)
        self.assertIn('Upper Band', trace_names)
        self.assertIn('Lower Band', trace_names)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_make_candles_with_stops(self):
        """Test candlestick chart with stop lines."""
        stops = {
            'stop_loss': 4200.0,
            'trail_stop': [4210.0] * 100,  # Array of values
            'take_profit': 4300.0
        }
        
        fig = make_candles(self.sample_df, stops=stops)
        
        # Should have candlestick + stop traces
        self.assertGreater(len(fig.data), 1)
        
        # Check for stop traces
        trace_names = [trace.name for trace in fig.data]
        self.assertIn('Stop Loss', trace_names)
        self.assertIn('Trailing Stop', trace_names)
        self.assertIn('Take Profit', trace_names)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_make_candles_timezone_handling(self):
        """Test timezone conversion in charts."""
        # Test with different timezone
        fig = make_candles(self.sample_df, tz="America/New_York")
        
        self.assertIsInstance(fig, go.Figure)
        
        # Check that layout includes timezone info in title
        self.assertIn("America/New_York", fig.layout.title.text)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_make_candles_missing_columns(self):
        """Test error handling for missing columns."""
        # Remove required column
        incomplete_df = self.sample_df.drop('close', axis=1)
        
        with self.assertRaises(ValueError) as context:
            make_candles(incomplete_df)
        
        self.assertIn("Missing required columns", str(context.exception))
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_make_candles_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        fig = make_candles(empty_df)
        
        # Should still return a figure, but might be empty or have annotation
        self.assertIsInstance(fig, go.Figure)
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_create_mnq_808_overlays(self):
        """Test MNQ 808 strategy overlay creation."""
        overlays = create_mnq_808_overlays(self.sample_df)
        
        # Should return dictionary with expected keys
        expected_keys = ['anchor', 'upper_band', 'lower_band', 'atr', 'sma_tr']
        for key in expected_keys:
            self.assertIn(key, overlays)
        
        # All overlays should be pandas Series
        for key, series in overlays.items():
            self.assertIsInstance(series, pd.Series)
            self.assertEqual(len(series), len(self.sample_df))
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    def test_trade_marker_parsing(self):
        """Test trade timestamp parsing."""
        # Test various timestamp formats
        test_trades = [
            {'time': '2025-01-01T08:15:00-05:00', 'side': 'buy', 'qty': 1, 'price': 4250.0, 'reason': 'test'},
            {'time': '2025-01-01T08:15:00Z', 'side': 'sell', 'qty': 1, 'price': 4250.0, 'reason': 'test'},
            {'time': '2025-01-01', 'side': 'buy', 'qty': 1, 'price': 4250.0, 'reason': 'test'},
        ]
        
        # Should not raise errors
        try:
            fig = make_candles(self.sample_df, trades=test_trades)
            self.assertIsInstance(fig, go.Figure)
        except Exception as e:
            self.fail(f"Trade parsing should handle various formats: {e}")


class TestCandlesWithoutPlotly(unittest.TestCase):
    """Test behavior when plotly is not available."""
    
    def setUp(self):
        """Set up minimal test data."""
        self.sample_df = pd.DataFrame({
            'open': [4200, 4210],
            'high': [4220, 4230],
            'low': [4190, 4200],
            'close': [4210, 4220],
            'volume': [1000, 1500]
        })
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE, "Requires quantzoo")
    @patch('quantzoo.plot.candles.PLOTLY_AVAILABLE', False)
    def test_make_candles_without_plotly(self):
        """Test that appropriate error is raised when plotly is not available."""
        with self.assertRaises(ImportError) as context:
            make_candles(self.sample_df)
        
        self.assertIn("Plotly is required", str(context.exception))


class TestCandlesPerformance(unittest.TestCase):
    """Performance tests for candles module."""
    
    def setUp(self):
        """Set up large dataset for performance testing."""
        dates = pd.date_range('2025-01-01', periods=10000, freq='1min', tz='UTC')
        self.large_df = pd.DataFrame({
            'open': np.random.uniform(4200, 4300, 10000),
            'high': np.random.uniform(4220, 4320, 10000),
            'low': np.random.uniform(4180, 4280, 10000),
            'close': np.random.uniform(4200, 4300, 10000),
            'volume': np.random.randint(1000, 5000, 10000)
        }, index=dates)
        
        # Large trade dataset
        self.large_trades = []
        for i in range(1000):
            self.large_trades.append({
                'time': dates[i * 10].isoformat(),
                'side': 'buy' if i % 2 == 0 else 'sell',
                'qty': 1,
                'price': 4200 + i,
                'reason': f'trade_{i}'
            })
    
    @unittest.skipUnless(QUANTZOO_AVAILABLE and PLOTLY_AVAILABLE, "Requires quantzoo and plotly")
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        import time
        
        start_time = time.time()
        fig = make_candles(self.large_df, trades=self.large_trades[:100])  # Limit trades for performance
        end_time = time.time()
        
        # Should complete within reasonable time (10 seconds)
        self.assertLess(end_time - start_time, 10.0)
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()