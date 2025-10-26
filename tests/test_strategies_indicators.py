"""Tests for additional strategies and technical indicators."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from quantzoo.indicators.ta import ATR, RSI, MFI, SMA, EMA, MACD, BollingerBands
from quantzoo.strategies.momentum import Momentum, MomentumParams
from quantzoo.strategies.vol_breakout import VolBreakout, VolBreakoutParams
from quantzoo.strategies.pairs import Pairs, PairsParams
from quantzoo.backtest.engine import StrategyContext


class TestTechnicalIndicators:
    """Test technical analysis indicators."""
    
    def test_sma(self):
        """Test Simple Moving Average."""
        sma = SMA(period=3)
        
        # Not ready initially
        assert sma.update(10.0) is None
        assert sma.update(12.0) is None
        
        # Ready after period
        result = sma.update(14.0)
        assert result is not None
        assert result == 12.0  # (10 + 12 + 14) / 3
        
        # Continue updating
        result = sma.update(16.0)
        assert result == 14.0  # (12 + 14 + 16) / 3
    
    def test_ema(self):
        """Test Exponential Moving Average."""
        ema = EMA(period=3)
        
        # First value
        result = ema.update(10.0)
        assert result == 10.0
        assert ema.ready
        
        # Subsequent values
        result = ema.update(12.0)
        assert result is not None
        assert result != 12.0  # Should be smoothed
    
    def test_atr(self):
        """Test Average True Range."""
        atr = ATR(period=3)
        
        # First bar
        result = atr.update(15.0, 14.0, 14.5)
        assert result is not None  # First bar gives initial value
        
        # Second bar
        result = atr.update(15.5, 14.5, 15.0)
        assert result is not None
        
        # Should be ready after first calculation
        assert atr.ready
    
    def test_rsi(self):
        """Test Relative Strength Index."""
        rsi = RSI(period=3)
        
        # Need multiple prices to calculate
        assert rsi.update(10.0) is None  # First price
        assert rsi.update(11.0) is None  # Second price (gain)
        
        # Should start giving values after enough data
        result = rsi.update(12.0)
        assert result is not None
        assert 0 <= result <= 100
    
    def test_mfi(self):
        """Test Money Flow Index."""
        mfi = MFI(period=3)
        
        # Need multiple bars
        assert mfi.update(15.0, 14.0, 14.5, 1000) is None
        assert mfi.update(15.5, 14.5, 15.0, 1100) is None
        
        # Should give value after period
        result = mfi.update(16.0, 15.0, 15.5, 1200)
        assert result is not None
        assert 0 <= result <= 100
    
    def test_macd(self):
        """Test MACD."""
        macd = MACD(fast_period=3, slow_period=5, signal_period=2)
        
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        for price in prices:
            result = macd.update(price)
            if result is not None:
                assert 'macd' in result
                assert 'signal' in result
                assert 'histogram' in result
                break
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands."""
        bb = BollingerBands(period=3, std_dev=2.0)
        
        # Need period data
        assert bb.update(10.0) is None
        assert bb.update(12.0) is None
        
        result = bb.update(14.0)
        assert result is not None
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        assert result['upper'] > result['middle'] > result['lower']


class TestMomentumStrategy:
    """Test momentum strategy."""
    
    def create_mock_context(self, bar_index: int, position: int = 0, close_price: float = 15000.0):
        """Create mock strategy context."""
        context = Mock(spec=StrategyContext)
        context.bar_index = bar_index
        context.position = position
        context.current_bar = {
            'open': close_price - 5,
            'high': close_price + 10,
            'low': close_price - 10,
            'close': close_price,
            'volume': 1000
        }
        
        # Mock get_bar method
        def mock_get_bar(offset):
            return {
                'open': close_price - 5 + offset,
                'high': close_price + 10 + offset,
                'low': close_price - 10 + offset,
                'close': close_price + offset * 10,  # Create momentum
                'volume': 1000
            }
        
        context.get_bar = mock_get_bar
        return context
    
    def test_momentum_strategy_init(self):
        """Test momentum strategy initialization."""
        params = MomentumParams(lookback=20, holding_period=5)
        strategy = Momentum(params)
        
        assert strategy.params.lookback == 20
        assert strategy.params.holding_period == 5
        assert strategy.position_entry_bar is None
    
    def test_momentum_can_trade(self):
        """Test momentum strategy trading conditions."""
        params = MomentumParams(lookback=20, sma_period=50)
        strategy = Momentum(params)
        
        # Not enough data initially
        context = self.create_mock_context(10)
        assert not strategy.can_trade(context)
        
        # Enough data
        context = self.create_mock_context(60)
        assert strategy.can_trade(context)
    
    def test_momentum_calculation(self):
        """Test momentum calculation."""
        params = MomentumParams(lookback=20)
        strategy = Momentum(params)
        
        context = self.create_mock_context(30, close_price=15000.0)
        momentum = strategy.calculate_momentum(context)
        
        # Should detect upward momentum due to mock data
        assert momentum > 0
    
    def test_momentum_signal_generation(self):
        """Test momentum signal generation."""
        params = MomentumParams(
            lookback=20, 
            min_momentum_threshold=0.01,
            use_sma_filter=False
        )
        strategy = Momentum(params)
        
        # Test with sufficient data and momentum
        context = self.create_mock_context(30, position=0)
        signal = strategy.next(context)
        
        # Should generate buy signal due to positive momentum
        if signal:
            assert signal['action'] == 'buy'


class TestVolBreakoutStrategy:
    """Test volatility breakout strategy."""
    
    def test_vol_breakout_init(self):
        """Test volatility breakout initialization."""
        params = VolBreakoutParams(atr_period=14)
        strategy = VolBreakout(params)
        
        assert strategy.params.atr_period == 14
        assert strategy.entry_price is None
        assert strategy.atr is not None
    
    def test_vol_breakout_can_trade(self):
        """Test trading conditions."""
        params = VolBreakoutParams(atr_period=14, trend_period=50)
        strategy = VolBreakout(params)
        
        context = Mock(spec=StrategyContext)
        context.bar_index = 10
        strategy.atr.ready = False
        assert not strategy.can_trade(context)
        
        context.bar_index = 60
        strategy.atr.ready = True
        assert strategy.can_trade(context)


class TestPairsStrategy:
    """Test pairs trading strategy."""
    
    def test_pairs_init(self):
        """Test pairs strategy initialization."""
        params = PairsParams(lookback=60)
        strategy = Pairs(params)
        
        assert strategy.params.lookback == 60
        assert len(strategy.spread_history) == 0
        assert strategy.current_position is None
    
    def test_pairs_second_asset_simulation(self):
        """Test second asset price simulation."""
        params = PairsParams()
        strategy = Pairs(params)
        
        # First price
        price2 = strategy.simulate_second_asset(15000.0)
        assert price2 > 0
        
        # Add to history and test correlation
        strategy.price1_history.append(15000.0)
        strategy.price2_history.append(price2)
        
        price2_next = strategy.simulate_second_asset(15100.0)
        assert price2_next > 0
    
    def test_pairs_hedge_ratio_calculation(self):
        """Test hedge ratio calculation methods."""
        params = PairsParams(hedge_ratio_method="simple")
        strategy = Pairs(params)
        
        # Add some test data
        for i in range(25):
            strategy.price1_history.append(15000 + i * 10)
            strategy.price2_history.append(14500 + i * 9)
        
        ratio = strategy.calculate_hedge_ratio()
        assert ratio > 0


class TestIndicatorIntegration:
    """Test indicator integration with strategies."""
    
    def test_no_lookahead_bias(self):
        """Test that indicators don't use future data."""
        sma = SMA(period=5)
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        results = []
        for price in prices:
            result = sma.update(price)
            results.append(result)
        
        # First 4 results should be None
        assert all(r is None for r in results[:4])
        
        # 5th result should use only first 5 prices
        assert results[4] == 12.0  # (10+11+12+13+14)/5
        
        # 6th result should use prices 2-6
        assert results[5] == 13.0  # (11+12+13+14+15)/5
    
    def test_indicator_reset(self):
        """Test indicator reset functionality."""
        rsi = RSI(period=5)
        
        # Update with some data
        for i in range(10):
            rsi.update(100 + i)
        
        assert rsi.ready
        
        # Reset
        rsi.reset()
        assert not rsi.ready
        assert rsi.prev_close is None


if __name__ == "__main__":
    pytest.main([__file__])