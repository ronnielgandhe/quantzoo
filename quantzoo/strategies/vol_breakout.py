"""Volatility breakout strategy using ATR-based stops and targets."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from quantzoo.backtest.engine import Strategy, StrategyContext
from quantzoo.indicators.ta import ATR, SMA


@dataclass
class VolBreakoutParams:
    """Parameters for volatility breakout strategy."""
    atr_period: int = 14  # ATR calculation period
    breakout_multiplier: float = 2.0  # ATR multiplier for breakout threshold
    stop_multiplier: float = 1.5  # ATR multiplier for stop loss
    target_multiplier: float = 3.0  # ATR multiplier for profit target
    min_atr: float = 5.0  # Minimum ATR to consider trading
    max_atr: float = 100.0  # Maximum ATR to consider trading
    contracts: int = 1
    use_trend_filter: bool = True
    trend_period: int = 50


class VolBreakout(Strategy):
    """Volatility breakout strategy with ATR-based stop and target.
    
    Strategy Logic:
    1. Calculate ATR for volatility measurement
    2. Identify breakouts: price moves > ATR * breakout_multiplier from recent range
    3. Enter long on upward breakout (with trend filter if enabled)
    4. Enter short on downward breakout (with trend filter if enabled)
    5. Set stop loss at entry ± ATR * stop_multiplier
    6. Set profit target at entry ± ATR * target_multiplier
    7. Exit when stop or target is hit
    """
    
    def __init__(self, params: VolBreakoutParams):
        self.params = params
        self.atr = ATR(params.atr_period)
        self.trend_sma = SMA(params.trend_period) if params.use_trend_filter else None
        
        # Position tracking
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.position_side = None
        
        # Price range tracking
        self.recent_high = None
        self.recent_low = None
        self.range_bars = 10  # Bars to look back for range
        
    def can_trade(self, context: StrategyContext) -> bool:
        """Check if we have enough data to trade."""
        return (context.bar_index >= max(self.params.atr_period, self.params.trend_period, self.range_bars) 
                and self.atr.ready)
    
    def update_price_range(self, context: StrategyContext):
        """Update recent price range."""
        if context.bar_index < self.range_bars:
            return
        
        # Get recent bars
        recent_highs = []
        recent_lows = []
        
        for i in range(self.range_bars):
            bar = context.get_bar(-i)
            recent_highs.append(bar['high'])
            recent_lows.append(bar['low'])
        
        self.recent_high = max(recent_highs)
        self.recent_low = min(recent_lows)
    
    def get_trend_signal(self, context: StrategyContext) -> Optional[str]:
        """Get trend filter signal."""
        if not self.params.use_trend_filter or self.trend_sma is None:
            return None
        
        current_price = context.current_bar['close']
        sma_value = self.trend_sma.update(current_price)
        
        if sma_value is None:
            return None
        
        if current_price > sma_value:
            return "uptrend"
        elif current_price < sma_value:
            return "downtrend"
        else:
            return "neutral"
    
    def check_exit_conditions(self, context: StrategyContext) -> Optional[Dict[str, Any]]:
        """Check if position should be exited."""
        if context.position == 0 or self.entry_price is None:
            return None
        
        current_high = context.current_bar['high']
        current_low = context.current_bar['low']
        current_close = context.current_bar['close']
        
        # Check stop loss
        if self.position_side == "long":
            if current_low <= self.stop_price:
                return {
                    'action': 'close',
                    'reason': f'stop_loss_hit_{self.stop_price:.2f}'
                }
            # Check profit target
            if current_high >= self.target_price:
                return {
                    'action': 'close',
                    'reason': f'profit_target_hit_{self.target_price:.2f}'
                }
        
        elif self.position_side == "short":
            if current_high >= self.stop_price:
                return {
                    'action': 'close',
                    'reason': f'stop_loss_hit_{self.stop_price:.2f}'
                }
            # Check profit target
            if current_low <= self.target_price:
                return {
                    'action': 'close',
                    'reason': f'profit_target_hit_{self.target_price:.2f}'
                }
        
        return None
    
    def detect_breakout(self, context: StrategyContext, atr_value: float) -> Optional[str]:
        """Detect breakout conditions."""
        if self.recent_high is None or self.recent_low is None:
            return None
        
        current_high = context.current_bar['high']
        current_low = context.current_bar['low']
        current_close = context.current_bar['close']
        
        breakout_threshold = atr_value * self.params.breakout_multiplier
        
        # Upward breakout
        if current_high > self.recent_high and (current_close - self.recent_low) > breakout_threshold:
            return "upward_breakout"
        
        # Downward breakout
        if current_low < self.recent_low and (self.recent_high - current_close) > breakout_threshold:
            return "downward_breakout"
        
        return None
    
    def next(self, context: StrategyContext) -> Optional[Dict[str, Any]]:
        """Strategy logic for each bar."""
        current_bar = context.current_bar
        
        # Update ATR
        atr_value = self.atr.update(current_bar['high'], current_bar['low'], current_bar['close'])
        
        # Check exit conditions first
        exit_signal = self.check_exit_conditions(context)
        if exit_signal:
            # Clear position tracking
            self.entry_price = None
            self.stop_price = None
            self.target_price = None
            self.position_side = None
            return exit_signal
        
        # Don't enter new position if we already have one
        if context.position != 0:
            return None
        
        # Check if we can trade
        if not self.can_trade(context):
            return None
        
        # Check ATR limits
        if atr_value < self.params.min_atr or atr_value > self.params.max_atr:
            return None
        
        # Update price range
        self.update_price_range(context)
        
        # Get trend signal
        trend_signal = self.get_trend_signal(context)
        
        # Detect breakout
        breakout = self.detect_breakout(context, atr_value)
        
        current_close = current_bar['close']
        
        if breakout == "upward_breakout":
            # Consider long entry
            if trend_signal in [None, "uptrend", "neutral"]:
                self.entry_price = current_close
                self.stop_price = self.entry_price - (atr_value * self.params.stop_multiplier)
                self.target_price = self.entry_price + (atr_value * self.params.target_multiplier)
                self.position_side = "long"
                
                return {
                    'action': 'buy',
                    'quantity': self.params.contracts,
                    'reason': f'upward_breakout_atr_{atr_value:.2f}_trend_{trend_signal}'
                }
        
        elif breakout == "downward_breakout":
            # Consider short entry
            if trend_signal in [None, "downtrend", "neutral"]:
                self.entry_price = current_close
                self.stop_price = self.entry_price + (atr_value * self.params.stop_multiplier)
                self.target_price = self.entry_price - (atr_value * self.params.target_multiplier)
                self.position_side = "short"
                
                return {
                    'action': 'sell',
                    'quantity': self.params.contracts,
                    'reason': f'downward_breakout_atr_{atr_value:.2f}_trend_{trend_signal}'
                }
        
        return None