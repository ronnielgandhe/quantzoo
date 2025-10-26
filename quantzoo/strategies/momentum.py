"""Momentum strategy based on time-series momentum."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from quantzoo.backtest.engine import Strategy, StrategyContext
from quantzoo.indicators.ta import SMA


@dataclass
class MomentumParams:
    """Parameters for momentum strategy."""
    lookback: int = 20  # Momentum lookback period
    holding_period: int = 5  # Holding period for positions
    min_momentum_threshold: float = 0.02  # Minimum momentum to trade (2%)
    contracts: int = 1
    use_sma_filter: bool = True  # Use SMA trend filter
    sma_period: int = 50


class Momentum(Strategy):
    """Time-series momentum strategy with lookback and holding period.
    
    Strategy Logic:
    1. Calculate returns over lookback period
    2. Enter long if momentum > threshold and price > SMA (if enabled)
    3. Enter short if momentum < -threshold and price < SMA (if enabled)
    4. Hold for specified holding period
    5. Exit after holding period expires
    """
    
    def __init__(self, params: MomentumParams):
        self.params = params
        self.sma = SMA(params.sma_period) if params.use_sma_filter else None
        self.position_entry_bar = None
        
        # Track momentum calculations
        self.price_history = []
        
    def can_trade(self, context: StrategyContext) -> bool:
        """Check if we have enough history to trade."""
        return context.bar_index >= max(self.params.lookback, self.params.sma_period)
    
    def calculate_momentum(self, context: StrategyContext) -> float:
        """Calculate momentum over lookback period."""
        if context.bar_index < self.params.lookback:
            return 0.0
        
        current_price = context.current_bar['close']
        lookback_price = context.get_bar(-self.params.lookback)['close']
        
        momentum = (current_price - lookback_price) / lookback_price
        return momentum
    
    def get_sma_signal(self, context: StrategyContext) -> Optional[str]:
        """Get SMA trend filter signal."""
        if not self.params.use_sma_filter or self.sma is None:
            return None
        
        current_price = context.current_bar['close']
        sma_value = self.sma.update(current_price)
        
        if sma_value is None:
            return None
        
        if current_price > sma_value:
            return "bullish"
        elif current_price < sma_value:
            return "bearish"
        else:
            return "neutral"
    
    def should_exit_position(self, context: StrategyContext) -> bool:
        """Check if position should be exited due to holding period."""
        if self.position_entry_bar is None:
            return False
        
        bars_held = context.bar_index - self.position_entry_bar
        return bars_held >= self.params.holding_period
    
    def next(self, context: StrategyContext) -> Optional[Dict[str, Any]]:
        """Strategy logic for each bar."""
        current_price = context.current_bar['close']
        
        # Update price history
        self.price_history.append(current_price)
        
        # Check if we should exit current position
        if context.position != 0 and self.should_exit_position(context):
            self.position_entry_bar = None
            return {
                'action': 'close',
                'reason': f'holding_period_expired_{self.params.holding_period}'
            }
        
        # Don't enter new positions if we already have one
        if context.position != 0:
            return None
        
        # Check if we can trade
        if not self.can_trade(context):
            return None
        
        # Calculate momentum
        momentum = self.calculate_momentum(context)
        
        # Get SMA filter signal
        sma_signal = self.get_sma_signal(context)
        
        # Generate trading signals
        if momentum > self.params.min_momentum_threshold:
            # Positive momentum - consider long
            if sma_signal in [None, "bullish", "neutral"]:
                self.position_entry_bar = context.bar_index
                return {
                    'action': 'buy',
                    'quantity': self.params.contracts,
                    'reason': f'momentum_long_{momentum:.4f}_sma_{sma_signal}'
                }
        elif momentum < -self.params.min_momentum_threshold:
            # Negative momentum - consider short
            if sma_signal in [None, "bearish", "neutral"]:
                self.position_entry_bar = context.bar_index
                return {
                    'action': 'sell',
                    'quantity': self.params.contracts,
                    'reason': f'momentum_short_{momentum:.4f}_sma_{sma_signal}'
                }
        
        return None