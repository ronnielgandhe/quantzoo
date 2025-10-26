"""Pairs trading strategy with z-score mean reversion."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from collections import deque

from quantzoo.backtest.engine import Strategy, StrategyContext


@dataclass
class PairsParams:
    """Parameters for pairs trading strategy."""
    lookback: int = 60  # Lookback period for hedge ratio calculation
    zscore_entry: float = 2.0  # Z-score threshold for entry
    zscore_exit: float = 0.5  # Z-score threshold for exit
    max_holding_period: int = 20  # Maximum holding period
    contracts: int = 1
    hedge_ratio_method: str = "simple"  # simple, ols, rolling
    min_correlation: float = 0.7  # Minimum correlation to trade


class Pairs(Strategy):
    """Pairs trading strategy with z-score mean reversion.
    
    Strategy Logic:
    1. Calculate hedge ratio between two correlated assets
    2. Compute spread = asset1 - hedge_ratio * asset2
    3. Calculate z-score of spread over lookback period
    4. Enter long spread when z-score < -entry_threshold
    5. Enter short spread when z-score > entry_threshold
    6. Exit when z-score crosses back through exit_threshold
    7. Force exit after max_holding_period
    
    Note: This is a simplified pairs strategy for demonstration.
    In practice, you would need actual price data for both assets.
    """
    
    def __init__(self, params: PairsParams):
        self.params = params
        
        # Spread calculation
        self.price1_history = deque(maxlen=params.lookback)
        self.price2_history = deque(maxlen=params.lookback)  # Simulated second asset
        self.spread_history = deque(maxlen=params.lookback)
        
        # Position tracking
        self.entry_bar = None
        self.entry_zscore = None
        self.current_position = None  # "long_spread" or "short_spread"
        
        # Hedge ratio
        self.hedge_ratio = 1.0
        
    def can_trade(self, context: StrategyContext) -> bool:
        """Check if we have enough data to calculate statistics."""
        return len(self.spread_history) >= self.params.lookback
    
    def simulate_second_asset(self, price1: float) -> float:
        """Simulate correlated second asset price.
        
        In practice, this would be actual market data for the second asset.
        For demonstration, we create a correlated price series.
        """
        # Simple simulation: add some noise but maintain correlation
        base_correlation = 0.8
        noise_factor = 0.1
        
        if len(self.price1_history) == 0:
            return price1 * 0.95  # Start slightly lower
        
        prev_price1 = self.price1_history[-1]
        prev_price2 = self.price2_history[-1] if self.price2_history else price1 * 0.95
        
        # Calculate price1 return
        price1_return = (price1 - prev_price1) / prev_price1 if prev_price1 != 0 else 0
        
        # Create correlated return for price2
        correlated_return = base_correlation * price1_return
        noise = np.random.normal(0, noise_factor * abs(price1_return)) if price1_return != 0 else 0
        price2_return = correlated_return + noise
        
        # Calculate new price2
        price2 = prev_price2 * (1 + price2_return)
        return max(price2, 0.01)  # Ensure positive price
    
    def calculate_hedge_ratio(self) -> float:
        """Calculate hedge ratio between the two assets."""
        if len(self.price1_history) < 20:  # Need minimum data
            return 1.0
        
        prices1 = np.array(self.price1_history)
        prices2 = np.array(self.price2_history)
        
        if self.params.hedge_ratio_method == "simple":
            # Simple ratio of average prices
            return np.mean(prices1) / np.mean(prices2) if np.mean(prices2) != 0 else 1.0
        
        elif self.params.hedge_ratio_method == "ols":
            # Ordinary least squares regression
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(prices2.reshape(-1, 1), prices1)
                return model.coef_[0]
            except:
                # Fallback to correlation-based ratio
                correlation = np.corrcoef(prices1, prices2)[0, 1]
                std_ratio = np.std(prices1) / np.std(prices2) if np.std(prices2) != 0 else 1.0
                return correlation * std_ratio
        
        else:  # rolling
            # Rolling correlation-based hedge ratio
            if len(prices1) >= 20:
                recent_corr = np.corrcoef(prices1[-20:], prices2[-20:])[0, 1]
                recent_std_ratio = np.std(prices1[-20:]) / np.std(prices2[-20:]) if np.std(prices2[-20:]) != 0 else 1.0
                return recent_corr * recent_std_ratio
            else:
                return 1.0
    
    def calculate_zscore(self) -> Optional[float]:
        """Calculate z-score of current spread."""
        if len(self.spread_history) < self.params.lookback:
            return None
        
        spreads = np.array(self.spread_history)
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads, ddof=1)
        
        if std_spread == 0:
            return 0.0
        
        current_spread = spreads[-1]
        zscore = (current_spread - mean_spread) / std_spread
        return zscore
    
    def check_correlation(self) -> float:
        """Check correlation between assets."""
        if len(self.price1_history) < 20:
            return 0.0
        
        prices1 = np.array(self.price1_history[-20:])
        prices2 = np.array(self.price2_history[-20:])
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def should_force_exit(self, context: StrategyContext) -> bool:
        """Check if position should be force-exited due to time limit."""
        if self.entry_bar is None:
            return False
        
        bars_held = context.bar_index - self.entry_bar
        return bars_held >= self.params.max_holding_period
    
    def next(self, context: StrategyContext) -> Optional[Dict[str, Any]]:
        """Strategy logic for each bar."""
        current_price1 = context.current_bar['close']
        current_price2 = self.simulate_second_asset(current_price1)
        
        # Update price histories
        self.price1_history.append(current_price1)
        self.price2_history.append(current_price2)
        
        # Calculate hedge ratio
        self.hedge_ratio = self.calculate_hedge_ratio()
        
        # Calculate spread
        spread = current_price1 - (self.hedge_ratio * current_price2)
        self.spread_history.append(spread)
        
        # Force exit if holding too long
        if context.position != 0 and self.should_force_exit(context):
            self.entry_bar = None
            self.entry_zscore = None
            self.current_position = None
            return {
                'action': 'close',
                'reason': f'max_holding_period_{self.params.max_holding_period}'
            }
        
        # Check if we can trade
        if not self.can_trade(context):
            return None
        
        # Check correlation requirement
        correlation = self.check_correlation()
        if abs(correlation) < self.params.min_correlation:
            return None
        
        # Calculate z-score
        zscore = self.calculate_zscore()
        if zscore is None:
            return None
        
        # Exit logic
        if context.position != 0:
            should_exit = False
            exit_reason = ""
            
            if self.current_position == "long_spread":
                # Exit long spread when z-score rises above exit threshold
                if zscore > -self.params.zscore_exit:
                    should_exit = True
                    exit_reason = f"zscore_exit_long_{zscore:.2f}"
            
            elif self.current_position == "short_spread":
                # Exit short spread when z-score falls below exit threshold
                if zscore < self.params.zscore_exit:
                    should_exit = True
                    exit_reason = f"zscore_exit_short_{zscore:.2f}"
            
            if should_exit:
                self.entry_bar = None
                self.entry_zscore = None
                self.current_position = None
                return {
                    'action': 'close',
                    'reason': exit_reason
                }
        
        # Entry logic
        if context.position == 0:
            if zscore < -self.params.zscore_entry:
                # Z-score is very negative - spread is cheap, go long spread
                # Long spread = Long asset1, Short asset2 (but we only trade asset1)
                self.entry_bar = context.bar_index
                self.entry_zscore = zscore
                self.current_position = "long_spread"
                
                return {
                    'action': 'buy',
                    'quantity': self.params.contracts,
                    'reason': f'long_spread_zscore_{zscore:.2f}_hedge_{self.hedge_ratio:.3f}'
                }
            
            elif zscore > self.params.zscore_entry:
                # Z-score is very positive - spread is expensive, go short spread
                # Short spread = Short asset1, Long asset2 (but we only trade asset1)
                self.entry_bar = context.bar_index
                self.entry_zscore = zscore
                self.current_position = "short_spread"
                
                return {
                    'action': 'sell',
                    'quantity': self.params.contracts,
                    'reason': f'short_spread_zscore_{zscore:.2f}_hedge_{self.hedge_ratio:.3f}'
                }
        
        return None