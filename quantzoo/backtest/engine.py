"""Backtesting engine with event loop and strategy context."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
import pandas as pd
import numpy as np
import random
import time


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "long" or "short"
    pnl: float
    fees: float
    slippage: float
    entry_reason: str
    exit_reason: str


@dataclass
class Position:
    """Represents current position state."""
    size: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.size) < 1e-8
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 1e-8
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < -1e-8


class Strategy(Protocol):
    """Protocol for trading strategies."""
    
    def on_start(self, ctx: 'StrategyContext') -> None:
        """Called once at the start of backtesting."""
        ...
    
    def on_bar(self, ctx: 'StrategyContext', bar: pd.Series) -> None:
        """Called for each bar in the backtest."""
        ...


class StrategyContext:
    """Context object provided to strategies with market data and order methods."""
    
    def __init__(self, engine: 'BacktestEngine'):
        self.engine = engine
        self._bar_index = 0
        self._current_bar: Optional[pd.Series] = None
        
    def bar_index(self) -> int:
        """Get current bar index."""
        return self._bar_index
    
    def bar_confirmed(self) -> bool:
        """Check if current bar is confirmed (not real-time)."""
        return True  # Always true in backtesting
    
    def position_size(self) -> float:
        """Get current position size."""
        return self.engine.position.size
    
    def avg_price(self) -> float:
        """Get average entry price of position."""
        return self.engine.position.avg_price
    
    def in_session(self, start: str, end: str) -> bool:
        """Check if current bar is within session hours."""
        if self._current_bar is None:
            return False
        
        bar_time = self._current_bar.name.time()
        from datetime import time
        start_time = time.fromisoformat(start)
        end_time = time.fromisoformat(end)
        
        return start_time <= bar_time <= end_time
    
    def get_series(self, name: str, offset: int = 0) -> float:
        """Get historical data series value with offset."""
        if name not in self.engine.data.columns:
            raise ValueError(f"Series '{name}' not found in data")
        
        # Prevent future data access (only allow current and past data)
        if offset > 0:
            return float('nan')
        
        idx = self._bar_index + offset
        if idx < 0 or idx >= len(self.engine.data):
            return float('nan')
        
        return float(self.engine.data.iloc[idx][name])
    
    @property
    def open(self) -> float:
        """Current bar open price."""
        return self.get_series("open")
    
    @property
    def high(self) -> float:
        """Current bar high price."""
        return self.get_series("high")
    
    @property
    def low(self) -> float:
        """Current bar low price."""
        return self.get_series("low")
    
    @property
    def close(self) -> float:
        """Current bar close price."""
        return self.get_series("close")
    
    @property
    def volume(self) -> float:
        """Current bar volume."""
        return self.get_series("volume")
    
    def buy(self, qty: float, tag: str = "Long") -> None:
        """Place a buy order."""
        self.engine._place_order(qty, tag, "buy")
    
    def sell(self, qty: float, tag: str = "Short") -> None:
        """Place a sell order."""
        self.engine._place_order(-qty, tag, "sell")
    
    def close_position(self, reason: str = "Close") -> None:
        """Close current position."""
        if not self.engine.position.is_flat():
            self.engine._place_order(-self.engine.position.size, reason, "close")
    
    def set_exit(self, stop_loss: Optional[float] = None, 
                 trail_points: Optional[float] = None) -> None:
        """Set exit conditions for current position."""
        self.engine.stop_loss = stop_loss
        self.engine.trail_points = trail_points
        if trail_points and not self.engine.position.is_flat():
            # Initialize trailing stop
            current_price = self.close
            if self.engine.position.is_long():
                self.engine.trail_stop = current_price - trail_points
            else:
                self.engine.trail_stop = current_price + trail_points
    
    def current_position(self) -> Dict[str, Any]:
        """Get current position information for API access.
        
        Returns:
            Dictionary with side, qty, avg_price, entry_time
        """
        pos = self.engine.position
        entry_time = None
        
        if self.engine.entry_bar is not None:
            entry_time = self.engine.data.index[self.engine.entry_bar]
        
        if pos.is_flat():
            side = "flat"
        elif pos.is_long():
            side = "long"
        else:
            side = "short"
        
        return {
            "side": side,
            "qty": abs(pos.size),
            "avg_price": pos.avg_price,
            "entry_time": entry_time
        }
    
    def unrealized_pnl(self) -> Dict[str, float]:
        """Get unrealized PnL information for API access.
        
        Returns:
            Dictionary with dollars and percent
        """
        pos = self.engine.position
        
        if pos.is_flat():
            return {"dollars": 0.0, "percent": 0.0}
        
        # Calculate percentage based on notional value
        notional = abs(pos.size) * pos.avg_price
        pnl_percent = (pos.unrealized_pnl / notional * 100) if notional > 0 else 0.0
        
        return {
            "dollars": pos.unrealized_pnl,
            "percent": pnl_percent
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest engine."""
    initial_capital: float = 100000.0
    fees_bps: float = 1.0  # Fees in basis points per trade
    slippage_bps: float = 1.0  # Slippage in basis points
    seed: int = 42


class BacktestEngine:
    """Main backtesting engine with event loop."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.position = Position()
        self.cash = config.initial_capital
        self.equity = config.initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        
        # Exit conditions
        self.stop_loss: Optional[float] = None
        self.trail_points: Optional[float] = None
        self.trail_stop: Optional[float] = None
        
        # Entry tracking for adverse exit rule
        self.entry_bar: Optional[int] = None
        self.entry_price: Optional[float] = None
        self.entry_fees: float = 0.0  # Track entry fees for complete trade accounting
        self.entry_slippage: float = 0.0  # Track entry slippage for complete trade accounting
        
        # Latency tracking
        self.bar_latencies: List[float] = []
        self.total_start_time: Optional[float] = None
        self.current_bar_index: int = 0  # Track current bar index
        
        # Random state for slippage
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def run(self, data: pd.DataFrame, strategy: Strategy) -> Dict[str, Any]:
        """
        Run backtest on provided data with given strategy.
        
        Args:
            data: OHLCV DataFrame with datetime index
            strategy: Strategy instance to backtest
            
        Returns:
            Dictionary with backtest results including latency metrics
        """
        self.data = data
        self.total_start_time = time.time()
        
        # Initialize strategy
        ctx = StrategyContext(self)
        ctx._bar_index = 0
        strategy.on_start(ctx)
        
        # Main event loop
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            bar_start_time = time.time()
            
            self.current_bar_index = i  # Track current bar index
            ctx = StrategyContext(self)
            ctx._bar_index = i
            ctx._current_bar = bar
            
            # Update equity curve
            self._update_equity(bar.close)
            
            # Check exit conditions
            self._check_exits(bar, i)
            
            # Update trailing stop
            self._update_trailing_stop(bar.close)
            
            # Call strategy
            strategy.on_bar(ctx, bar)
            
            # Record bar processing time
            bar_latency = (time.time() - bar_start_time) * 1000  # Convert to milliseconds
            self.bar_latencies.append(bar_latency)
        
        total_runtime = time.time() - self.total_start_time
        
        # Calculate latency statistics
        latency_stats = self._calculate_latency_stats()
        
        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "final_equity": self.equity,
            "final_cash": self.cash,
            "latency_metrics": latency_stats,
            "total_runtime_seconds": total_runtime
        }
    
    def _place_order(self, qty: float, tag: str, order_type: str) -> None:
        """Execute an order with fees and slippage."""
        if abs(qty) < 1e-8:
            return
        
        current_price = self.data.iloc[self.current_bar_index]["close"]
        
        # Apply slippage
        slippage_factor = self.config.slippage_bps / 10000.0
        slippage_amount = slippage_factor * current_price * np.random.uniform(-1, 1)
        execution_price = current_price + slippage_amount
        
        # Calculate fees
        notional = abs(qty * execution_price)
        fees = notional * self.config.fees_bps / 10000.0
        
        # Update position
        if self.position.is_flat():
            # New position
            self.position.size = qty
            self.position.avg_price = execution_price
            self.entry_bar = self.current_bar_index
            self.entry_price = execution_price
            self.entry_fees = fees  # Track entry fees
            self.entry_slippage = abs(slippage_amount * abs(qty))  # Track entry slippage
        else:
            # Modify existing position
            old_size = self.position.size
            old_avg_price = self.position.avg_price
            
            new_size = old_size + qty
            
            if abs(new_size) < 1e-8:
                # Position closed
                gross_pnl = self._calculate_pnl(old_size, old_avg_price, execution_price)
                total_fees = self.entry_fees + fees  # Entry fees + exit fees
                exit_slippage = abs(slippage_amount * abs(old_size))  # Exit slippage
                total_slippage = self.entry_slippage + exit_slippage  # Total slippage
                net_pnl = gross_pnl - total_fees - total_slippage  # Net PnL after costs
                
                trade = Trade(
                    entry_time=self.data.index[self.entry_bar] if self.entry_bar else self.data.index[0],
                    exit_time=self.data.index[self.current_bar_index],
                    entry_price=old_avg_price,
                    exit_price=execution_price,
                    quantity=abs(old_size),
                    side="long" if old_size > 0 else "short",
                    pnl=net_pnl,
                    fees=total_fees,
                    slippage=total_slippage,
                    entry_reason="Entry",
                    exit_reason=tag,
                )
                self.trades.append(trade)
                
                # Reset position
                self.position = Position()
                self.entry_bar = None
                self.entry_price = None
                self.entry_fees = 0.0  # Reset entry fees
                self.entry_slippage = 0.0  # Reset entry slippage
                self.stop_loss = None
                self.trail_points = None
                self.trail_stop = None
            else:
                # Position partially closed or increased
                if np.sign(new_size) == np.sign(old_size):
                    # Position increased
                    total_cost = old_size * old_avg_price + qty * execution_price
                    self.position.avg_price = total_cost / new_size
                else:
                    # Position partially closed
                    closed_qty = min(abs(old_size), abs(qty))
                    pnl = self._calculate_pnl(
                        closed_qty * np.sign(old_size), 
                        old_avg_price, 
                        execution_price
                    )
                    trade = Trade(
                        entry_time=self.data.index[self.entry_bar] if self.entry_bar else self.data.index[0],
                        exit_time=self.data.index[len(self.equity_curve) - 1],
                        entry_price=old_avg_price,
                        exit_price=execution_price,
                        quantity=closed_qty,
                        side="long" if old_size > 0 else "short",
                        pnl=pnl,
                        fees=fees * (closed_qty / abs(qty)),
                        slippage=abs(slippage_amount * closed_qty),
                        entry_reason="Entry",
                        exit_reason=tag,
                    )
                    self.trades.append(trade)
                
                self.position.size = new_size
        
        # Update cash
        self.cash -= qty * execution_price + fees
    
    def _calculate_pnl(self, size: float, entry_price: float, exit_price: float) -> float:
        """Calculate PnL for a trade."""
        return size * (exit_price - entry_price)
    
    def _update_equity(self, current_price: float) -> None:
        """Update equity based on current market price."""
        if self.position.is_flat():
            self.position.unrealized_pnl = 0.0
        else:
            self.position.unrealized_pnl = self._calculate_pnl(
                self.position.size, self.position.avg_price, current_price
            )
        
        self.equity = self.cash + self.position.unrealized_pnl
    
    def _check_exits(self, bar: pd.Series, bar_index: int) -> None:
        """Check and execute exit conditions."""
        if self.position.is_flat():
            return
        
        current_price = bar["close"]
        
        # One bar adverse exit rule
        if (self.entry_bar is not None and 
            bar_index == self.entry_bar + 1 and 
            self.entry_price is not None):
            
            should_exit = False
            if self.position.is_long() and current_price < self.entry_price:
                should_exit = True
            elif self.position.is_short() and current_price > self.entry_price:
                should_exit = True
            
            if should_exit:
                self._place_order(-self.position.size, "OneBarAdverse", "exit")
                return
        
        # Stop loss
        if self.stop_loss is not None:
            should_exit = False
            if self.position.is_long() and current_price <= self.stop_loss:
                should_exit = True
            elif self.position.is_short() and current_price >= self.stop_loss:
                should_exit = True
            
            if should_exit:
                self._place_order(-self.position.size, "StopLoss", "exit")
                return
        
        # Trailing stop
        if self.trail_stop is not None:
            should_exit = False
            if self.position.is_long() and current_price <= self.trail_stop:
                should_exit = True
            elif self.position.is_short() and current_price >= self.trail_stop:
                should_exit = True
            
            if should_exit:
                self._place_order(-self.position.size, "TrailingStop", "exit")
                return
    
    def _update_trailing_stop(self, current_price: float) -> None:
        """Update trailing stop level."""
        if self.trail_points is None or self.position.is_flat():
            return
        
        if self.position.is_long():
            new_trail_stop = current_price - self.trail_points
            if self.trail_stop is None or new_trail_stop > self.trail_stop:
                self.trail_stop = new_trail_stop
        else:
            new_trail_stop = current_price + self.trail_points
            if self.trail_stop is None or new_trail_stop < self.trail_stop:
                self.trail_stop = new_trail_stop
    
    def _calculate_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics from bar processing times."""
        if not self.bar_latencies:
            return {
                "latency_ms_mean": 0.0,
                "latency_ms_p95": 0.0,
                "latency_ms_p99": 0.0,
                "latency_ms_max": 0.0
            }
        
        latencies = np.array(self.bar_latencies)
        
        return {
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "latency_ms_p99": float(np.percentile(latencies, 99)),
            "latency_ms_max": float(np.max(latencies))
        }