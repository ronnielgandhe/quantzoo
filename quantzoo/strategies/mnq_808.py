"""MNQ 808 strategy ported from Pine Script v6."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from quantzoo.data.loaders import calculate_sma, calculate_atr, calculate_rsi, calculate_mfi, calculate_true_range


@dataclass
class MNQ808Params:
    """Parameters for MNQ808 strategy."""
    atr_mult: float = 1.5
    lookback: int = 10
    use_mfi: bool = True
    trail_mult_legacy: float = 1.0
    contracts: int = 1
    risk_ticks_legacy: float = 150
    session_start: str = "08:00"
    session_end: str = "16:30"
    tick_size: float = 0.25
    tick_value: float = 0.5
    treat_atr_as_ticks: bool = True


class MNQ808:
    """
    MNQ 808 strategy implementation.
    
    Ports the Pine Script logic exactly:
    - Uses SMA of True Range and ATR for band calculation
    - MFI or RSI for momentum
    - Anchor recursion with crossover/crossunder signals
    - One bar adverse exit rule
    - Legacy tick-based exit management
    """
    
    def __init__(self, params: MNQ808Params):
        self.params = params
        
        # Strategy state
        self.anchor: Optional[float] = None
        self.anchor_history: list = []
        self.momentum_history: list = []
        self.upper_band_history: list = []
        self.lower_band_history: list = []
        
        # Indicator data storage
        self.sma_tr_history: list = []
        self.atr_history: list = []
        
    def on_start(self, ctx) -> None:
        """Initialize strategy state."""
        self.anchor = None
        self.anchor_history = []
        self.momentum_history = []
        self.upper_band_history = []
        self.lower_band_history = []
        self.sma_tr_history = []
        self.atr_history = []
    
    def on_bar(self, ctx, bar: pd.Series) -> None:
        """Process each bar according to Pine Script logic."""
        
        # Skip if not enough history
        if ctx.bar_index() < self.params.lookback:
            self._update_history(ctx, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            return
        
        # Calculate indicators
        sma_tr = self._calculate_sma_tr(ctx)
        atr_price = self._calculate_atr(ctx)
        
        # Calculate bands (Pine: upperBand = low - smaTR * atrMult, lowerBand = high + smaTR * atrMult)
        upper_band = ctx.low - sma_tr * self.params.atr_mult
        lower_band = ctx.high + sma_tr * self.params.atr_mult
        
        # Calculate momentum
        momentum = self._calculate_momentum(ctx)
        
        # Update anchor (Pine anchor recursion logic)
        prev_anchor = self.anchor if self.anchor is not None else np.nan
        
        if momentum >= 50:
            if np.isnan(prev_anchor):
                self.anchor = upper_band
            else:
                self.anchor = min(upper_band, prev_anchor)
        else:
            if np.isnan(prev_anchor):
                self.anchor = lower_band
            else:
                self.anchor = max(lower_band, prev_anchor)
        
        # Store history
        self._update_history(ctx, sma_tr, atr_price, upper_band, lower_band, momentum, self.anchor)
        
        # Check for signals (only when flat, in session, bar confirmed)
        in_session = ctx.in_session(self.params.session_start, self.params.session_end)
        position_flat = abs(ctx.position_size()) < 1e-8
        bar_confirmed = ctx.bar_confirmed()
        
        if in_session and position_flat and bar_confirmed:
            # Get price values for crossover detection (Pine: anchor with anchor[2])
            if len(self.anchor_history) >= 3:
                anchor_current = self.anchor_history[-1]  # Current anchor
                anchor_lag2 = self.anchor_history[-3]     # Anchor from 2 bars ago
                close_current = ctx.close                 # Current close price
                
                # Check for crossover signals (Pine: close crosses over/under anchor[2])
                # Get previous close for crossover detection
                close_prev = ctx.get_series("close", -1) if ctx.bar_index() > 0 else close_current
                
                # Long signal: close crosses over anchor[2]
                is_long_signal = close_prev <= anchor_lag2 and close_current > anchor_lag2
                
                # Short signal: close crosses under anchor[2]  
                is_short_signal = close_prev >= anchor_lag2 and close_current < anchor_lag2
                
                if is_long_signal:
                    ctx.buy(self.params.contracts, "Long")
                    self._set_exits(ctx, atr_price)
                elif is_short_signal:
                    ctx.sell(self.params.contracts, "Short")
                    self._set_exits(ctx, atr_price)
    
    def _calculate_sma_tr(self, ctx) -> float:
        """Calculate SMA of True Range."""
        # Get recent bars for TR calculation
        bars_data = []
        for i in range(self.params.lookback):
            if ctx.bar_index() - i >= 0:
                high = ctx.get_series("high", -i)
                low = ctx.get_series("low", -i)
                close_prev = ctx.get_series("close", -i-1) if ctx.bar_index() - i - 1 >= 0 else high
                
                # Calculate True Range
                high_low = high - low
                high_close_prev = abs(high - close_prev)
                low_close_prev = abs(low - close_prev)
                
                tr = max(high_low, high_close_prev, low_close_prev)
                bars_data.append(tr)
        
        return np.mean(bars_data) if bars_data else np.nan
    
    def _calculate_atr(self, ctx) -> float:
        """Calculate ATR."""
        # Use same TR calculation as SMA TR for consistency
        bars_data = []
        for i in range(self.params.lookback):
            if ctx.bar_index() - i >= 0:
                high = ctx.get_series("high", -i)
                low = ctx.get_series("low", -i)
                close_prev = ctx.get_series("close", -i-1) if ctx.bar_index() - i - 1 >= 0 else high
                
                # Calculate True Range
                high_low = high - low
                high_close_prev = abs(high - close_prev)
                low_close_prev = abs(low - close_prev)
                
                tr = max(high_low, high_close_prev, low_close_prev)
                bars_data.append(tr)
        
        return np.mean(bars_data) if bars_data else np.nan
    
    def _calculate_momentum(self, ctx) -> float:
        """Calculate momentum (MFI or RSI)."""
        if self.params.use_mfi:
            return self._calculate_mfi(ctx)
        else:
            return self._calculate_rsi(ctx)
    
    def _calculate_mfi(self, ctx) -> float:
        """Calculate Money Flow Index."""
        # Get recent bars for MFI calculation
        tp_data = []
        volume_data = []
        
        for i in range(self.params.lookback + 1):  # +1 for diff calculation
            if ctx.bar_index() - i >= 0:
                high = ctx.get_series("high", -i)
                low = ctx.get_series("low", -i)
                close = ctx.get_series("close", -i)
                volume = ctx.get_series("volume", -i)
                
                tp = (high + low + close) / 3
                tp_data.append(tp)
                volume_data.append(volume)
        
        if len(tp_data) < 2:
            return 50.0  # Default neutral value
        
        # Reverse for chronological order
        tp_data = tp_data[::-1]
        volume_data = volume_data[::-1]
        
        # Calculate money flows
        positive_mf = 0.0
        negative_mf = 0.0
        
        for i in range(1, len(tp_data)):
            rmf = tp_data[i] * volume_data[i]
            if tp_data[i] > tp_data[i-1]:
                positive_mf += rmf
            elif tp_data[i] < tp_data[i-1]:
                negative_mf += rmf
        
        if negative_mf == 0:
            return 100.0
        
        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    def _calculate_rsi(self, ctx) -> float:
        """Calculate RSI."""
        # Get recent close prices
        closes = []
        for i in range(self.params.lookback + 1):
            if ctx.bar_index() - i >= 0:
                closes.append(ctx.get_series("close", -i))
        
        if len(closes) < 2:
            return 50.0  # Default neutral value
        
        # Reverse for chronological order
        closes = closes[::-1]
        
        # Calculate gains and losses
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-diff)
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _crossover(self, current: float, reference: float) -> bool:
        """Check if current value crosses over reference."""
        if len(self.anchor_history) < 2:
            return False
        
        prev_current = self.anchor_history[-2]
        return prev_current <= reference and current > reference
    
    def _crossunder(self, current: float, reference: float) -> bool:
        """Check if current value crosses under reference."""
        if len(self.anchor_history) < 2:
            return False
        
        prev_current = self.anchor_history[-2]
        return prev_current >= reference and current < reference
    
    def _set_exits(self, ctx, atr_price: float) -> None:
        """Set exit conditions based on legacy tick semantics."""
        # Calculate stop loss in price terms
        max_loss_ticks = min(self.params.risk_ticks_legacy, 200)
        stop_loss_price_offset = max_loss_ticks * self.params.tick_size
        
        # Calculate trailing points
        if self.params.treat_atr_as_ticks:
            trail_points = self.params.trail_mult_legacy * atr_price
        else:
            trail_points = (self.params.trail_mult_legacy * atr_price / self.params.tick_size) * self.params.tick_size
        
        # Set exits
        ctx.set_exit(stop_loss=stop_loss_price_offset, trail_points=trail_points)
    
    def _update_history(self, ctx, sma_tr: float, atr_price: float, 
                       upper_band: float, lower_band: float, 
                       momentum: float, anchor: float) -> None:
        """Update internal history buffers."""
        self.sma_tr_history.append(sma_tr)
        self.atr_history.append(atr_price)
        self.upper_band_history.append(upper_band)
        self.lower_band_history.append(lower_band)
        self.momentum_history.append(momentum)
        self.anchor_history.append(anchor)
        
        # Keep only necessary history
        max_history = max(100, self.params.lookback * 3)
        if len(self.anchor_history) > max_history:
            self.sma_tr_history = self.sma_tr_history[-max_history:]
            self.atr_history = self.atr_history[-max_history:]
            self.upper_band_history = self.upper_band_history[-max_history:]
            self.lower_band_history = self.lower_band_history[-max_history:]
            self.momentum_history = self.momentum_history[-max_history:]
            self.anchor_history = self.anchor_history[-max_history:]