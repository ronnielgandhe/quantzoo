"""Technical analysis indicators with rolling, past-safe implementations."""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from collections import deque


class RollingIndicator:
    """Base class for rolling window indicators."""
    
    def __init__(self, period: int):
        self.period = period
        self.values = deque(maxlen=period)
        self.ready = False
    
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value."""
        self.values.append(value)
        
        if len(self.values) == self.period:
            self.ready = True
            return self._calculate()
        return None
    
    def _calculate(self) -> float:
        """Calculate indicator value. Override in subclasses."""
        raise NotImplementedError
    
    def reset(self):
        """Reset indicator state."""
        self.values.clear()
        self.ready = False


class SMA(RollingIndicator):
    """Simple Moving Average."""
    
    def _calculate(self) -> float:
        return np.mean(self.values)


class EMA:
    """Exponential Moving Average."""
    
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value = None
        self.ready = False
    
    def update(self, price: float) -> Optional[float]:
        """Update EMA with new price."""
        if self.value is None:
            self.value = price
            self.ready = True
        else:
            self.value = self.alpha * price + (1 - self.alpha) * self.value
        
        return self.value
    
    def reset(self):
        """Reset EMA state."""
        self.value = None
        self.ready = False


class ATR:
    """Average True Range."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_ema = EMA(period)
        self.prev_close = None
        self.ready = False
    
    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update ATR with OHLC data."""
        if self.prev_close is None:
            # First bar - use high-low
            tr = high - low
        else:
            # True Range = max(H-L, |H-Cp|, |L-Cp|)
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        
        self.prev_close = close
        atr_value = self.tr_ema.update(tr)
        
        if atr_value is not None:
            self.ready = True
        
        return atr_value
    
    def reset(self):
        """Reset ATR state."""
        self.tr_ema.reset()
        self.prev_close = None
        self.ready = False


class RSI:
    """Relative Strength Index."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.gain_ema = EMA(period)
        self.loss_ema = EMA(period)
        self.prev_close = None
        self.ready = False
    
    def update(self, close: float) -> Optional[float]:
        """Update RSI with close price."""
        if self.prev_close is None:
            self.prev_close = close
            return None
        
        change = close - self.prev_close
        gain = max(change, 0)
        loss = max(-change, 0)
        
        avg_gain = self.gain_ema.update(gain)
        avg_loss = self.loss_ema.update(loss)
        
        self.prev_close = close
        
        if avg_gain is not None and avg_loss is not None:
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            self.ready = True
            return rsi
        
        return None
    
    def reset(self):
        """Reset RSI state."""
        self.gain_ema.reset()
        self.loss_ema.reset()
        self.prev_close = None
        self.ready = False


class MFI:
    """Money Flow Index."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.pos_flows = deque(maxlen=period)
        self.neg_flows = deque(maxlen=period)
        self.prev_typical_price = None
        self.ready = False
    
    def update(self, high: float, low: float, close: float, volume: float) -> Optional[float]:
        """Update MFI with OHLCV data."""
        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume
        
        if self.prev_typical_price is None:
            # First bar - neutral flow
            self.pos_flows.append(0.0)
            self.neg_flows.append(0.0)
        else:
            if typical_price > self.prev_typical_price:
                self.pos_flows.append(raw_money_flow)
                self.neg_flows.append(0.0)
            elif typical_price < self.prev_typical_price:
                self.pos_flows.append(0.0)
                self.neg_flows.append(raw_money_flow)
            else:
                self.pos_flows.append(0.0)
                self.neg_flows.append(0.0)
        
        self.prev_typical_price = typical_price
        
        if len(self.pos_flows) == self.period:
            pos_mf = sum(self.pos_flows)
            neg_mf = sum(self.neg_flows)
            
            if neg_mf == 0:
                mfi = 100.0
            else:
                money_ratio = pos_mf / neg_mf
                mfi = 100 - (100 / (1 + money_ratio))
            
            self.ready = True
            return mfi
        
        return None
    
    def reset(self):
        """Reset MFI state."""
        self.pos_flows.clear()
        self.neg_flows.clear()
        self.prev_typical_price = None
        self.ready = False


class MACD:
    """Moving Average Convergence Divergence."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        self.ready = False
    
    def update(self, close: float) -> Optional[dict]:
        """Update MACD with close price."""
        fast_value = self.fast_ema.update(close)
        slow_value = self.slow_ema.update(close)
        
        if fast_value is None or slow_value is None:
            return None
        
        macd_line = fast_value - slow_value
        signal_line = self.signal_ema.update(macd_line)
        
        if signal_line is not None:
            histogram = macd_line - signal_line
            self.ready = True
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        
        return None
    
    def reset(self):
        """Reset MACD state."""
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.ready = False


class BollingerBands:
    """Bollinger Bands."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.sma = SMA(period)
        self.values = deque(maxlen=period)
        self.ready = False
    
    def update(self, close: float) -> Optional[dict]:
        """Update Bollinger Bands with close price."""
        self.values.append(close)
        middle = self.sma.update(close)
        
        if middle is None:
            return None
        
        if len(self.values) == self.period:
            std = np.std(self.values, ddof=1)
            upper = middle + (self.std_dev * std)
            lower = middle - (self.std_dev * std)
            
            self.ready = True
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'bandwidth': (upper - lower) / middle,
                'percent_b': (close - lower) / (upper - lower) if upper != lower else 0.5
            }
        
        return None
    
    def reset(self):
        """Reset Bollinger Bands state."""
        self.sma.reset()
        self.values.clear()
        self.ready = False


# Utility functions for vectorized calculations
def sma_series(data: Union[pd.Series, List[float]], period: int) -> pd.Series:
    """Calculate SMA for entire series."""
    if isinstance(data, list):
        data = pd.Series(data)
    return data.rolling(window=period, min_periods=period).mean()


def ema_series(data: Union[pd.Series, List[float]], period: int) -> pd.Series:
    """Calculate EMA for entire series."""
    if isinstance(data, list):
        data = pd.Series(data)
    return data.ewm(span=period, adjust=False).mean()


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR for entire series."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for entire series."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi