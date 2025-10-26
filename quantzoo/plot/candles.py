"""TradingView-style candlestick charts with trade markers and overlays."""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import pytz
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def make_candles(
    df: pd.DataFrame, 
    trades: Optional[List[Dict[str, Any]]] = None,
    overlays: Optional[Dict[str, Union[List, pd.Series]]] = None,
    stops: Optional[Dict[str, Union[float, List[float]]]] = None,
    tz: str = "America/Toronto"
) -> "go.Figure":
    """Create TradingView-style candlestick chart with trade markers and overlays.
    
    Args:
        df: DataFrame with OHLCV data and datetime index
        trades: List of trade dictionaries with time, side, qty, price, reason
        overlays: Dictionary with overlay arrays (anchor, upper_band, lower_band, etc.)
        stops: Dictionary with stop levels (stop_loss, trail_stop, etc.)
        tz: Timezone for x-axis labels
        
    Returns:
        Plotly Figure object
        
    Raises:
        ImportError: If plotly is not available
        ValueError: If required columns are missing from dataframe
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for charting. Install with: pip install plotly"
        )
    
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert timezone for x-axis labels if needed
    x_values = df.index
    if hasattr(df.index, 'tz_convert'):
        target_tz = pytz.timezone(tz)
        if df.index.tz is None:
            # Assume UTC if no timezone
            x_values = df.index.tz_localize('UTC').tz_convert(target_tz)
        else:
            x_values = df.index.tz_convert(target_tz)
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
    ))
    
    # Add overlays if provided
    if overlays:
        _add_overlays(fig, x_values, overlays)
    
    # Add trade markers if provided
    if trades:
        _add_trade_markers(fig, trades, tz)
    
    # Add stop lines if provided
    if stops:
        _add_stop_lines(fig, x_values, stops)
    
    # Configure layout
    fig.update_layout(
        title=f"Trading Chart ({tz})",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Configure x-axis for better time display
    fig.update_xaxes(
        type="date",
        tickformat="%H:%M\n%m/%d",
        tickmode="auto",
        nticks=20
    )
    
    return fig


def _add_overlays(fig: "go.Figure", x_values: pd.Index, overlays: Dict[str, Any]) -> None:
    """Add overlay lines to the chart."""
    
    # Define overlay styles
    overlay_styles = {
        'anchor': dict(color='#ffa726', width=2, name="Anchor"),
        'upper_band': dict(color='#42a5f5', width=1, dash="dot", name="Upper Band"),
        'lower_band': dict(color='#42a5f5', width=1, dash="dot", name="Lower Band"),
        'sma': dict(color='#ab47bc', width=1, name="SMA"),
        'ema': dict(color='#66bb6a', width=1, name="EMA"),
        'support': dict(color='#26a69a', width=1, dash="dash", name="Support"),
        'resistance': dict(color='#ef5350', width=1, dash="dash", name="Resistance")
    }
    
    for overlay_name, values in overlays.items():
        if values is None:
            continue
            
        # Convert to list if pandas Series
        if hasattr(values, 'values'):
            values = values.values
        
        # Get style for this overlay
        style = overlay_styles.get(overlay_name, dict(color='#fff', width=1, name=overlay_name))
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=values,
            mode="lines",
            line=dict(
                width=style.get('width', 1),
                color=style.get('color', '#fff'),
                dash=style.get('dash', 'solid')
            ),
            name=style.get('name', overlay_name),
            hovertemplate=f"{style.get('name', overlay_name)}: %{{y:.2f}}<extra></extra>"
        ))


def _add_trade_markers(fig: "go.Figure", trades: List[Dict[str, Any]], tz: str) -> None:
    """Add trade markers to the chart."""
    
    target_tz = pytz.timezone(tz)
    
    buy_times, buy_prices, buy_texts = [], [], []
    sell_times, sell_prices, sell_texts = [], [], []
    
    for trade in trades:
        # Parse trade time
        trade_time = trade.get('time')
        if isinstance(trade_time, str):
            try:
                # Parse ISO format with timezone
                if 'T' in trade_time:
                    dt = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=pytz.UTC)
                    trade_time = dt.astimezone(target_tz)
                else:
                    # Assume date only
                    dt = datetime.strptime(trade_time, '%Y-%m-%d')
                    trade_time = target_tz.localize(dt)
            except:
                continue
        elif hasattr(trade_time, 'astimezone'):
            trade_time = trade_time.astimezone(target_tz)
        else:
            continue
        
        side = trade.get('side', 'unknown').lower()
        qty = trade.get('qty', 0)
        price = trade.get('price', 0)
        reason = trade.get('reason', 'Trade')
        
        # Create hover text
        hover_text = f"{side.title()} {qty} @ {price:.2f}"
        if reason:
            hover_text += f" — reason: {reason}"
        
        if side in ['buy', 'long']:
            buy_times.append(trade_time)
            buy_prices.append(price)
            buy_texts.append(hover_text)
        elif side in ['sell', 'short']:
            sell_times.append(trade_time)
            sell_prices.append(price)
            sell_texts.append(hover_text)
    
    # Add buy markers (green up triangles)
    if buy_times:
        fig.add_trace(go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#26a69a',
                line=dict(width=1, color='white')
            ),
            name='Buys',
            text=buy_texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
    
    # Add sell markers (red down triangles)
    if sell_times:
        fig.add_trace(go.Scatter(
            x=sell_times,
            y=sell_prices,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='#ef5350',
                line=dict(width=1, color='white')
            ),
            name='Sells',
            text=sell_texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))


def _add_stop_lines(fig: "go.Figure", x_values: pd.Index, stops: Dict[str, Any]) -> None:
    """Add stop loss and trailing stop lines to the chart."""
    
    stop_styles = {
        'stop_loss': dict(color='#f44336', dash='dash', name='Stop Loss'),
        'trail_stop': dict(color='#ff9800', dash='dash', name='Trailing Stop'),
        'take_profit': dict(color='#4caf50', dash='dash', name='Take Profit')
    }
    
    for stop_name, level in stops.items():
        if level is None:
            continue
        
        style = stop_styles.get(stop_name, dict(color='#999', dash='dash', name=stop_name))
        
        # Handle single value or array
        if isinstance(level, (int, float)):
            # Single horizontal line
            y_values = [level] * len(x_values)
        else:
            # Array of values
            y_values = level
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(
                width=1,
                color=style['color'],
                dash=style['dash']
            ),
            name=style['name'],
            hovertemplate=f"{style['name']}: %{{y:.2f}}<extra></extra>"
        ))


# Example usage for MNQ 808 strategy overlays
def create_mnq_808_overlays(df: pd.DataFrame, atr_period: int = 14, 
                           sma_period: int = 20) -> Dict[str, pd.Series]:
    """Create overlay data for MNQ 808 strategy visualization.
    
    Args:
        df: DataFrame with OHLCV data
        atr_period: Period for ATR calculation
        sma_period: Period for SMA calculation
        
    Returns:
        Dictionary with overlay series for anchor, upper_band, lower_band
    """
    # Calculate ATR (simplified version)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean()
    
    # Calculate SMA of True Range
    sma_tr = true_range.rolling(window=sma_period).mean()
    
    # Create anchor line (using close price as example)
    anchor = df['close'].rolling(window=sma_period).mean()
    
    # Create bands
    upper_band = anchor + (2 * atr)
    lower_band = anchor - (2 * atr)
    
    return {
        'anchor': anchor,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'atr': atr,
        'sma_tr': sma_tr
    }