"""QuantZoo Real-time Trading Dashboard.

TradingView-like interface for monitoring strategies, positions, and metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import time
import asyncio
from datetime import datetime, timedelta, date
import os
import yaml
import glob
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="QuantZoo Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8001"

# Global state
if 'api_state' not in st.session_state:
    st.session_state.api_state = {}
if 'position_data' not in st.session_state:
    st.session_state.position_data = {}
if 'trades_data' not in st.session_state:
    st.session_state.trades_data = []
if 'signal_data' not in st.session_state:
    st.session_state.signal_data = {}
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True


def api_request(endpoint: str, method: str = "GET", params: Dict = None, timeout: int = 5) -> Optional[Dict]:
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=params, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, params=params, timeout=timeout)
        else:
            return None
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        # st.error(f"API Error: {e}")  # Comment out to avoid spam
        return None


def check_api_status() -> bool:
    """Check if FastAPI service is running."""
    result = api_request("/healthz")
    return result is not None and result.get("ok", False)


def get_api_state() -> Dict[str, Any]:
    """Get current API state."""
    return api_request("/state") or {}


def get_current_position() -> Dict[str, Any]:
    """Get current position."""
    return api_request("/positions/current") or {}


def get_recent_trades(n: int = 200) -> Dict[str, Any]:
    """Get recent trades."""
    return api_request("/trades/recent", params={"n": n}) or {"trades": [], "count": 0}


def get_latest_signals() -> Dict[str, Any]:
    """Get latest signals."""
    return api_request("/signals/latest") or {"signals": {}}


def get_latest_metrics() -> Dict[str, Any]:
    """Get latest metrics."""
    return api_request("/metrics/latest") or {}


def start_replay(symbols: List[str], csv_path: str, speed: float, config_path: str = None, 
                timeframe: str = "15m", start_date: str = "2025-01-01", end_date: str = None) -> Dict:
    """Start replay via API."""
    params = {
        "symbols": symbols,
        "csv_path": csv_path,
        "speed": speed,
        "timeframe": timeframe,
        "start_date": start_date
    }
    if config_path:
        params["config_path"] = config_path
    if end_date:
        params["end_date"] = end_date
    
    return api_request("/replay/start", "POST", params) or {"status": "error"}


def stop_replay() -> Dict:
    """Stop replay via API."""
    return api_request("/replay/stop", "POST") or {"status": "error"}


def load_config_files() -> List[str]:
    """Load available configuration files."""
    config_files = []
    for pattern in ["configs/*.yaml", "configs/*.yml"]:
        config_files.extend(glob.glob(pattern))
    return sorted(config_files)


def parse_config(config_path: str) -> Dict:
    """Parse YAML config file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}


def create_candlestick_chart(chart_data: List[Dict], trades_data: List[Dict] = None,
                           signals: Dict = None, overlays: Dict = None) -> go.Figure:
    """Create TradingView-style candlestick chart."""
    if not chart_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No chart data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    df = pd.DataFrame(chart_data)
    if df.empty:
        return go.Figure()
    
    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        x_values = df['timestamp']
    else:
        x_values = df.index
    
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df.get('open', []),
        high=df.get('high', []),
        low=df.get('low', []),
        close=df.get('close', []),
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Add overlays if provided
    if overlays:
        # Anchor line
        if 'anchor' in overlays:
            fig.add_trace(go.Scatter(
                x=x_values, y=overlays['anchor'], name="Anchor", mode="lines",
                line=dict(width=2, color='#ffa726')
            ))
        
        # ATR bands
        if 'upper_band' in overlays:
            fig.add_trace(go.Scatter(
                x=x_values, y=overlays['upper_band'], name="Upper Band", mode="lines",
                line=dict(width=1, dash="dot", color='#42a5f5')
            ))
        
        if 'lower_band' in overlays:
            fig.add_trace(go.Scatter(
                x=x_values, y=overlays['lower_band'], name="Lower Band", mode="lines",
                line=dict(width=1, dash="dot", color='#42a5f5')
            ))
    
    # Add trade markers
    if trades_data:
        buy_times, buy_prices, buy_texts = [], [], []
        sell_times, sell_prices, sell_texts = [], [], []
        
        for trade in trades_data:
            if not trade.get('time') or not trade.get('price'):
                continue
            
            trade_time = pd.to_datetime(trade['time'])
            side = trade.get('side', '').lower()
            qty = trade.get('qty', 0)
            price = trade.get('price', 0)
            reason = trade.get('reason', 'Trade')
            
            hover_text = f"{side.title()} {qty} @ {price:.2f} — reason: {reason}"
            
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
                x=buy_times, y=buy_prices, mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#26a69a',
                           line=dict(width=1, color='white')),
                name='Buys', text=buy_texts, hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add sell markers (red down triangles)
        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#ef5350',
                           line=dict(width=1, color='white')),
                name='Sells', text=sell_texts, hovertemplate='%{text}<extra></extra>'
            ))
    
    # Configure layout
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode='x unified',
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig


def create_equity_chart(metrics: Dict) -> go.Figure:
    """Create equity curve chart."""
    fig = go.Figure()
    
    # Mock equity data - in real implementation this would come from metrics
    if metrics and 'equity_curve' in metrics:
        equity_data = metrics['equity_curve']
        fig.add_trace(go.Scatter(
            y=equity_data,
            mode='lines',
            name='Equity',
            line=dict(color='#2E86AB', width=2)
        ))
    else:
        # Mock data for demo
        x = list(range(100))
        y = [100000 + i * 50 + np.random.normal(0, 100) for i in x]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', name='Equity',
            line=dict(color='#2E86AB', width=2)
        ))
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity ($)",
        height=300,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_drawdown_chart(metrics: Dict) -> go.Figure:
    """Create drawdown chart."""
    fig = go.Figure()
    
    # Mock drawdown data
    x = list(range(100))
    y = [max(0, -i * 0.1 + np.random.normal(0, 2)) for i in x]
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', fill='tonexty',
        name='Drawdown', line=dict(color='#ef5350', width=1),
        fillcolor='rgba(239, 83, 80, 0.3)'
    ))
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        height=300,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def format_currency(value: float) -> str:
    """Format currency values."""
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """Format percentage values."""
    return f"{value:.2f}%"


# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("QuantZoo Real-time Dashboard")
with col2:
    # API state info
    api_state = get_api_state()
    if api_state:
        tz = api_state.get('tz', 'UTC')
        heartbeat = api_state.get('last_heartbeat', '')
        if heartbeat:
            try:
                hb_time = pd.to_datetime(heartbeat).strftime('%H:%M:%S')
                st.caption(f"Last heartbeat: {hb_time} {tz}")
            except:
                st.caption(f"Heartbeat: {tz}")
        else:
            st.caption(f"Timezone: {tz}")

# Check API status
api_status = check_api_status()

# Sidebar Controls
st.sidebar.header("⚙️ Controls")

# Status pill
if api_status:
    st.sidebar.success("✅ API Online")
    api_state = get_api_state()
    if api_state.get('is_running'):
        st.sidebar.info("🔄 Replay Running")
    else:
        st.sidebar.warning("⏸️ Replay Stopped")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.code("uvicorn quantzoo.rt.api:app --reload", language="bash")

# Configuration
st.sidebar.subheader("Configuration")
config_files = load_config_files()
selected_config = st.sidebar.selectbox(
    "Config File",
    options=[""] + config_files,
    index=1 if config_files else 0,
    key="config_selector"
)

# Data Provider Selection
st.sidebar.subheader("📡 Data Provider")
provider_type = st.sidebar.selectbox(
    "Provider",
    ["replay", "polygon", "alpha_vantage"],
    help="Select real-time data provider",
    key="provider_selector"
)

if provider_type != "replay":
    st.sidebar.info(f"🔴 Live data requires {provider_type.title()} API key")
    if provider_type == "polygon":
        st.sidebar.markdown("Get key: https://polygon.io/")
    elif provider_type == "alpha_vantage":
        st.sidebar.markdown("Get key: https://www.alphavantage.co/")
else:
    st.sidebar.success("🟢 Using simulated replay data")

# Parse config defaults
config_data = {}
default_csv = "tests/data/mnq_15m_2025.csv"
default_symbols = ["MNQ"]
default_timeframe = "15m"

if selected_config:
    config_data = parse_config(selected_config)
    data_config = config_data.get('data', {})
    default_csv = data_config.get('path', default_csv)
    default_timeframe = data_config.get('timeframe', default_timeframe)

# Symbol selection
symbols_input = st.sidebar.multiselect(
    "Symbols",
    options=["MNQ", "ES", "NQ", "YM", "RTY"],
    default=default_symbols
)
if not symbols_input:
    symbols_input = default_symbols

# Timeframe
timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    index=2  # Default to 15m
)

# Date range
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2025, 1, 1),
    min_value=date(2020, 1, 1),
    max_value=date.today()
)
end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=start_date,
    max_value=date.today()
)

# CSV path
csv_path = st.sidebar.text_input("CSV Path", value=default_csv)

# Replay speed
speed_factor = st.sidebar.slider(
    "Replay Speed",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)

# Symbol/state validation
state_symbol = api_state.get('symbol')
ui_symbol = symbols_input[0] if symbols_input else 'MNQ'
symbol_mismatch = bool(state_symbol and state_symbol != ui_symbol)

if symbol_mismatch:
    st.sidebar.warning(f"⚠️ Symbol mismatch: API({state_symbol}) ≠ UI({ui_symbol})")

# Control buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    start_disabled = not api_status or symbol_mismatch
    if st.button("▶️ Start", disabled=start_disabled):
        if api_status and not symbol_mismatch:
            result = start_replay(
                symbols=symbols_input,
                csv_path=csv_path,
                speed=speed_factor,
                config_path=selected_config,
                timeframe=timeframe,
                start_date=str(start_date),
                end_date=str(end_date) if end_date != date.today() else None
            )
            if result.get('status') == 'started':
                st.sidebar.success("✅ Started!")
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error(f"❌ Error: {result.get('message', 'Unknown')}")

with col2:
    if st.button("⏹️ Stop"):
        if api_status:
            result = stop_replay()
            if result.get('status') == 'stopped':
                st.sidebar.success("✅ Stopped!")
                time.sleep(1)
                st.rerun()

if not api_status:
    st.error("❌ FastAPI service is not running. Please start it to view live data.")
    st.code("uvicorn quantzoo.rt.api:app --reload", language="bash")
else:
    # Fetch live data
    current_position = get_current_position()
    recent_trades_resp = get_recent_trades()
    trades_data = recent_trades_resp.get('trades', [])
    signals = get_latest_signals().get('signals', {})
    metrics = get_latest_metrics()
    
    # Position card
    if current_position and current_position.get('qty', 0) != 0:
        side = current_position.get('side', 'flat')
        qty = current_position.get('qty', 0)
        avg_price = current_position.get('avg_price', 0)
        unrealized_pnl = current_position.get('unrealized_pnl', 0)
        unrealized_pnl_pct = current_position.get('unrealized_pnl_pct', 0)
        
        # Color-coded position display
        side_color = "green" if side == "long" else "red" if side == "short" else "gray"
        pnl_color = "green" if unrealized_pnl >= 0 else "red"
        
        st.markdown(f"""
        <div style="background-color: #2d3748; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {side_color};">
            <h4 style="margin: 0; color: {side_color};">Position: {side.upper()} {qty}</h4>
            <p style="margin: 0.5rem 0;">Avg Price: {format_currency(avg_price)}</p>
            <p style="margin: 0.5rem 0; color: {pnl_color};">
                Unrealized P&L: {format_currency(unrealized_pnl)} ({format_percentage(unrealized_pnl_pct)})
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 No open position")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Charts", "💼 Positions", "📋 Trades"])
    
    with tab1:
        # Main chart with signal inspector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Mock chart data for now - in real implementation this would be fetched from API
            chart_fig = create_candlestick_chart([], trades_data)
            st.plotly_chart(chart_fig, use_container_width=True)
        
        with col2:
            st.subheader("Signal Inspector")
            if signals:
                # Display signals in a compact grid
                st.metric("ATR", f"{signals.get('atr', 0):.2f}" if signals.get('atr') else "N/A")
                st.metric("SMA(TR)", f"{signals.get('smaTR', 0):.2f}" if signals.get('smaTR') else "N/A")
                st.metric("Momentum", f"{signals.get('momentum_kind', 'N/A')}: {signals.get('momentum_value', 0):.2f}")
                st.metric("Anchor Current", f"{signals.get('anchor_current', 0):.2f}" if signals.get('anchor_current') else "N/A")
                st.metric("Anchor Lag2", f"{signals.get('anchor_lag2', 0):.2f}" if signals.get('anchor_lag2') else "N/A")
                st.metric("Crossover State", signals.get('crossover_state', 'N/A'))
                if signals.get('bar_time'):
                    st.caption(f"Bar Time: {signals['bar_time']}")
                if signals.get('bar_price'):
                    st.metric("Bar Price", f"{signals['bar_price']:.2f}")
            else:
                st.info("No signal data")
    
    with tab2:
        st.subheader("📈 Performance Charts")
        
        col1, col2 = st.columns(2)
        with col1:
            equity_fig = create_equity_chart(metrics)
            st.plotly_chart(equity_fig, use_container_width=True)
        
        with col2:
            drawdown_fig = create_drawdown_chart(metrics)
            st.plotly_chart(drawdown_fig, use_container_width=True)
        
        # Rolling Sharpe
        st.subheader("Rolling Sharpe Ratio")
        sharpe_fig = go.Figure()
        x = list(range(100))
        y = [np.random.normal(1.5, 0.5) for _ in x]
        sharpe_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Rolling Sharpe (30d)'))
        sharpe_fig.update_layout(
            height=300, template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(sharpe_fig, use_container_width=True)
    
    with tab3:
        st.subheader("💼 Positions")
        
        # Current position details
        if current_position and current_position.get('qty', 0) != 0:
            pos_df = pd.DataFrame([{
                "Side": current_position.get('side', '').title(),
                "Quantity": current_position.get('qty', 0),
                "Avg Price": format_currency(current_position.get('avg_price', 0)),
                "Unrealized P&L": format_currency(current_position.get('unrealized_pnl', 0)),
                "Unrealized P&L %": format_percentage(current_position.get('unrealized_pnl_pct', 0)),
                "Entry Time": current_position.get('entry_time', 'N/A')
            }])
            st.dataframe(pos_df, hide_index=True, use_container_width=True)
        else:
            st.info("No open positions")
        
        # Closed positions (collapsible)
        with st.expander("Closed Positions (MAE/MFE)"):
            if trades_data:
                closed_trades = [t for t in trades_data if t.get('exit_time')]
                if closed_trades:
                    closed_df = pd.DataFrame([{
                        "Entry Time": t.get('time', ''),
                        "Exit Time": t.get('exit_time', ''),
                        "Side": t.get('side', '').title(),
                        "Quantity": t.get('qty', 0),
                        "Entry Price": format_currency(t.get('price', 0)),
                        "Exit Price": format_currency(t.get('exit_price', 0)) if t.get('exit_price') else 'N/A',
                        "P&L": format_currency(t.get('pnl', 0)) if t.get('pnl') else 'N/A',
                        "Reason": t.get('reason', '')
                    } for t in closed_trades[-20:]])  # Last 20 trades
                    st.dataframe(closed_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No closed positions")
            else:
                st.info("No trade data")
    
    with tab4:
        st.subheader("📋 Trades")
        
        if trades_data:
            # Format trades for display
            trades_df = pd.DataFrame([{
                "Time": t.get('time', ''),
                "Side": t.get('side', '').title(),
                "Quantity": t.get('qty', 0),
                "Price": format_currency(t.get('price', 0)),
                "Fees (bps)": f"{t.get('fees_bps', 0):.1f}",
                "Slippage (bps)": f"{t.get('slippage_bps', 0):.1f}",
                "Reason": t.get('reason', ''),
                "P&L": format_currency(t.get('pnl', 0)) if t.get('pnl') else 'Open'
            } for t in trades_data])
            
            # Make sortable
            st.dataframe(
                trades_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Time": st.column_config.DatetimeColumn("Time"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "P&L": st.column_config.NumberColumn("P&L", format="$%.2f")
                }
            )
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades to display")

# Auto-refresh
if st.session_state.auto_refresh and api_status:
    time.sleep(2)
    st.rerun()