"""QuantZoo Strategy Backtesting Dashboard.

Streamlined interface for running backtests on any strategy and symbol with custom date ranges.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="QuantZoo Backtester",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8001"

# Global state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'symbols' not in st.session_state:
    st.session_state.symbols = []


def api_request(endpoint: str, method: str = "GET", data: Dict = None, timeout: int = 30) -> Optional[Dict]:
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def load_strategies():
    """Load available strategies from API."""
    result = api_request("/strategies/list")
    if result and "strategies" in result:
        st.session_state.strategies = result["strategies"]
    return st.session_state.strategies


def load_symbols():
    """Load available symbols from API."""
    result = api_request("/symbols/list")
    if result and "symbols" in result:
        st.session_state.symbols = result["symbols"]
    return st.session_state.symbols


def run_backtest(strategy: str, symbol: str, start_date: str, end_date: str, initial_balance: float, commission: float, slippage: float):
    """Run backtest via API."""
    
    backtest_data = {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "initial_balance": initial_balance,
        "commission": commission,
        "slippage": slippage
    }
    
    with st.spinner("Running backtest... This may take a moment."):
        result = api_request("/backtest/run", method="POST", data=backtest_data)
        
    if result and result.get("success"):
        st.session_state.backtest_results = result["results"]
        return True
    else:
        error_msg = result.get("error", "Unknown error") if result else "API request failed"
        st.error(f"Backtest failed: {error_msg}")
        return False


def create_pnl_chart(results: Dict) -> go.Figure:
    """Create P&L equity curve chart."""
    positions = results.get("positions", [])
    
    if not positions:
        return go.Figure()
    
    # Create equity curve
    timestamps = [pos["timestamp"] for pos in positions]
    total_pnl = [results["initial_balance"] + pos["total_pnl"] for pos in positions]
    
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=total_pnl,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00D4AA', width=2),
        hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    # Starting balance line
    fig.add_hline(
        y=results["initial_balance"],
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Balance"
    )
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def create_price_chart(results: Dict) -> go.Figure:
    """Create price chart with trade markers."""
    bars = results.get("bars", [])
    trades = results.get("trades", [])
    
    if not bars:
        return go.Figure()
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create candlestick chart
    fig = go.Figure(data=go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=results["symbol"],
        hovertext=df['volume']
    ))
    
    # Add trade markers
    for trade in trades:
        if trade["status"] == "closed":
            # Entry marker
            fig.add_trace(go.Scatter(
                x=[trade["entry_time"]],
                y=[trade["entry_price"]],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if trade["side"] == "long" else 'triangle-down',
                    size=12,
                    color='green' if trade["side"] == "long" else 'red'
                ),
                name=f'{trade["side"].title()} Entry',
                showlegend=False,
                hovertemplate=f'Entry: {trade["side"]}<br>Price: ${trade["entry_price"]}<br>Time: %{{x}}<extra></extra>'
            ))
            
            # Exit marker
            fig.add_trace(go.Scatter(
                x=[trade["exit_time"]],
                y=[trade["exit_price"]],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='blue'
                ),
                name='Exit',
                showlegend=False,
                hovertemplate=f'Exit<br>Price: ${trade["exit_price"]}<br>P&L: ${trade["pnl"]}<br>Time: %{{x}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"{results['symbol']} Price Chart with Trades",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=False
    )
    
    return fig


def create_trades_table(results: Dict) -> pd.DataFrame:
    """Create trades table."""
    trades = results.get("trades", [])
    
    if not trades:
        return pd.DataFrame()
    
    df_trades = pd.DataFrame(trades)
    
    # Format columns
    if 'entry_time' in df_trades.columns:
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    if 'exit_time' in df_trades.columns:
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Reorder columns
    column_order = ['entry_time', 'side', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'quantity']
    available_columns = [col for col in column_order if col in df_trades.columns]
    df_trades = df_trades[available_columns]
    
    return df_trades


def create_performance_metrics(results: Dict) -> Dict:
    """Create performance metrics summary."""
    
    metrics = {
        "Total Return": f"{results.get('total_return_pct', 0):.2f}%",
        "Total P&L": f"${results.get('total_pnl', 0):,.2f}",
        "Final Balance": f"${results.get('final_balance', 0):,.2f}",
        "Total Trades": f"{results.get('total_trades', 0)}",
        "Win Rate": f"{results.get('win_rate_pct', 0):.1f}%",
        "Avg Win": f"${results.get('avg_win', 0):.2f}",
        "Avg Loss": f"${results.get('avg_loss', 0):.2f}",
        "Max Drawdown": f"{results.get('max_drawdown_pct', 0):.2f}%",
        "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}"
    }
    
    return metrics


# Main App
st.title("🎯 QuantZoo Strategy Backtester")
st.markdown("*Run backtests on any strategy and symbol with custom parameters*")

# Check API connection
health = api_request("/healthz")
if not health:
    st.error("❌ Cannot connect to QuantZoo API. Make sure the server is running on port 8001.")
    st.code("cd /Users/ronniel/quantzoo && python -m uvicorn quantzoo.rt.api:app --port 8001", language="bash")
    st.stop()

st.success("✅ Connected to QuantZoo API")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Backtest Configuration")
    
    # Load data
    strategies = load_strategies()
    symbols = load_symbols()
    
    if not strategies or not symbols:
        st.error("Failed to load strategies or symbols from API")
        st.stop()
    
    # Strategy selection
    st.subheader("📈 Strategy")
    strategy_options = {s["name"]: s["display_name"] for s in strategies}
    selected_strategy_name = st.selectbox(
        "Choose Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        key="strategy_selector"
    )
    
    # Show strategy info
    selected_strategy = next(s for s in strategies if s["name"] == selected_strategy_name)
    with st.expander("Strategy Details"):
        st.write(f"**Description:** {selected_strategy['description']}")
        st.write(f"**Risk Level:** {selected_strategy['risk_level']}")
        st.write(f"**Supported Symbols:** {', '.join(selected_strategy['symbols'])}")
        st.write(f"**Timeframes:** {', '.join(selected_strategy['timeframes'])}")
    
    # Symbol selection  
    st.subheader("🎯 Symbol")
    symbol_options = {s["symbol"]: f"{s['symbol']} - {s['name']}" for s in symbols}
    selected_symbol = st.selectbox(
        "Choose Symbol",
        options=list(symbol_options.keys()),
        format_func=lambda x: symbol_options[x],
        key="symbol_selector"
    )
    
    # Show symbol info
    selected_symbol_info = next(s for s in symbols if s["symbol"] == selected_symbol)
    with st.expander("Symbol Details"):
        st.write(f"**Exchange:** {selected_symbol_info['exchange']}")
        st.write(f"**Tick Size:** ${selected_symbol_info['tick_size']}")
        st.write(f"**Tick Value:** ${selected_symbol_info['tick_value']}")
        st.write(f"**Margin:** ${selected_symbol_info['margin']:,}")
        st.write(f"**Session:** {selected_symbol_info['session_hours']}")
    
    # Date range selection
    st.subheader("📅 Date Range")
    
    # Preset ranges
    preset_ranges = {
        "Last Week": (date.today() - timedelta(days=7), date.today()),
        "Last Month": (date.today() - timedelta(days=30), date.today()),
        "Last 3 Months": (date.today() - timedelta(days=90), date.today()),
        "YTD 2025": (date(2025, 1, 1), date(2025, 12, 31)),
        "Q1 2025": (date(2025, 1, 1), date(2025, 3, 31)),
        "Custom": None
    }
    
    range_selection = st.selectbox("Quick Range", options=list(preset_ranges.keys()))
    
    if range_selection == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2025, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=date(2025, 1, 31))
    else:
        start_date, end_date = preset_ranges[range_selection]
        st.info(f"Range: {start_date} to {end_date}")
    
    # Balance and risk settings
    st.subheader("💰 Balance & Risk")
    initial_balance = st.number_input(
        "Starting Balance ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    col1, col2 = st.columns(2)
    with col1:
        commission = st.number_input(
            "Commission ($)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.5
        )
    with col2:
        slippage = st.number_input(
            "Slippage ($)",
            min_value=0.0,
            max_value=10.0,
            value=0.25,
            step=0.25
        )
    
    # Run backtest button
    st.markdown("---")
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        success = run_backtest(
            strategy=selected_strategy_name,
            symbol=selected_symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            initial_balance=initial_balance,
            commission=commission,
            slippage=slippage
        )
        
        if success:
            st.success("✅ Backtest completed!")
            st.rerun()

# Main content area
if st.session_state.backtest_results is None:
    # Welcome screen
    st.markdown("""
    ## Welcome to QuantZoo Strategy Backtester! 🎯
    
    ### How to Use:
    1. **Choose a Strategy** - Select from our library of professional trading strategies
    2. **Pick a Symbol** - Choose from MNQ, ES, NQ, YM, RTY futures contracts  
    3. **Set Date Range** - Use presets or custom dates for your backtest period
    4. **Configure Balance** - Set starting capital and risk parameters
    5. **Run Backtest** - Click the button and get comprehensive results!
    
    ### Features:
    - 📊 **Interactive Charts** - Price charts with trade markers and equity curves
    - 📈 **Performance Metrics** - Win rate, Sharpe ratio, drawdown, and more
    - 📋 **Trade Analysis** - Detailed trade-by-trade breakdown
    - ⚡ **Fast Execution** - Get results in seconds
    
    **Ready to start? Configure your backtest in the sidebar and click "Run Backtest"!**
    """)
    
    # Show example strategies
    st.subheader("📈 Available Strategies")
    
    if strategies:
        cols = st.columns(len(strategies))
        for i, strategy in enumerate(strategies):
            with cols[i]:
                with st.container():
                    st.markdown(f"### {strategy['display_name']}")
                    st.write(strategy['description'])
                    st.write(f"**Risk:** {strategy['risk_level']}")
                    st.write(f"**Symbols:** {', '.join(strategy['symbols'][:3])}")

else:
    # Show results
    results = st.session_state.backtest_results
    
    st.header(f"📊 Backtest Results: {results['strategy'].upper()} on {results['symbol']}")
    
    # Performance summary cards
    metrics = create_performance_metrics(results)
    
    cols = st.columns(5)
    metric_items = list(metrics.items())
    
    for i, (label, value) in enumerate(metric_items[:5]):
        with cols[i]:
            st.metric(label, value)
    
    if len(metric_items) > 5:
        cols2 = st.columns(4)
        for i, (label, value) in enumerate(metric_items[5:]):
            with cols2[i % 4]:
                st.metric(label, value)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_pnl_chart(results), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_price_chart(results), use_container_width=True)
    
    # Trades table
    st.subheader("📋 Trade History")
    trades_df = create_trades_table(results)
    
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        
        # Download trades
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades CSV",
            data=csv,
            file_name=f"trades_{results['strategy']}_{results['symbol']}_{results['start_date']}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades generated in this backtest.")
    
    # Run another backtest
    if st.button("🔄 Run Another Backtest"):
        st.session_state.backtest_results = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Powered by QuantZoo Framework | Professional Trading Strategy Backtesting*")