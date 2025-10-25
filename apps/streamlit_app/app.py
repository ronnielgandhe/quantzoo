"""QuantZoo Streamlit Demo Application."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import os
from pathlib import Path

# Add quantzoo to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from quantzoo.strategies.mnq_808 import MNQ808, MNQ808Params
from quantzoo.strategies.regime_hybrid import RegimeHybrid, RegimeHybridParams
from quantzoo.backtest.engine import BacktestEngine, BacktestConfig
from quantzoo.data.loaders import load_csv_ohlcv, load_news_csv, join_news_prices
from quantzoo.metrics.core import calculate_metrics


def main():
    st.set_page_config(
        page_title="QuantZoo Strategy Backtesting",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 QuantZoo Strategy Backtesting Demo")
    st.markdown("Interactive backtesting framework for systematic trading strategies")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Select Strategy",
        ["MNQ 808", "Regime Hybrid"]
    )
    
    # File upload section
    st.sidebar.subheader("Data Upload")
    
    price_file = st.sidebar.file_uploader(
        "Upload Price Data (CSV)",
        type=['csv'],
        help="CSV with columns: time, open, high, low, close, volume"
    )
    
    news_file = None
    if strategy_type == "Regime Hybrid":
        news_file = st.sidebar.file_uploader(
            "Upload News Data (CSV)",
            type=['csv'],
            help="CSV with columns: timestamp, headline"
        )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    if strategy_type == "MNQ 808":
        params = configure_mnq808_params()
    else:
        params = configure_regime_hybrid_params()
    
    # Backtest settings
    st.sidebar.subheader("Backtest Settings")
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000, min_value=1000)
    fees_bps = st.sidebar.number_input("Fees (bps)", value=1.0, min_value=0.0)
    slippage_bps = st.sidebar.number_input("Slippage (bps)", value=1.0, min_value=0.0)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=1)
    
    # Main content area
    if price_file is not None:
        if strategy_type == "Regime Hybrid" and news_file is None:
            st.warning("Regime Hybrid strategy requires both price and news data.")
            return
        
        # Load and validate data
        try:
            price_data = load_data_from_upload(price_file)
            st.success(f"✅ Loaded {len(price_data)} price bars")
            
            news_data = None
            if news_file is not None:
                news_data = load_news_from_upload(news_file)
                st.success(f"✅ Loaded {len(news_data)} news items")
            
            # Show data preview
            show_data_preview(price_data, news_data)
            
            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary"):
                run_backtest(
                    strategy_type, params, price_data, news_data,
                    initial_capital, fees_bps, slippage_bps, seed
                )
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    else:
        # Show demo instructions
        show_demo_instructions()


def configure_mnq808_params():
    """Configure MNQ 808 strategy parameters."""
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        atr_mult = st.number_input("ATR Multiplier", value=1.5, min_value=0.1)
        lookback = st.number_input("Lookback Period", value=10, min_value=1)
        contracts = st.number_input("Contracts", value=1, min_value=1)
    
    with col2:
        use_mfi = st.checkbox("Use MFI", value=True)
        risk_ticks = st.number_input("Risk Ticks", value=150, min_value=10)
        trail_mult = st.number_input("Trail Multiplier", value=1.0, min_value=0.1)
    
    return MNQ808Params(
        atr_mult=atr_mult,
        lookback=lookback,
        use_mfi=use_mfi,
        trail_mult_legacy=trail_mult,
        contracts=contracts,
        risk_ticks_legacy=risk_ticks,
        session_start="08:00",
        session_end="16:30",
        tick_size=0.25,
        tick_value=0.5,
        treat_atr_as_ticks=True
    )


def configure_regime_hybrid_params():
    """Configure Regime Hybrid strategy parameters."""
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        text_mode = st.selectbox("Text Mode", ["tfidf", "hf"], index=0)
        lookback = st.number_input("Lookback Period", value=20, min_value=1)
        contracts = st.number_input("Contracts", value=1, min_value=1)
    
    with col2:
        news_window = st.selectbox("News Window", ["15min", "30min", "1h"], index=1)
        risk_ticks = st.number_input("Risk Ticks", value=100, min_value=10)
        clf = st.selectbox("Classifier", ["logreg"], index=0)
    
    return RegimeHybridParams(
        text_mode=text_mode,
        lookback=lookback,
        news_window=news_window,
        price_features=["returns", "zscore", "atr"],
        clf=clf,
        contracts=contracts,
        session_start="08:00",
        session_end="16:30",
        tick_size=0.25,
        tick_value=0.5,
        risk_ticks=risk_ticks
    )


def load_data_from_upload(uploaded_file):
    """Load price data from uploaded file."""
    # Save uploaded file temporarily
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load using quantzoo loader
    data = load_csv_ohlcv(temp_path, tz=None, timeframe="15m")
    
    # Clean up
    os.remove(temp_path)
    
    return data


def load_news_from_upload(uploaded_file):
    """Load news data from uploaded file."""
    # Save uploaded file temporarily
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load using quantzoo loader
    data = load_news_csv(temp_path)
    
    # Clean up
    os.remove(temp_path)
    
    return data


def show_data_preview(price_data, news_data=None):
    """Show preview of loaded data."""
    st.subheader("📊 Data Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Data Sample:**")
        st.dataframe(price_data.head())
        
        st.write("**Price Statistics:**")
        st.write(f"- Bars: {len(price_data)}")
        st.write(f"- Date Range: {price_data.index[0]} to {price_data.index[-1]}")
        st.write(f"- Price Range: {price_data['close'].min():.2f} - {price_data['close'].max():.2f}")
    
    with col2:
        if news_data is not None:
            st.write("**News Data Sample:**")
            st.dataframe(news_data.head())
            
            st.write("**News Statistics:**")
            st.write(f"- Items: {len(news_data)}")
            st.write(f"- Date Range: {news_data.index[0]} to {news_data.index[-1]}")
        else:
            st.write("**No news data loaded**")


def run_backtest(strategy_type, params, price_data, news_data, initial_capital, fees_bps, slippage_bps, seed):
    """Run backtest and display results."""
    
    with st.spinner("Running backtest..."):
        try:
            # Prepare data
            if strategy_type == "Regime Hybrid" and news_data is not None:
                data = join_news_prices(news_data, price_data, params.news_window)
            else:
                data = price_data
            
            # Initialize strategy
            if strategy_type == "MNQ 808":
                strategy = MNQ808(params)
            else:
                strategy = RegimeHybrid(params)
            
            # Configure backtest
            config = BacktestConfig(
                initial_capital=initial_capital,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
                seed=seed
            )
            
            # Run backtest
            engine = BacktestEngine(config)
            results = engine.run(data, strategy)
            
            # Calculate metrics
            trades = results.get('trades', [])
            equity_curve = results.get('equity_curve', [])
            metrics = calculate_metrics(trades, equity_curve)
            
            # Display results
            display_results(metrics, trades, equity_curve, data)
            
            # Special handling for regime hybrid
            if strategy_type == "Regime Hybrid" and hasattr(strategy, 'get_regime_predictions'):
                display_regime_predictions(strategy.get_regime_predictions())
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")


def display_results(metrics, trades, equity_curve, data):
    """Display backtest results."""
    st.subheader("📈 Backtest Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
    
    with col2:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
    
    with col3:
        st.metric("Total Trades", len(trades))
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    
    with col4:
        avg_trade = sum(t.pnl for t in trades) / len(trades) if trades else 0
        st.metric("Avg Trade", f"${avg_trade:.2f}")
        st.metric("Exposure", f"{metrics.get('exposure', 0):.1%}")
    
    # Equity curve chart
    if equity_curve:
        st.subheader("💰 Equity Curve")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve, linewidth=2, color='#1f77b4')
        ax.set_title("Portfolio Equity Over Time")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Trade table
    if trades:
        st.subheader("📋 Trade Details")
        
        trade_data = []
        for trade in trades:
            trade_data.append({
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'Side': trade.side,
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Price': f"${trade.exit_price:.2f}",
                'P&L': f"${trade.pnl:.2f}",
                'Exit Reason': trade.exit_reason
            })
        
        trade_df = pd.DataFrame(trade_data)
        st.dataframe(trade_df, use_container_width=True)


def display_regime_predictions(predictions):
    """Display regime predictions for hybrid strategy."""
    if not predictions:
        return
    
    st.subheader("🔮 Regime Predictions")
    
    pred_df = pd.DataFrame(predictions)
    
    # Summary stats
    risk_on_pct = (pred_df['regime'] > 0.5).mean() * 100
    st.write(f"**Risk-On Regime:** {risk_on_pct:.1f}% of time")
    
    # Chart
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pred_df['timestamp'], pred_df['regime'], marker='o', markersize=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Neutral')
    ax.set_title("Regime Classification Over Time")
    ax.set_ylabel("Regime Score (0=Risk-Off, 1=Risk-On)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)


def show_demo_instructions():
    """Show demo instructions when no data is uploaded."""
    st.markdown("""
    ## 🚀 Welcome to QuantZoo Demo
    
    This interactive application lets you backtest systematic trading strategies on your own data.
    
    ### 📋 Getting Started
    
    1. **Upload Data**: Use the sidebar to upload price data (required) and news data (for Regime Hybrid)
    2. **Choose Strategy**: Select between MNQ 808 or Regime Hybrid
    3. **Configure Parameters**: Tune strategy settings in the sidebar
    4. **Run Backtest**: Click the run button to execute the backtest
    
    ### 📊 Data Format Requirements
    
    **Price Data CSV:**
    ```
    time,open,high,low,close,volume
    2023-01-01 08:00:00,4200.50,4205.25,4198.75,4203.00,1500
    2023-01-01 08:15:00,4203.00,4207.50,4200.25,4205.75,1850
    ```
    
    **News Data CSV (for Regime Hybrid):**
    ```
    timestamp,headline
    2023-01-01 08:05:00,"Fed signals dovish stance on rates"
    2023-01-01 08:12:00,"Tech earnings beat expectations"
    ```
    
    ### 🔧 Strategies Available
    
    - **MNQ 808**: Systematic futures strategy with momentum and anchor recursion
    - **Regime Hybrid**: News+price sentiment model for regime detection
    
    ### ⚠️ Important Notes
    
    - Results are for educational purposes only
    - Past performance does not guarantee future results
    - All backtests include realistic fees and slippage
    """)


if __name__ == "__main__":
    main()