"""FastAPI service for streaming real-time data and metrics."""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn
import pytz

from .providers import create_provider, BaseProvider
from .replay import ReplayEngine


app = FastAPI(title="QuantZoo Real-time API", version="1.0.0")

# Global state
current_provider: Optional[BaseProvider] = None
replay_engine: Optional[ReplayEngine] = None
latest_metrics: Dict[str, Any] = {}

# Real-time state tracking
current_state: Dict[str, Any] = {
    "symbol": None,
    "timeframe": None,
    "timezone": "America/Toronto",
    "config_path": None,
    "replay_speed": 1.0,
    "last_heartbeat": None,
    "is_running": False
}

# Position and trade tracking
current_position: Dict[str, Any] = {
    "side": "flat",
    "qty": 0.0,
    "avg_price": 0.0,
    "unrealized_pnl": 0.0,
    "entry_time": None
}

recent_trades: List[Dict[str, Any]] = []
latest_signals: Dict[str, Any] = {}


@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    return {
        "ok": True,
        "version": "1.0.0",
        "service": "quantzoo-rt-api"
    }


@app.get("/state")
def get_state():
    """Get current real-time state."""
    tz = pytz.timezone(current_state["timezone"])
    now = datetime.now(tz)
    
    return {
        "symbol": current_state["symbol"],
        "timeframe": current_state["timeframe"],
        "tz": current_state["timezone"],
        "config_path": current_state["config_path"],
        "replay_speed": current_state["replay_speed"],
        "last_heartbeat": current_state["last_heartbeat"],
        "is_running": current_state["is_running"],
        "current_time": now.isoformat()
    }


@app.get("/positions/current")
def get_current_position():
    """Get current position information."""
    tz = pytz.timezone(current_state["timezone"])
    
    response = {
        "side": current_position["side"],
        "qty": current_position["qty"],
        "avg_price": current_position["avg_price"],
        "unrealized_pnl": current_position["unrealized_pnl"],
        "entry_time": current_position["entry_time"]
    }
    
    # Calculate unrealized PnL percentage if we have a position
    if current_position["qty"] != 0 and current_position["avg_price"] != 0:
        # We'd need current market price for accurate calculation
        # For now, return the stored value
        notional = abs(current_position["qty"]) * current_position["avg_price"]
        response["unrealized_pnl_pct"] = (current_position["unrealized_pnl"] / notional * 100) if notional > 0 else 0.0
    else:
        response["unrealized_pnl_pct"] = 0.0
    
    return response


@app.get("/trades/recent")
def get_recent_trades(n: int = Query(200, description="Number of recent trades to return")):
    """Get recent trade history."""
    tz = pytz.timezone(current_state["timezone"])
    
    # Return last N trades, ensuring timezone-aware timestamps
    trades_to_return = recent_trades[-n:] if len(recent_trades) > n else recent_trades
    
    for trade in trades_to_return:
        # Ensure timestamps are timezone-aware ISO-8601 strings
        if trade.get("time") and not isinstance(trade["time"], str):
            if hasattr(trade["time"], "astimezone"):
                trade["time"] = trade["time"].astimezone(tz).isoformat()
            else:
                # Convert to timezone-aware datetime if needed
                dt = datetime.fromisoformat(str(trade["time"]).replace("Z", "+00:00"))
                trade["time"] = dt.astimezone(tz).isoformat()
        
        if trade.get("exit_time") and not isinstance(trade["exit_time"], str):
            if hasattr(trade["exit_time"], "astimezone"):
                trade["exit_time"] = trade["exit_time"].astimezone(tz).isoformat()
            else:
                dt = datetime.fromisoformat(str(trade["exit_time"]).replace("Z", "+00:00"))
                trade["exit_time"] = dt.astimezone(tz).isoformat()
    
    return {
        "trades": trades_to_return,
        "count": len(trades_to_return),
        "timezone": current_state["timezone"]
    }


@app.get("/signals/latest")
def get_latest_signals():
    """Get latest signal inputs used on the last bar."""
    tz = pytz.timezone(current_state["timezone"])
    
    response = {**latest_signals}
    
    # Ensure bar_time is timezone-aware ISO-8601 string
    if response.get("bar_time") and not isinstance(response["bar_time"], str):
        if hasattr(response["bar_time"], "astimezone"):
            response["bar_time"] = response["bar_time"].astimezone(tz).isoformat()
        else:
            dt = datetime.fromisoformat(str(response["bar_time"]).replace("Z", "+00:00"))
            response["bar_time"] = dt.astimezone(tz).isoformat()
    
    return {
        "signals": response,
        "timezone": current_state["timezone"]
    }


@app.get("/bars")
async def stream_bars(
    symbol: str = Query("MNQ", description="Symbol to stream"),
    provider: str = Query("replay", description="Provider type"),
    csv_path: str = Query("tests/data/mini_mnq_15m.csv", description="CSV path for replay"),
    speed: float = Query(1.0, description="Speed factor for replay")
):
    """Stream real-time bars via Server-Sent Events."""
    
    async def event_generator():
        try:
            # Create provider
            data_provider = create_provider(
                provider_type=provider,
                csv_path=csv_path,
                speed_factor=speed
            )
            
            # Stream bars
            if hasattr(data_provider, 'iter_bars'):
                async for bar in data_provider.iter_bars([symbol]):
                    yield {
                        "event": "bar",
                        "data": json.dumps(bar)
                    }
            else:
                # Fallback for sync providers
                data_provider.subscribe([symbol])
                while True:
                    bar = data_provider.next_bar()
                    if bar is None:
                        break
                    yield {
                        "event": "bar", 
                        "data": json.dumps(bar)
                    }
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
        finally:
            if data_provider:
                data_provider.close()
    
    return EventSourceResponse(event_generator())


@app.get("/metrics/latest")
def get_latest_metrics():
    """Get latest computed portfolio metrics snapshot."""
    if not latest_metrics:
        return {
            "status": "no_data",
            "message": "No metrics computed yet",
            "metrics": {}
        }
    
    return {
        "status": "ok",
        "timestamp": latest_metrics.get("timestamp"),
        "metrics": latest_metrics
    }


@app.post("/metrics/update")
def update_metrics(metrics: Dict[str, Any]):
    """Update the latest metrics snapshot."""
    global latest_metrics
    import time
    
    latest_metrics = {
        **metrics,
        "timestamp": time.time(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return {"status": "updated", "count": len(metrics)}


@app.post("/replay/start")
async def start_replay(
    symbols: list = Query(["MNQ"]),
    csv_path: str = Query("tests/data/mnq_15m_2025.csv"),
    speed: float = Query(1.0),
    config_path: str = Query(None),
    timeframe: str = Query("15m"),
    start_date: str = Query("2025-01-01"),
    end_date: str = Query(None)
):
    """Start replay engine in background."""
    global replay_engine
    
    try:
        if replay_engine and replay_engine.is_running:
            return {"status": "error", "message": "Replay already running"}
        
        # Update state
        update_state(
            symbol=symbols[0] if symbols else "MNQ",
            timeframe=timeframe,
            config_path=config_path,
            replay_speed=speed,
            is_running=True
        )
        
        replay_engine = ReplayEngine(
            csv_path=csv_path, 
            speed_factor=speed,
            start_date=start_date,
            end_date=end_date,
            timezone=current_state["timezone"]
        )
        
        # Start in background task
        asyncio.create_task(replay_engine.start(symbols))
        
        return {
            "status": "started",
            "symbols": symbols,
            "csv_path": csv_path,
            "speed_factor": speed,
            "timeframe": timeframe,
            "config_path": config_path,
            "start_date": start_date,
            "end_date": end_date
        }
        
    except Exception as e:
        update_state(is_running=False)
        return {"status": "error", "message": str(e)}


@app.post("/replay/stop")
def stop_replay():
    """Stop replay engine."""
    global replay_engine
    
    if replay_engine:
        replay_engine.stop()
        update_state(is_running=False)
        return {"status": "stopped"}
    
    update_state(is_running=False)
    return {"status": "not_running"}


@app.post("/backtest/run")
async def run_backtest(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run a backtest with specified strategy, symbol, date range, and balance."""
    
    try:
        # Extract parameters
        strategy_name = request.get("strategy", "mnq_808")
        symbol = request.get("symbol", "MNQ")
        start_date = request.get("start_date", "2025-01-01")
        end_date = request.get("end_date", "2025-01-31")
        initial_balance = request.get("initial_balance", 100000)
        commission = request.get("commission", 2.0)
        slippage = request.get("slippage", 0.25)
        
        # Generate synthetic data for backtesting (in production, this would load real data)
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create date range
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Generate 15-minute bars between dates
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='15min')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        
        base_prices = {
            "MNQ": 4200,
            "ES": 4600,
            "NQ": 16800, 
            "YM": 36000,
            "RTY": 2100
        }
        
        base_price = base_prices.get(symbol, 4200)
        
        # Generate price series with some trend and volatility
        returns = np.random.normal(0.0001, 0.008, len(date_range))  # Small positive drift
        prices = [base_price]
        
        for r in returns[1:]:
            new_price = prices[-1] * (1 + r)
            prices.append(new_price)
        
        # Create OHLCV data
        bars = []
        for i, (timestamp, close) in enumerate(zip(date_range, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.002)))
            low = close * (1 - abs(np.random.normal(0, 0.002)))
            open_price = close + np.random.normal(0, close * 0.001)
            volume = np.random.randint(800, 2000)
            
            bars.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
                "symbol": symbol
            })
        
        # Simulate strategy performance
        trades = []
        positions = []
        current_position = 0
        cash = initial_balance
        total_pnl = 0
        
        # Simple strategy simulation - buy when price goes up, sell when down
        for i in range(10, len(bars) - 10):
            bar = bars[i]
            prev_bars = bars[i-10:i]
            
            # Simple momentum strategy
            recent_closes = [b["close"] for b in prev_bars]
            sma = sum(recent_closes) / len(recent_closes)
            current_price = bar["close"]
            
            # Entry logic
            if current_position == 0:
                if current_price > sma * 1.002:  # Buy signal
                    current_position = 1
                    entry_price = current_price
                    
                    trade = {
                        "entry_time": bar["timestamp"],
                        "entry_price": entry_price,
                        "side": "long",
                        "quantity": 1,
                        "status": "open"
                    }
                    
                elif current_price < sma * 0.998:  # Sell signal
                    current_position = -1
                    entry_price = current_price
                    
                    trade = {
                        "entry_time": bar["timestamp"],
                        "entry_price": entry_price,
                        "side": "short",
                        "quantity": 1,
                        "status": "open"
                    }
            
            # Exit logic
            elif current_position != 0 and i % 20 == 0:  # Exit every 20 bars
                exit_price = current_price
                pnl = (exit_price - entry_price) * current_position
                total_pnl += pnl
                
                trade.update({
                    "exit_time": bar["timestamp"],
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "status": "closed"
                })
                trades.append(trade)
                current_position = 0
            
            # Track position
            positions.append({
                "timestamp": bar["timestamp"],
                "position": current_position,
                "unrealized_pnl": (current_price - entry_price) * current_position if current_position != 0 else 0,
                "total_pnl": total_pnl
            })
        
        # Calculate performance metrics
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        final_balance = initial_balance + total_pnl
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        # Calculate drawdown
        running_max = initial_balance
        max_drawdown = 0
        for pos in positions:
            balance = initial_balance + pos["total_pnl"]
            running_max = max(running_max, balance)
            drawdown = (running_max - balance) / running_max
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            "strategy": strategy_name,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": initial_balance,
            "final_balance": round(final_balance, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(total_return / (max_drawdown * 100 + 1), 2),
            "trades": trades,
            "positions": positions,
            "bars": bars
        }
        
        return {"success": True, "results": results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/strategies/list")
async def list_strategies() -> Dict[str, Any]:
    """Get available trading strategies."""
    
    strategies = [
        {
            "name": "mnq_808",
            "display_name": "MNQ 808 Strategy",
            "description": "ATR-based momentum strategy with anchor recursion",
            "author": "Pine Script Port",
            "symbols": ["MNQ", "ES", "NQ", "YM", "RTY"],
            "timeframes": ["15m", "1h", "4h"],
            "risk_level": "Medium"
        },
        {
            "name": "vol_breakout", 
            "display_name": "Volatility Breakout",
            "description": "ATR-based volatility breakout with stop loss and targets",
            "author": "QuantZoo",
            "symbols": ["MNQ", "ES", "NQ", "YM", "RTY"],
            "timeframes": ["15m", "1h"],
            "risk_level": "High"
        },
        {
            "name": "momentum",
            "display_name": "Momentum Strategy", 
            "description": "Multi-timeframe momentum with RSI confirmation",
            "author": "QuantZoo",
            "symbols": ["MNQ", "ES", "NQ", "YM", "RTY"],
            "timeframes": ["15m", "1h", "4h"],
            "risk_level": "Medium"
        },
        {
            "name": "regime_hybrid",
            "display_name": "Regime Hybrid",
            "description": "Market regime detection with hybrid signals",
            "author": "QuantZoo", 
            "symbols": ["MNQ", "ES", "NQ", "YM", "RTY"],
            "timeframes": ["1h", "4h", "1d"],
            "risk_level": "Low"
        },
        {
            "name": "pairs",
            "display_name": "Pairs Trading",
            "description": "Statistical arbitrage between correlated instruments",
            "author": "QuantZoo",
            "symbols": ["MNQ/ES", "NQ/RTY"],
            "timeframes": ["15m", "1h"],
            "risk_level": "Medium"
        }
    ]
    
    return {"strategies": strategies}


@app.get("/symbols/list")
async def list_symbols() -> Dict[str, Any]:
    """Get available trading symbols."""
    
    symbols = [
        {
            "symbol": "MNQ",
            "name": "Micro E-mini NASDAQ-100",
            "description": "Tech-heavy index futures contract",
            "exchange": "CME",
            "tick_size": 0.25,
            "tick_value": 0.50,
            "margin": 1300,
            "session_hours": "08:00-16:30 ET"
        },
        {
            "symbol": "ES", 
            "name": "E-mini S&P 500",
            "description": "Broad market index futures contract",
            "exchange": "CME",
            "tick_size": 0.25,
            "tick_value": 12.50,
            "margin": 13000,
            "session_hours": "08:00-16:30 ET"
        },
        {
            "symbol": "NQ",
            "name": "E-mini NASDAQ-100", 
            "description": "Tech-heavy index futures contract",
            "exchange": "CME",
            "tick_size": 0.25,
            "tick_value": 5.00,
            "margin": 17600,
            "session_hours": "08:00-16:30 ET"
        },
        {
            "symbol": "YM",
            "name": "E-mini Dow Jones",
            "description": "Blue chip index futures contract", 
            "exchange": "CBOT",
            "tick_size": 1.0,
            "tick_value": 5.00,
            "margin": 8800,
            "session_hours": "08:00-16:30 ET"
        },
        {
            "symbol": "RTY",
            "name": "E-mini Russell 2000",
            "description": "Small cap index futures contract",
            "exchange": "ICE",
            "tick_size": 0.10,
            "tick_value": 5.00, 
            "margin": 5500,
            "session_hours": "08:00-16:30 ET"
        }
    ]
    
    return {"symbols": symbols}


@app.get("/replay/status")
def replay_status():
    """Get replay engine status."""
    global replay_engine
    
    if not replay_engine:
        return {"status": "not_initialized"}
    
    return replay_engine.get_stats()


# Helper functions for updating global state
def update_state(symbol: str = None, timeframe: str = None, config_path: str = None, 
                replay_speed: float = None, is_running: bool = None):
    """Update current state."""
    global current_state
    tz = pytz.timezone(current_state["timezone"])
    
    if symbol is not None:
        current_state["symbol"] = symbol
    if timeframe is not None:
        current_state["timeframe"] = timeframe
    if config_path is not None:
        current_state["config_path"] = config_path
    if replay_speed is not None:
        current_state["replay_speed"] = replay_speed
    if is_running is not None:
        current_state["is_running"] = is_running
    
    current_state["last_heartbeat"] = datetime.now(tz).isoformat()


def update_position(side: str = None, qty: float = None, avg_price: float = None, 
                   unrealized_pnl: float = None, entry_time = None):
    """Update current position."""
    global current_position
    
    if side is not None:
        current_position["side"] = side
    if qty is not None:
        current_position["qty"] = qty
    if avg_price is not None:
        current_position["avg_price"] = avg_price
    if unrealized_pnl is not None:
        current_position["unrealized_pnl"] = unrealized_pnl
    if entry_time is not None:
        tz = pytz.timezone(current_state["timezone"])
        if isinstance(entry_time, str):
            current_position["entry_time"] = entry_time
        elif hasattr(entry_time, "astimezone"):
            current_position["entry_time"] = entry_time.astimezone(tz).isoformat()
        else:
            current_position["entry_time"] = str(entry_time)


def add_trade(time, side: str, qty: float, price: float, fees_bps: float, 
             slippage_bps: float, reason: str, exit_time = None, 
             exit_price: float = None, pnl: float = None):
    """Add a new trade to recent trades."""
    global recent_trades
    tz = pytz.timezone(current_state["timezone"])
    
    # Convert timestamps to timezone-aware ISO-8601 strings
    time_str = time
    if not isinstance(time, str):
        if hasattr(time, "astimezone"):
            time_str = time.astimezone(tz).isoformat()
        else:
            dt = datetime.fromisoformat(str(time).replace("Z", "+00:00"))
            time_str = dt.astimezone(tz).isoformat()
    
    exit_time_str = exit_time
    if exit_time and not isinstance(exit_time, str):
        if hasattr(exit_time, "astimezone"):
            exit_time_str = exit_time.astimezone(tz).isoformat()
        else:
            dt = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00"))
            exit_time_str = dt.astimezone(tz).isoformat()
    
    trade = {
        "time": time_str,
        "side": side,
        "qty": qty,
        "price": price,
        "fees_bps": fees_bps,
        "slippage_bps": slippage_bps,
        "reason": reason,
        "exit_time": exit_time_str,
        "exit_price": exit_price,
        "pnl": pnl
    }
    
    recent_trades.append(trade)
    
    # Keep only last 1000 trades to prevent memory issues
    if len(recent_trades) > 1000:
        recent_trades[:] = recent_trades[-1000:]


def update_signals(atr: float = None, sma_tr: float = None, momentum_kind: str = None,
                  momentum_value: float = None, anchor_current: float = None,
                  anchor_lag2: float = None, crossover_state: str = None,
                  bar_time = None, bar_price: float = None):
    """Update latest signal data."""
    global latest_signals
    tz = pytz.timezone(current_state["timezone"])
    
    if atr is not None:
        latest_signals["atr"] = atr
    if sma_tr is not None:
        latest_signals["smaTR"] = sma_tr
    if momentum_kind is not None:
        latest_signals["momentum_kind"] = momentum_kind
    if momentum_value is not None:
        latest_signals["momentum_value"] = momentum_value
    if anchor_current is not None:
        latest_signals["anchor_current"] = anchor_current
    if anchor_lag2 is not None:
        latest_signals["anchor_lag2"] = anchor_lag2
    if crossover_state is not None:
        latest_signals["crossover_state"] = crossover_state
    if bar_price is not None:
        latest_signals["bar_price"] = bar_price
    
    if bar_time is not None:
        if isinstance(bar_time, str):
            latest_signals["bar_time"] = bar_time
        elif hasattr(bar_time, "astimezone"):
            latest_signals["bar_time"] = bar_time.astimezone(tz).isoformat()
        else:
            dt = datetime.fromisoformat(str(bar_time).replace("Z", "+00:00"))
            latest_signals["bar_time"] = dt.astimezone(tz).isoformat()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)