"""Core performance metrics calculation."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from quantzoo.backtest.engine import Trade


def var_historic(returns: np.ndarray, alpha: float = 0.95) -> Optional[float]:
    """
    Calculate Historical Value-at-Risk with robust guards.
    
    Args:
        returns: Array of returns
        alpha: Confidence level (0.95 = 95% VaR)
        
    Returns:
        VaR value or None if insufficient data
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    
    # Robust guards
    if n < 50 or not np.isfinite(r).all():
        return None
    
    # Check for degenerate distribution
    if np.std(r) == 0:
        return None
    
    q = np.quantile(r, 1 - alpha)
    return float(q)


def es_historic(returns: np.ndarray, alpha: float = 0.95) -> Optional[float]:
    """
    Calculate Historical Expected Shortfall (Conditional VaR) with robust guards.
    
    Args:
        returns: Array of returns
        alpha: Confidence level (0.95 = 95% ES)
        
    Returns:
        ES value or None if insufficient data
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    
    # Robust guards
    if n < 50 or not np.isfinite(r).all():
        return None
    
    # Check for degenerate distribution
    if np.std(r) == 0:
        return None
    
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    
    if tail.size == 0:
        return None
    
    return float(tail.mean())


def calculate_metrics(trades: List[Trade], equity_curve: List[float]) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics including VaR and ES.
    
    Args:
        trades: List of completed trades
        equity_curve: List of equity values over time
        
    Returns:
        Dictionary with calculated metrics
    """
    if not trades and not equity_curve:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "exposure": 0.0,
            "var_95": "NA",
            "es_95": "NA",
        }
    
    # Basic trade statistics
    total_trades = len(trades)
    pnls = [trade.pnl for trade in trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl < 0]
    
    total_pnl = sum(pnls) if pnls else 0.0
    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
    
    # Win rate
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    # Average trades
    avg_trade = np.mean(pnls) if pnls else 0.0
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0
    
    # Largest win/loss
    largest_win = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0
    
    # Initialize risk metrics
    var_95 = "NA"
    es_95 = "NA"
    
    # Equity curve metrics
    if equity_curve:
        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Convert to returns
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        returns = returns[np.isfinite(returns)]  # Remove any inf/nan values
        
        # Calculate VaR and ES
        var_result = var_historic(returns, alpha=0.95)
        es_result = es_historic(returns, alpha=0.95)
        
        if var_result is not None:
            var_95 = f"{var_result:.4f}"
        if es_result is not None:
            es_95 = f"{es_result:.4f}"
        
        # Sharpe ratio (annualized, assuming 252 trading days, 24 bars per day for 15min data)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            # Annualization factor for 15-minute bars: 252 days * 24 bars/day = 6048
            annualization_factor = np.sqrt(6048)
            sharpe_ratio = (mean_return * annualization_factor) / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    else:
        total_return = 0.0
        sharpe_ratio = 0.0
        max_dd = 0.0
    
    # Exposure (percentage of time in position)
    if trades:
        total_time_in_position = sum((trade.exit_time - trade.entry_time).total_seconds() for trade in trades)
        if trades:
            total_period = (trades[-1].exit_time - trades[0].entry_time).total_seconds()
            exposure = total_time_in_position / total_period if total_period > 0 else 0.0
        else:
            exposure = 0.0
    else:
        exposure = 0.0
    
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "exposure": exposure,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "var_95": var_95,
        "es_95": es_95,
    }


def calculate_rolling_sharpe(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """Calculate rolling Sharpe ratio."""
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    
    return rolling_mean / rolling_std * np.sqrt(252)


def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    if max_drawdown == 0:
        return 0.0
    
    annual_return = np.mean(returns) * 252  # Assuming daily returns
    return annual_return / max_drawdown


def calculate_sortino_ratio(returns: np.ndarray) -> float:
    """Calculate Sortino ratio (return / downside deviation)."""
    excess_returns = returns - 0  # Assuming 0 risk-free rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_std * np.sqrt(252)


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Calculate Information ratio."""
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(active_returns) / tracking_error * np.sqrt(252)