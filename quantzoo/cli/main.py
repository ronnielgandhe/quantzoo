"""Main CLI application for QuantZoo."""

import hashlib
import json
import uuid
import numpy as np
from pathlib import Path
from typing import Optional

import typer
import yaml

from quantzoo.backtest.engine import BacktestEngine
from quantzoo.data.loaders import load_csv_ohlcv, filter_session, load_news_csv, join_news_prices
from quantzoo.strategies.mnq_808 import MNQ808, MNQ808Params
from quantzoo.strategies.regime_hybrid import RegimeHybrid, RegimeHybridParams
from quantzoo.strategies.momentum import Momentum, MomentumParams
from quantzoo.strategies.vol_breakout import VolBreakout, VolBreakoutParams
from quantzoo.strategies.pairs import Pairs, PairsParams
from quantzoo.portfolio.engine import PortfolioEngine, load_portfolio_config
from quantzoo.eval.walkforward import WalkForwardAnalysis
from quantzoo.metrics.core import calculate_metrics
from quantzoo.reports.report_md import generate_report
from quantzoo.reports.leaderboard import generate_leaderboard_report
from quantzoo.store.duck import DuckStore
from quantzoo.rt.replay import ReplayEngine

app = typer.Typer(help="QuantZoo: Trading Strategy Backtesting Framework")


def load_strategy(strategy_name: str, params: dict):
    """Load strategy from name and parameters."""
    strategy_name = strategy_name.lower()
    
    if strategy_name == "mnq_808":
        return MNQ808(MNQ808Params(**params))
    elif strategy_name == "regime_hybrid" or strategy_name == "regimehybrid":
        return RegimeHybrid(RegimeHybridParams(**params))
    elif strategy_name == "momentum":
        return Momentum(MomentumParams(**params))
    elif strategy_name == "vol_breakout":
        return VolBreakout(VolBreakoutParams(**params))
    elif strategy_name == "pairs":
        return Pairs(PairsParams(**params))
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


@app.command()
def run(
    config: str = typer.Option(..., "-c", "--config", help="Path to config file"),
    seed: int = typer.Option(42, "-s", "--seed", help="Random seed for reproducibility"),
) -> None:
    """Run a backtest with the specified configuration."""
    typer.echo(f"🚀 Running backtest with config: {config}")
    
    # Load configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Load data based on strategy requirements
    strategy_type = cfg.get("strategy", {}).get("type", cfg.get("strategy", ""))
    
    if strategy_type.lower() == "regime_hybrid":
        # Load both price and news data
        df = load_csv_ohlcv(cfg["data"]["prices_path"], "US/Eastern", cfg["data"]["timeframe"])
        news_df = load_news_csv(cfg["data"]["news_path"])
        df = join_news_prices(news_df, df, cfg["params"].get("news_window", "30min"))
    else:
        # Load only price data
        data_path = cfg["data"].get("file", cfg["data"].get("path", "tests/data/mini_mnq_15m.csv"))
        df = load_csv_ohlcv(data_path, "US/Eastern", "15min")
    
    # Filter session if needed
    params = cfg.get("params", {})
    if "session_start" in params and "session_end" in params:
        df = filter_session(df, params["session_start"], params["session_end"])
    
    # Initialize strategy
    strategy_name = strategy_type or cfg.get("strategy", "")
    strategy = load_strategy(strategy_name, params)
    
    # Initialize backtest engine
    backtest_cfg = cfg.get("backtest", {})
    engine = BacktestEngine(
        initial_cash=backtest_cfg.get("initial_cash", 100000),
        commission=backtest_cfg.get("commission", 2.0),
        slippage=backtest_cfg.get("slippage", 0.25)
    )
    
    # Run backtest
    result = engine.run(df, strategy, seed)
    
    # Calculate metrics with VaR/ES
    metrics = calculate_metrics(result.trades, result.equity_curve)
    
    # Add additional metadata
    metrics.update({
        "strategy": strategy_name,
        "seed": seed,
        "config_path": config,
        "total_bars": len(df),
        **result.latency_metrics
    })
    
    # Initialize storage
    store = DuckStore()
    
    # Convert trades to DataFrame
    import pandas as pd
    if result.trades:
        trades_df = pd.DataFrame([{
            'symbol': trade.symbol,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'side': trade.side,
            'quantity': trade.quantity,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'commission': trade.commission
        } for trade in result.trades])
    else:
        trades_df = pd.DataFrame()
    
    # Convert equity to DataFrame
    equity_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=len(result.equity_curve), freq='15min'),
        'equity': result.equity_curve
    })
    
    # Store results
    store.write_trades(trades_df, run_id, {
        'seed': seed,
        'strategy_name': strategy_name,
        'config_path': config,
        'metrics': metrics
    })
    
    store.write_equity(equity_df, run_id, {
        'seed': seed,
        'strategy_name': strategy_name,
        'config_path': config,
        'metrics': metrics
    })
    
    typer.echo(f"✅ Backtest completed! Run ID: {run_id}")
    typer.echo(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    typer.echo(f"📉 Max Drawdown: {metrics['max_drawdown']:.3f}")
    typer.echo(f"🎯 Win Rate: {metrics['win_rate']:.3f}")
    typer.echo(f"📈 VaR (95%): {metrics['var_95']}")
    typer.echo(f"📊 ES (95%): {metrics['es_95']}")
    typer.echo(f"💾 Results stored with ID: {run_id}")


@app.command()
def run_portfolio(
    config: str = typer.Option(..., "-c", "--config", help="Path to portfolio config file"),
    seed: int = typer.Option(42, "-s", "--seed", help="Random seed for reproducibility"),
) -> None:
    """Run portfolio-level backtest with multiple strategies."""
    typer.echo(f"🎯 Running portfolio backtest with config: {config}")
    
    # Load portfolio configuration
    portfolio_config = load_portfolio_config(config)
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Load data (assuming all strategies use same data for simplicity)
    df = load_csv_ohlcv("tests/data/mini_mnq_15m.csv", "US/Eastern", "15min")
    
    # Initialize portfolio engine
    portfolio_engine = PortfolioEngine(portfolio_config)
    
    # Run portfolio backtest
    portfolio_result = portfolio_engine.run_portfolio(df, seed)
    
    # Calculate portfolio metrics
    portfolio_metrics = portfolio_result['portfolio_metrics']
    
    # Initialize storage
    store = DuckStore()
    
    # Store portfolio results
    import pandas as pd
    
    # Portfolio trades
    if portfolio_result['portfolio_trades']:
        portfolio_trades_df = pd.DataFrame(portfolio_result['portfolio_trades'])
    else:
        portfolio_trades_df = pd.DataFrame()
    
    # Portfolio equity
    portfolio_equity_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=len(portfolio_result['portfolio_equity']), freq='15min'),
        'equity': portfolio_result['portfolio_equity']
    })
    
    # Store results
    store.write_trades(portfolio_trades_df, f"portfolio_{run_id}", {
        'seed': seed,
        'strategy_name': 'portfolio',
        'config_path': config,
        'metrics': portfolio_metrics,
        'allocation_type': portfolio_config.allocation.get('type', 'equal'),
        'num_strategies': len(portfolio_config.strategies)
    })
    
    store.write_equity(portfolio_equity_df, f"portfolio_{run_id}", {
        'seed': seed,
        'strategy_name': 'portfolio',
        'config_path': config,
        'metrics': portfolio_metrics
    })
    
    typer.echo(f"✅ Portfolio backtest completed! Run ID: portfolio_{run_id}")
    typer.echo(f"📊 Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
    typer.echo(f"📉 Max Drawdown: {portfolio_metrics['max_drawdown']:.3f}")
    typer.echo(f"🔄 Rebalances: {len(portfolio_result['rebalance_dates'])}")
    typer.echo(f"💼 Total Trades: {len(portfolio_result['portfolio_trades'])}")


@app.command()
def ingest_replay(
    csv_path: str = typer.Option(..., "-p", "--path", help="Path to CSV file"),
    symbol: str = typer.Option("MNQ", "-s", "--symbol", help="Symbol name"),
    speed: float = typer.Option(1.0, "--speed", help="Replay speed factor"),
) -> None:
    """Start replay provider for real-time simulation."""
    typer.echo(f"🔄 Starting replay for {symbol} from {csv_path} at {speed}x speed")
    
    try:
        engine = ReplayEngine(csv_path, speed)
        import asyncio
        asyncio.run(engine.start([symbol]))
    except KeyboardInterrupt:
        typer.echo("\n⏹️ Replay stopped by user")
    except Exception as e:
        typer.echo(f"❌ Replay failed: {e}")
        raise typer.Exit(1)


@app.command()
def report(
    run_id: str = typer.Option(..., "-r", "--run-id", help="Run ID to generate report for"),
) -> None:
    """Generate a markdown report for a specific run."""
    typer.echo(f"📝 Generating report for run: {run_id}")
    
    # Try to load from DuckDB store first
    store = DuckStore()
    runs = store.list_runs()
    
    run_data = None
    for run in runs:
        if run['run_id'] == run_id:
            run_data = run
            break
    
    if not run_data:
        # Fallback to legacy metrics.json
        try:
            with open("artifacts/metrics.json", "r") as f:
                metrics_data = json.load(f)
            
            if run_id not in metrics_data:
                typer.echo(f"❌ Run ID {run_id} not found")
                raise typer.Exit(1)
            
            run_data = metrics_data[run_id]
        except FileNotFoundError:
            typer.echo(f"❌ No data found for run ID {run_id}")
            raise typer.Exit(1)
    
    # Generate report
    report_path = generate_report(run_id, run_data)
    typer.echo(f"✅ Report generated: {report_path}")


@app.command()
def leaderboard() -> None:
    """Generate leaderboard report from all backtest results."""
    typer.echo("🏆 Generating leaderboard from stored results...")
    
    try:
        report_path = generate_leaderboard_report()
        typer.echo(f"✅ Leaderboard generated: {report_path}")
    except Exception as e:
        typer.echo(f"❌ Failed to generate leaderboard: {e}")
        raise typer.Exit(1)


@app.command()
def list_runs() -> None:
    """List all stored backtest runs."""
    typer.echo("📋 Listing all stored runs...")
    
    store = DuckStore()
    runs = store.list_runs()
    
    if not runs:
        typer.echo("No runs found in storage.")
        return
    
    typer.echo(f"\nFound {len(runs)} runs:")
    typer.echo("-" * 80)
    
    for run in runs[:10]:  # Show last 10 runs
        typer.echo(f"Run ID: {run['run_id']}")
        typer.echo(f"  Strategy: {run.get('strategy_name', 'Unknown')}")
        typer.echo(f"  Timestamp: {run.get('timestamp', 'Unknown')}")
        typer.echo(f"  Sharpe: {run.get('sharpe_ratio', 0):.3f}")
        typer.echo(f"  Trades: {run.get('total_trades', 0)}")
        typer.echo("")


@app.command()
def cleanup(
    keep: int = typer.Option(50, "--keep", help="Number of recent runs to keep"),
) -> None:
    """Clean up old backtest artifacts."""
    typer.echo(f"🧹 Cleaning up old runs, keeping last {keep}...")
    
    store = DuckStore()
    store.cleanup_old_runs(keep_last_n=keep)
    
    typer.echo("✅ Cleanup completed!")


if __name__ == "__main__":
    app()


@app.command()
def run(
    config: str = typer.Option(..., "-c", "--config", help="Path to config file"),
    seed: int = typer.Option(42, "-s", "--seed", help="Random seed for reproducibility"),
) -> None:
    """Run a backtest with the specified configuration."""
    typer.echo(f"Running backtest with config: {config}")
    
    # Load configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Load data
    if cfg["strategy"] == "RegimeHybrid":
        # Load both price and news data
        df = load_csv_ohlcv(cfg["data"]["prices_path"], "US/Eastern", cfg["data"]["timeframe"])
        news_df = load_news_csv(cfg["data"]["news_path"])
        
        # Join news with prices
        df = join_news_prices(news_df, df, cfg["params"].get("news_window", "30min"))
    else:
        # Load only price data
        df = load_csv_ohlcv(cfg["data"]["path"], "US/Eastern", cfg["data"]["timeframe"])
    
    # Filter session if needed
    if "session_start" in cfg["params"] and "session_end" in cfg["params"]:
        df = filter_session(df, cfg["params"]["session_start"], cfg["params"]["session_end"])
    
    # Initialize strategy
    if cfg["strategy"] == "MNQ808":
        params = MNQ808Params(**cfg["params"])
        strategy = MNQ808(params)
    elif cfg["strategy"] == "RegimeHybrid":
        params = RegimeHybridParams(**cfg["params"])
        strategy = RegimeHybrid(params)
    else:
        raise ValueError(f"Unknown strategy: {cfg['strategy']}")
    
    # Run walk-forward analysis
    wf_analysis = WalkForwardAnalysis(
        kind=cfg["eval"]["walkforward"]["kind"],
        train_bars=cfg["eval"]["walkforward"]["train_bars"],
        test_bars=cfg["eval"]["walkforward"]["test_bars"],
    )
    
    results = wf_analysis.run(
        df=df,
        strategy=strategy,
        fees_bps=cfg["fees_bps"],
        slippage_bps=cfg["slippage_bps"],
        seed=seed,
    )
    
    # Calculate aggregate metrics
    all_trades = []
    all_equity = []
    all_latency_metrics = []
    
    for result in results:
        all_trades.extend(result["trades"])
        all_equity.extend(result["equity_curve"])
        
        # Collect latency metrics
        if "latency_metrics" in result:
            all_latency_metrics.append(result["latency_metrics"])
    
    metrics = calculate_metrics(all_trades, all_equity)
    
    # Aggregate latency metrics
    if all_latency_metrics:
        latency_means = [lm.get("latency_ms_mean", 0) for lm in all_latency_metrics]
        latency_p95s = [lm.get("latency_ms_p95", 0) for lm in all_latency_metrics]
        
        aggregated_latency = {
            "latency_ms_mean": np.mean(latency_means),
            "latency_ms_p95": np.mean(latency_p95s),
            "latency_ms_max": max([lm.get("latency_ms_max", 0) for lm in all_latency_metrics])
        }
        metrics.update(aggregated_latency)
    
    # Save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save results CSV
    results_df = []
    for i, result in enumerate(results):
        window_metrics = calculate_metrics(result["trades"], result["equity_curve"])
        results_df.append({
            "strategy": cfg["strategy"],
            "window": i,
            "start_date": result["start_date"],
            "end_date": result["end_date"],
            "sharpe": window_metrics["sharpe_ratio"],
            "max_dd": window_metrics["max_drawdown"],
            "fees_bps": cfg["fees_bps"],
            "slippage_bps": cfg["slippage_bps"],
            "trades": len(result["trades"]),
            "seed": seed,
            "config_path": config,
            "commit_sha": "unknown",  # Would use git in real implementation
            "run_id": run_id,
        })
    
    import pandas as pd
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(artifacts_dir / "results.csv", index=False)
    
    # Save metrics JSON
    metrics_data = {
        run_id: {
            **metrics,
            "strategy": cfg["strategy"],
            "fees_bps": cfg["fees_bps"],
            "slippage_bps": cfg["slippage_bps"],
            "seed": seed,
            "config_path": config,
            "commit_sha": "unknown",
            "params": cfg["params"],
        }
    }
    
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    
    typer.echo(f"✅ Backtest completed! Run ID: {run_id}")
    typer.echo(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    typer.echo(f"📉 Max Drawdown: {metrics['max_drawdown']:.3f}")
    typer.echo(f"🎯 Win Rate: {metrics['win_rate']:.3f}")


@app.command()
def report(
    run_id: str = typer.Option(..., "-r", "--run-id", help="Run ID to generate report for"),
) -> None:
    """Generate a markdown report for a specific run."""
    typer.echo(f"Generating report for run: {run_id}")
    
    # Load metrics
    with open("artifacts/metrics.json", "r") as f:
        metrics_data = json.load(f)
    
    if run_id not in metrics_data:
        typer.echo(f"❌ Run ID {run_id} not found in metrics.json")
        raise typer.Exit(1)
    
    # Generate report
    report_path = generate_report(run_id, metrics_data[run_id])
    typer.echo(f"✅ Report generated: {report_path}")


@app.command()
def leaderboard() -> None:
    """Generate leaderboard report from all backtest results."""
    typer.echo("Generating leaderboard from artifacts...")
    
    try:
        report_path = generate_leaderboard_report()
        typer.echo(f"✅ Leaderboard generated: {report_path}")
    except Exception as e:
        typer.echo(f"❌ Failed to generate leaderboard: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()