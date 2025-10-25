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
from quantzoo.eval.walkforward import WalkForwardAnalysis
from quantzoo.metrics.core import calculate_metrics
from quantzoo.reports.report_md import generate_report
from quantzoo.reports.leaderboard import generate_leaderboard_report

app = typer.Typer(help="QuantZoo: Trading Strategy Backtesting Framework")


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