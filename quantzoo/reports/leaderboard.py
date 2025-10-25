"""Leaderboard generation and formatting utilities."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any


def generate_leaderboard_report() -> str:
    """
    Generate leaderboard markdown report from artifacts.
    
    Returns:
        Path to generated leaderboard report
    """
    artifacts_dir = Path("artifacts")
    
    # Load results data
    results_path = artifacts_dir / "results.csv"
    if not results_path.exists():
        return _generate_empty_leaderboard()
    
    df = pd.read_csv(results_path)
    
    if len(df) == 0:
        return _generate_empty_leaderboard()
    
    # Aggregate by strategy and config
    agg_results = df.groupby(['strategy', 'config_path']).agg({
        'sharpe': 'mean',
        'max_dd': 'mean',
        'trades': 'sum',
        'fees_bps': 'first',
        'slippage_bps': 'first',
        'run_id': 'count'  # Number of windows
    }).reset_index()
    
    # Rename columns for clarity
    agg_results = agg_results.rename(columns={
        'run_id': 'windows',
        'sharpe': 'avg_sharpe',
        'max_dd': 'avg_max_dd'
    })
    
    # Sort by Sharpe ratio descending, then max drawdown ascending
    agg_results = agg_results.sort_values(['avg_sharpe', 'avg_max_dd'], ascending=[False, True])
    
    # Add ranking
    agg_results['rank'] = range(1, len(agg_results) + 1)
    
    # Generate markdown report
    report_content = _format_leaderboard_markdown(agg_results, df)
    
    # Save report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / "leaderboard.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    return str(report_path)


def _format_leaderboard_markdown(agg_results: pd.DataFrame, raw_results: pd.DataFrame) -> str:
    """Format leaderboard data as markdown."""
    
    # Header
    content = ["# QuantZoo Strategy Leaderboard", ""]
    content.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append(f"**Total Configurations:** {len(agg_results)}")
    content.append(f"**Total Runs:** {len(raw_results)}")
    content.append("")
    
    # Summary stats
    content.append("## Summary Statistics")
    content.append("")
    content.append("| Metric | Mean | Median | Best | Worst |")
    content.append("|--------|------|--------|------|-------|")
    
    sharpe_stats = agg_results['avg_sharpe'].describe()
    maxdd_stats = agg_results['avg_max_dd'].describe()
    
    content.append(f"| Sharpe Ratio | {sharpe_stats['mean']:.3f} | {sharpe_stats['50%']:.3f} | {sharpe_stats['max']:.3f} | {sharpe_stats['min']:.3f} |")
    content.append(f"| Max Drawdown | {maxdd_stats['mean']:.3f} | {maxdd_stats['50%']:.3f} | {maxdd_stats['min']:.3f} | {maxdd_stats['max']:.3f} |")
    content.append("")
    
    # Leaderboard table
    content.append("## Strategy Rankings")
    content.append("")
    content.append("Ranked by average Sharpe ratio (descending), then average max drawdown (ascending).")
    content.append("")
    content.append("| Rank | Strategy | Config | Avg Sharpe | Avg Max DD | Total Trades | Windows | Fees (bps) |")
    content.append("|------|----------|--------|------------|------------|--------------|---------|------------|")
    
    for _, row in agg_results.iterrows():
        config_short = Path(row['config_path']).stem
        content.append(
            f"| {row['rank']} | {row['strategy']} | {config_short} | "
            f"{row['avg_sharpe']:.3f} | {row['avg_max_dd']:.3f} | "
            f"{row['trades']} | {row['windows']} | {row['fees_bps']} |"
        )
    
    content.append("")
    
    # Performance insights
    content.append("## Performance Insights")
    content.append("")
    
    if len(agg_results) > 0:
        best_strategy = agg_results.iloc[0]
        content.append(f"**Best Performing Strategy:** {best_strategy['strategy']}")
        content.append(f"- Configuration: {Path(best_strategy['config_path']).stem}")
        content.append(f"- Average Sharpe Ratio: {best_strategy['avg_sharpe']:.3f}")
        content.append(f"- Average Max Drawdown: {best_strategy['avg_max_dd']:.3f}")
        content.append("")
        
        # Strategy comparison
        strategy_summary = agg_results.groupby('strategy').agg({
            'avg_sharpe': 'mean',
            'avg_max_dd': 'mean',
            'trades': 'sum'
        }).round(3)
        
        if len(strategy_summary) > 1:
            content.append("### Strategy Comparison")
            content.append("")
            content.append("| Strategy | Avg Sharpe | Avg Max DD | Total Trades |")
            content.append("|----------|------------|------------|--------------|")
            
            for strategy, row in strategy_summary.iterrows():
                content.append(f"| {strategy} | {row['avg_sharpe']:.3f} | {row['avg_max_dd']:.3f} | {row['trades']} |")
            
            content.append("")
    
    # Footer
    content.append("## Notes")
    content.append("")
    content.append("- Rankings based on walk-forward analysis results")
    content.append("- Sharpe ratio calculated on out-of-sample test periods")
    content.append("- Max drawdown represents peak-to-trough decline")
    content.append("- All results include realistic fees and slippage")
    content.append("")
    content.append("---")
    content.append("*Generated by QuantZoo Leaderboard*")
    
    return "\n".join(content)


def _generate_empty_leaderboard() -> str:
    """Generate empty leaderboard when no results available."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    content = [
        "# QuantZoo Strategy Leaderboard",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## No Results Available",
        "",
        "No backtest results found in `artifacts/results.csv`.",
        "",
        "To populate the leaderboard:",
        "1. Run backtests: `qz run -c <config> -s <seed>`",
        "2. Generate leaderboard: `qz leaderboard`",
        "",
        "---",
        "*Generated by QuantZoo Leaderboard*"
    ]
    
    report_path = reports_dir / "leaderboard.md"
    with open(report_path, "w") as f:
        f.write("\n".join(content))
    
    return str(report_path)