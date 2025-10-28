"""
Export backtest results to leaderboard format for Hugging Face Hub.

Generates markdown leaderboard with:
- Strategy rankings by performance metrics
- Reproducible results with configs
- Links to model cards
- Timestamp and data provenance
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd


def load_backtest_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all backtest result JSON files from directory."""
    results = []
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                result['source_file'] = json_file.name
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def rank_strategies(results: List[Dict[str, Any]], metric: str = 'sharpe_ratio') -> List[Dict[str, Any]]:
    """
    Rank strategies by specified metric.
    
    Args:
        results: List of backtest results
        metric: Metric to rank by (e.g., 'sharpe_ratio', 'total_return', 'max_drawdown')
    
    Returns:
        Sorted list of results
    """
    # Extract metric from nested results
    for result in results:
        if 'metrics' in result and metric in result['metrics']:
            result['_sort_metric'] = result['metrics'][metric]
        else:
            result['_sort_metric'] = float('-inf')
    
    # Sort descending (except for drawdown)
    reverse = metric != 'max_drawdown'
    sorted_results = sorted(results, key=lambda x: x['_sort_metric'], reverse=reverse)
    
    return sorted_results


def generate_leaderboard_md(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate markdown leaderboard file."""
    
    content = f"""# QuantZoo Strategy Leaderboard

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

This leaderboard ranks trading strategies by performance metrics on standardized backtests.

⚠️ **Disclaimer**: Past performance does not guarantee future results. All strategies shown are for research purposes only.

## Methodology

- **Backtesting Engine**: QuantZoo v1.0
- **Fee Model**: 0.05% per trade + $1 minimum
- **Slippage**: 5 basis points
- **Data**: Historical OHLCV data
- **Walk-Forward**: 70/15/15 train/val/test split by time

## Top Strategies by Sharpe Ratio

"""
    
    # Table header
    content += "| Rank | Strategy | Sharpe | Return | Max DD | Win Rate | Trades | Config |\n"
    content += "|------|----------|--------|--------|--------|----------|--------|---------|\n"
    
    # Rank by Sharpe ratio
    ranked = rank_strategies(results, 'sharpe_ratio')
    
    for i, result in enumerate(ranked[:20], 1):  # Top 20
        metrics = result.get('metrics', {})
        strategy_name = result.get('strategy_name', 'Unknown')
        config_file = result.get('config_file', 'N/A')
        
        sharpe = metrics.get('sharpe_ratio', 0.0)
        total_return = metrics.get('total_return', 0.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        win_rate = metrics.get('win_rate', 0.0)
        num_trades = metrics.get('num_trades', 0)
        
        # Format config link
        if config_file != 'N/A':
            config_link = f"[config](configs/{config_file})"
        else:
            config_link = "N/A"
        
        content += f"| {i} | {strategy_name} | {sharpe:.2f} | {total_return:.1f}% | {max_dd:.1f}% | {win_rate:.1f}% | {num_trades} | {config_link} |\n"
    
    # Additional rankings
    content += "\n## Top Strategies by Total Return\n\n"
    content += "| Rank | Strategy | Return | Sharpe | Max DD |\n"
    content += "|------|----------|--------|--------|--------|\n"
    
    ranked_return = rank_strategies(results, 'total_return')
    for i, result in enumerate(ranked_return[:10], 1):
        metrics = result.get('metrics', {})
        strategy_name = result.get('strategy_name', 'Unknown')
        total_return = metrics.get('total_return', 0.0)
        sharpe = metrics.get('sharpe_ratio', 0.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        
        content += f"| {i} | {strategy_name} | {total_return:.1f}% | {sharpe:.2f} | {max_dd:.1f}% |\n"
    
    # Metrics definitions
    content += """
## Metrics Definitions

- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good, >2 is excellent)
- **Total Return**: Cumulative return over backtest period
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
- **Win Rate**: Percentage of profitable trades
- **Trades**: Total number of round-trip trades

## Reproducing Results

All results are reproducible using the configurations linked in the table.

To run a backtest:

```bash
qz backtest --config configs/<config_file>
```

## Model Cards

Detailed model cards for top strategies are available in `model_cards/`:

"""
    
    # Link to model cards if they exist
    model_cards_dir = output_path.parent.parent / "model_cards"
    if model_cards_dir.exists():
        for md_file in model_cards_dir.glob("*.md"):
            if md_file.name != "README.md":
                strategy_name = md_file.stem
                content += f"- [{strategy_name}](model_cards/{md_file.name})\n"
    
    content += """
## Contributing

To add your strategy to the leaderboard:

1. Run backtest with standardized config
2. Submit results JSON
3. Include model card with methodology
4. Open PR for review

## Data Provenance

All backtests use the same historical dataset to ensure fair comparison:
- Source: [Specify data source]
- Period: [Specify date range]
- Symbols: [List symbols]
- Timeframe: [e.g., daily, 15-minute]

## License

All strategies and results are provided under MIT license for research purposes.

---

**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
"""
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Leaderboard generated at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export backtest results to leaderboard")
    parser.add_argument('--input', type=str, required=True, help="Directory containing backtest JSON results")
    parser.add_argument('--output', type=str, required=True, help="Output markdown file path")
    parser.add_argument('--metric', type=str, default='sharpe_ratio', help="Primary ranking metric")
    args = parser.parse_args()
    
    results_dir = Path(args.input)
    output_path = Path(args.output)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Loading backtest results from {results_dir}...")
    results = load_backtest_results(results_dir)
    print(f"Loaded {len(results)} results")
    
    if not results:
        print("Warning: No results found")
        return
    
    print(f"Generating leaderboard ranked by {args.metric}...")
    generate_leaderboard_md(results, output_path)
    
    print(f"\nLeaderboard created: {output_path}")
    print(f"Top strategy: {results[0].get('strategy_name', 'Unknown')}")


if __name__ == "__main__":
    main()
