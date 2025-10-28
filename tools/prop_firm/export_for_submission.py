"""
Export backtest results in prop-firm portal format.

Generates artifacts for manual submission to prop trading firm eval portals.
DOES NOT automate submission - manual upload required.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import argparse


def export_for_prop_firm(
    backtest_results: Dict[str, Any],
    output_dir: Path,
    firm_format: str = "generic"
) -> Path:
    """
    Export backtest results for prop firm submission.
    
    Args:
        backtest_results: Backtest metrics and trades
        output_dir: Where to save artifacts
        firm_format: Format type ("generic", "topstep", "ftmo")
        
    Returns:
        Path to manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract trades
    trades = backtest_results.get('trades', [])
    trades_df = pd.DataFrame(trades)
    
    # Generate required files
    trades_csv = output_dir / "trades.csv"
    trades_df.to_csv(trades_csv, index=False)
    
    # PnL curve
    equity_curve = backtest_results.get('equity_curve', [])
    equity_df = pd.DataFrame(equity_curve)
    equity_csv = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_csv, index=False)
    
    # Metrics summary
    metrics = backtest_results.get('metrics', {})
    metrics_json = output_dir / "metrics.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Manifest
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'strategy_name': backtest_results.get('strategy_name', 'Unknown'),
        'firm_format': firm_format,
        'files': {
            'trades': str(trades_csv.name),
            'equity_curve': str(equity_csv.name),
            'metrics': str(metrics_json.name)
        },
        'metrics_summary': {
            'total_return': metrics.get('total_return'),
            'sharpe_ratio': metrics.get('sharpe_ratio'),
            'max_drawdown': metrics.get('max_drawdown'),
            'num_trades': len(trades)
        },
        'manual_submission_required': True,
        'instructions': "Upload files to prop firm portal manually. Do not automate submission."
    }
    
    manifest_path = output_dir / "submission_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Prop firm package created at {output_dir}")
    print(f"Files: {list(output_dir.glob('*'))}")
    print("\n📋 MANUAL SUBMISSION REQUIRED:")
    print(f"1. Review files in {output_dir}")
    print("2. Log into prop firm portal")
    print("3. Upload files according to firm's requirements")
    print("4. Keep manifest for your records")
    
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help="Path to backtest results JSON")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--format', default="generic", help="Prop firm format")
    args = parser.parse_args()
    
    with open(args.results) as f:
        results = json.load(f)
    
    export_for_prop_firm(results, Path(args.output), args.format)
