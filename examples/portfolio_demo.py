#!/usr/bin/env python3
"""
Portfolio Backtesting Demo

This script demonstrates how to use QuantZoo's portfolio backtesting capabilities
to run multi-strategy portfolios with different allocation methods.

Usage:
    python examples/portfolio_demo.py
"""

import pandas as pd
from quantzoo.portfolio.engine import PortfolioEngine
from quantzoo.portfolio.alloc import EqualWeightAllocator, VolTargetAllocator, RiskParityAllocator
from quantzoo.data.loaders import CSVLoader
from quantzoo.strategies import mnq_808, momentum, vol_breakout
import yaml


def load_strategy_config(config_path: str) -> dict:
    """Load strategy configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_portfolio_demo():
    """Run portfolio backtesting demo with different allocation methods."""
    print("QuantZoo Portfolio Demo")
    print("======================")
    
    # Define strategies to include in portfolio
    strategies = [
        {
            'name': 'MNQ808',
            'class': mnq_808.MNQ808,
            'config': load_strategy_config('configs/mnq_808.yaml')
        },
        {
            'name': 'Momentum',
            'class': momentum.Momentum,
            'config': load_strategy_config('configs/momentum.yaml')
        },
        {
            'name': 'VolBreakout', 
            'class': vol_breakout.VolBreakout,
            'config': load_strategy_config('configs/vol_breakout.yaml')
        }
    ]
    
    # Load market data
    print("\nLoading market data...")
    loader = CSVLoader()
    data = loader.load('tests/data/mini_mnq_15m.csv')
    print(f"Loaded {len(data)} bars")
    
    # Test different allocation methods
    allocators = [
        ("Equal Weight", EqualWeightAllocator()),
        ("Vol Target (15%)", VolTargetAllocator(target_vol=0.15)),
        ("Risk Parity", RiskParityAllocator())
    ]
    
    results = {}
    
    for alloc_name, allocator in allocators:
        print(f"\n{'='*50}")
        print(f"Running {alloc_name} Portfolio")
        print(f"{'='*50}")
        
        # Create portfolio engine
        engine = PortfolioEngine(
            strategies=strategies,
            allocator=allocator,
            initial_capital=1_000_000,
            rebalance_freq=21,  # Rebalance every 21 bars
            fee_rate=0.00025,   # 2.5 bps
            slippage_bps=1      # 1 bp
        )
        
        # Run backtest
        result = engine.backtest(data, seed=42)
        results[alloc_name] = result
        
        # Display key metrics
        metrics = result['metrics']
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        # Show strategy allocation
        print("\nStrategy Allocations:")
        for strategy_name, weight in result['final_weights'].items():
            print(f"  {strategy_name}: {weight:.1%}")
    
    # Compare results
    print(f"\n{'='*60}")
    print("Portfolio Comparison")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        name: {
            'Total Return': result['metrics']['total_return'],
            'Sharpe Ratio': result['metrics']['sharpe_ratio'], 
            'Max Drawdown': result['metrics']['max_drawdown'],
            'Win Rate': result['metrics']['win_rate'],
            'Total Trades': result['metrics']['total_trades']
        }
        for name, result in results.items()
    }).T
    
    print(comparison_df.round(3))
    
    # Show equity curves
    print(f"\nEquity curves saved to artifacts/portfolio_comparison.png")
    
    # Plot equity curves (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            equity_curve = result['equity_curve']
            plt.plot(equity_curve.index, equity_curve, label=name, linewidth=2)
            
        plt.title('Portfolio Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('artifacts/portfolio_comparison.png', dpi=300, bbox_inches='tight')
        print("Equity curve plot saved!")
        
    except ImportError:
        print("Install matplotlib to generate equity curve plots")
    
    return results


def demonstrate_rebalancing():
    """Demonstrate portfolio rebalancing mechanics."""
    print(f"\n{'='*60}")
    print("Rebalancing Demonstration")
    print(f"{'='*60}")
    
    # Create simple two-strategy portfolio
    strategies = [
        {
            'name': 'MNQ808',
            'class': mnq_808.MNQ808,
            'config': load_strategy_config('configs/mnq_808.yaml')
        },
        {
            'name': 'Momentum',
            'class': momentum.Momentum, 
            'config': load_strategy_config('configs/momentum.yaml')
        }
    ]
    
    # Use equal weight with frequent rebalancing
    allocator = EqualWeightAllocator()
    engine = PortfolioEngine(
        strategies=strategies,
        allocator=allocator,
        initial_capital=100_000,
        rebalance_freq=5,  # Rebalance every 5 bars for demo
        fee_rate=0.001,    # Higher fees to show impact
        slippage_bps=2
    )
    
    # Load data
    loader = CSVLoader()
    data = loader.load('tests/data/mini_mnq_15m.csv')
    
    # Run with detailed logging
    result = engine.backtest(data[:50], seed=42, verbose=True)  # Only first 50 bars
    
    print(f"\nRebalancing Summary:")
    print(f"Total Rebalances: {len(result['rebalance_dates'])}")
    print(f"Rebalancing Costs: ${result['rebalancing_costs']:,.2f}")
    print(f"Cost as % of Capital: {result['rebalancing_costs']/100000:.2%}")


if __name__ == "__main__":
    # Run portfolio demo
    results = run_portfolio_demo()
    
    # Demonstrate rebalancing
    demonstrate_rebalancing()
    
    print(f"\n{'='*60}")
    print("Demo Complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Try different allocation methods")
    print("2. Experiment with rebalancing frequencies")
    print("3. Add your own strategies to the portfolio")
    print("4. Analyze individual strategy contributions")