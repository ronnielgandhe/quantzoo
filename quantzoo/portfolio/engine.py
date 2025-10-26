"""Portfolio-level backtesting engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml
from pathlib import Path

from ..backtest.engine import BacktestEngine
from ..data.loaders import load_csv_ohlcv
from ..strategies.mnq_808 import MNQ808, MNQ808Params
from ..strategies.regime_hybrid import RegimeHybrid, RegimeHybridParams
from .alloc import BaseAllocator, create_allocator, AllocationParams


@dataclass
class PortfolioConfig:
    """Configuration for portfolio backtesting."""
    strategies: List[Dict[str, Any]]
    allocation: Dict[str, Any]
    rebalance_freq: str = "daily"
    transaction_cost: float = 0.001
    min_weight: float = 0.0
    max_weight: float = 1.0
    start_cash: float = 100000.0


class PortfolioEngine:
    """Engine for portfolio-level backtesting with multiple strategies."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.allocator = self._create_allocator()
        self.strategy_results: Dict[str, Any] = {}
        self.portfolio_equity: List[float] = []
        self.portfolio_trades: List[Dict[str, Any]] = []
        self.rebalance_dates: List[pd.Timestamp] = []
        
    def _create_allocator(self) -> BaseAllocator:
        """Create allocation strategy from config."""
        alloc_config = self.config.allocation
        
        params = AllocationParams(
            rebalance_freq=alloc_config.get('rebalance_freq', self.config.rebalance_freq),
            min_weight=alloc_config.get('min_weight', self.config.min_weight),
            max_weight=alloc_config.get('max_weight', self.config.max_weight),
            transaction_cost=alloc_config.get('transaction_cost', self.config.transaction_cost)
        )
        
        alloc_type = alloc_config.get('type', 'equal')
        return create_allocator(alloc_type, params, **alloc_config)
    
    def _load_strategy(self, strategy_config: Dict[str, Any]):
        """Load strategy from configuration."""
        strategy_type = strategy_config.get('type', '').lower()
        
        if strategy_type == 'mnq_808':
            params_dict = strategy_config.get('params', {})
            params = MNQ808Params(**params_dict)
            return MNQ808(params)
        elif strategy_type == 'regime_hybrid':
            params_dict = strategy_config.get('params', {})
            params = RegimeHybridParams(**params_dict)
            return RegimeHybrid(params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def run_individual_strategies(self, data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
        """Run each strategy individually."""
        results = {}
        
        for i, strategy_config in enumerate(self.config.strategies):
            strategy_name = strategy_config.get('name', f'strategy_{i}')
            print(f"Running strategy: {strategy_name}")
            
            try:
                # Load strategy
                strategy = self._load_strategy(strategy_config)
                
                # Create backtest engine
                engine = BacktestEngine(
                    initial_cash=self.config.start_cash,
                    commission=0.0,  # Apply commission at portfolio level
                    slippage=0.0     # Apply slippage at portfolio level
                )
                
                # Run backtest
                result = engine.run(data, strategy, seed)
                
                # Store results
                results[strategy_name] = {
                    'result': result,
                    'equity': result.equity_curve,
                    'trades': result.trades,
                    'metrics': result.metrics
                }
                
                print(f"✅ {strategy_name}: {len(result.trades)} trades, "
                      f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Error running {strategy_name}: {e}")
                results[strategy_name] = None
        
        self.strategy_results = results
        return results
    
    def combine_strategies(self) -> pd.DataFrame:
        """Combine individual strategy returns into portfolio."""
        if not self.strategy_results:
            raise ValueError("No strategy results available. Run individual strategies first.")
        
        # Collect valid strategy equity curves
        equity_series = {}
        for name, result in self.strategy_results.items():
            if result is not None and 'equity' in result:
                equity_data = result['equity']
                if isinstance(equity_data, list):
                    equity_series[name] = pd.Series(equity_data)
                elif isinstance(equity_data, pd.Series):
                    equity_series[name] = equity_data
        
        if not equity_series:
            raise ValueError("No valid equity curves found")
        
        # Align series and calculate returns
        equity_df = pd.DataFrame(equity_series)
        equity_df = equity_df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate returns
        returns_df = equity_df.pct_change().fillna(0)
        
        return returns_df
    
    def calculate_portfolio_weights(self, returns: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
        """Calculate portfolio weights over time."""
        weights_history = {}
        last_rebalance = None
        current_weights = None
        
        for date in returns.index:
            if self.allocator.should_rebalance(date, last_rebalance):
                # Calculate new weights based on historical returns up to this point
                historical_returns = returns.loc[:date]
                if len(historical_returns) >= 10:  # Minimum history
                    current_weights = self.allocator.calculate_weights(
                        historical_returns, current_weights
                    )
                    last_rebalance = date
                    self.rebalance_dates.append(date)
                else:
                    # Equal weight fallback for early dates
                    n_strategies = len(returns.columns)
                    current_weights = {col: 1.0/n_strategies for col in returns.columns}
            
            if current_weights:
                weights_history[date] = current_weights.copy()
        
        return weights_history
    
    def calculate_portfolio_returns(
        self, 
        returns: pd.DataFrame, 
        weights_history: Dict[pd.Timestamp, Dict[str, float]]
    ) -> pd.Series:
        """Calculate portfolio returns including transaction costs."""
        portfolio_returns = []
        previous_weights = None
        
        for date in returns.index:
            # Get weights for this date
            current_weights = weights_history.get(date, previous_weights)
            if current_weights is None:
                portfolio_returns.append(0.0)
                continue
            
            # Calculate strategy returns for this date
            date_returns = returns.loc[date]
            
            # Calculate portfolio return
            portfolio_return = sum(
                current_weights.get(strategy, 0) * date_returns.get(strategy, 0)
                for strategy in returns.columns
            )
            
            # Apply transaction costs on rebalancing
            transaction_cost = 0.0
            if previous_weights is not None and date in self.rebalance_dates:
                # Calculate turnover
                turnover = sum(
                    abs(current_weights.get(strategy, 0) - previous_weights.get(strategy, 0))
                    for strategy in returns.columns
                )
                transaction_cost = turnover * self.config.transaction_cost
            
            net_return = portfolio_return - transaction_cost
            portfolio_returns.append(net_return)
            
            previous_weights = current_weights
        
        return pd.Series(portfolio_returns, index=returns.index)
    
    def run_portfolio(self, data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
        """Run complete portfolio backtest."""
        print("🚀 Starting portfolio backtest...")
        
        # Step 1: Run individual strategies
        print("\n📊 Running individual strategies...")
        individual_results = self.run_individual_strategies(data, seed)
        
        # Step 2: Combine strategy returns
        print("\n🔗 Combining strategy returns...")
        returns_df = self.combine_strategies()
        
        # Step 3: Calculate portfolio weights
        print("\n⚖️ Calculating portfolio weights...")
        weights_history = self.calculate_portfolio_weights(returns_df)
        
        # Step 4: Calculate portfolio returns
        print("\n💰 Calculating portfolio returns...")
        portfolio_returns = self.calculate_portfolio_returns(returns_df, weights_history)
        
        # Step 5: Calculate portfolio equity curve
        portfolio_equity = (1 + portfolio_returns).cumprod() * self.config.start_cash
        self.portfolio_equity = portfolio_equity.tolist()
        
        # Step 6: Aggregate trades
        all_trades = []
        for name, result in individual_results.items():
            if result and 'trades' in result:
                for trade in result['trades']:
                    trade_copy = trade.copy()
                    trade_copy['strategy'] = name
                    all_trades.append(trade_copy)
        
        self.portfolio_trades = all_trades
        
        # Step 7: Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_returns)
        
        print(f"\n✅ Portfolio backtest complete!")
        print(f"📈 Total Return: {portfolio_metrics.get('total_return', 0):.3f}")
        print(f"📊 Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"📉 Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.3f}")
        print(f"🔄 Rebalances: {len(self.rebalance_dates)}")
        print(f"💼 Total Trades: {len(all_trades)}")
        
        return {
            'portfolio_equity': self.portfolio_equity,
            'portfolio_returns': portfolio_returns,
            'portfolio_trades': self.portfolio_trades,
            'portfolio_metrics': portfolio_metrics,
            'individual_results': individual_results,
            'weights_history': weights_history,
            'rebalance_dates': self.rebalance_dates
        }
    
    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        if len(returns) == 0:
            return {}
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Sharpe ratio
        mean_return = returns.mean() * 252  # Annualized
        vol = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = mean_return / vol if vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate (positive return days)
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'annual_return': mean_return,
            'annual_volatility': vol
        }


def load_portfolio_config(config_path: str) -> PortfolioConfig:
    """Load portfolio configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return PortfolioConfig(**config_dict)