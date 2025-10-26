"""Portfolio allocation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AllocationParams:
    """Base parameters for allocation strategies."""
    rebalance_freq: str = "daily"  # daily, weekly, monthly
    min_weight: float = 0.0
    max_weight: float = 1.0
    transaction_cost: float = 0.001  # 10bps default


class BaseAllocator(ABC):
    """Base class for portfolio allocation strategies."""
    
    def __init__(self, params: AllocationParams):
        self.params = params
    
    @abstractmethod
    def calculate_weights(
        self, 
        returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate portfolio weights for given returns."""
        pass
    
    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance: pd.Timestamp) -> bool:
        """Determine if portfolio should be rebalanced."""
        if last_rebalance is None:
            return True
            
        if self.params.rebalance_freq == "daily":
            return current_date.date() != last_rebalance.date()
        elif self.params.rebalance_freq == "weekly":
            return current_date.week != last_rebalance.week
        elif self.params.rebalance_freq == "monthly":
            return current_date.month != last_rebalance.month
        
        return False
    
    def apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints and normalize."""
        # Apply min/max constraints
        constrained = {}
        for symbol, weight in weights.items():
            constrained[symbol] = np.clip(weight, self.params.min_weight, self.params.max_weight)
        
        # Normalize to sum to 1
        total_weight = sum(constrained.values())
        if total_weight > 0:
            constrained = {symbol: weight / total_weight for symbol, weight in constrained.items()}
        
        return constrained


class EqualWeightAllocator(BaseAllocator):
    """Equal weight allocation strategy."""
    
    def calculate_weights(
        self, 
        returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Allocate equal weights to all strategies."""
        strategies = list(returns.columns)
        n_strategies = len(strategies)
        
        if n_strategies == 0:
            return {}
        
        equal_weight = 1.0 / n_strategies
        weights = {strategy: equal_weight for strategy in strategies}
        
        return self.apply_constraints(weights)


class VolTargetAllocator(BaseAllocator):
    """Volatility targeting allocation strategy."""
    
    def __init__(self, params: AllocationParams, target_vol: float = 0.15, lookback: int = 252):
        super().__init__(params)
        self.target_vol = target_vol
        self.lookback = lookback
    
    def calculate_weights(
        self, 
        returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Allocate based on inverse volatility."""
        if len(returns) < 20:  # Minimum data requirement
            # Fallback to equal weight
            fallback = EqualWeightAllocator(self.params)
            return fallback.calculate_weights(returns, current_weights)
        
        # Calculate recent volatilities
        recent_returns = returns.tail(min(self.lookback, len(returns)))
        vols = recent_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Handle zero volatility
        vols = vols.replace(0, vols.median())
        
        # Inverse volatility weights
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        
        # Scale to target volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(recent_returns.cov() * 252, weights)))
        if portfolio_vol > 0:
            vol_scalar = self.target_vol / portfolio_vol
            weights = weights * vol_scalar
        
        weight_dict = weights.to_dict()
        return self.apply_constraints(weight_dict)


class RiskParityAllocator(BaseAllocator):
    """Risk parity allocation strategy."""
    
    def __init__(self, params: AllocationParams, lookback: int = 252):
        super().__init__(params)
        self.lookback = lookback
    
    def calculate_weights(
        self, 
        returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Allocate based on risk contribution parity."""
        if len(returns) < 20:  # Minimum data requirement
            # Fallback to equal weight
            fallback = EqualWeightAllocator(self.params)
            return fallback.calculate_weights(returns, current_weights)
        
        # Calculate covariance matrix
        recent_returns = returns.tail(min(self.lookback, len(returns)))
        cov_matrix = recent_returns.cov() * 252  # Annualized
        
        n_assets = len(returns.columns)
        
        # Simple risk parity approximation: inverse volatility scaled by correlation
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Equal risk contribution weights (simplified)
        weights = 1.0 / vols
        weights = weights / weights.sum()
        
        weight_dict = dict(zip(returns.columns, weights))
        return self.apply_constraints(weight_dict)


def create_allocator(alloc_type: str, params: AllocationParams, **kwargs) -> BaseAllocator:
    """Factory function to create allocators."""
    alloc_type = alloc_type.lower()
    
    if alloc_type == "equal":
        return EqualWeightAllocator(params)
    elif alloc_type == "vol_target":
        target_vol = kwargs.get('target_vol', 0.15)
        lookback = kwargs.get('lookback', 252)
        return VolTargetAllocator(params, target_vol, lookback)
    elif alloc_type == "risk_parity":
        lookback = kwargs.get('lookback', 252)
        return RiskParityAllocator(params, lookback)
    else:
        raise ValueError(f"Unknown allocator type: {alloc_type}")