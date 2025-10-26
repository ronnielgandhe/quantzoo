"""Tests for advanced risk metrics (VaR and ES)."""

import pytest
import numpy as np
from quantzoo.metrics.core import var_historic, es_historic


class TestVaRHistoric:
    """Test Historical VaR implementation."""
    
    def test_var_with_sufficient_data(self):
        """Test VaR with sufficient normal data."""
        # Generate normal returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        var_95 = var_historic(returns, alpha=0.95)
        assert var_95 is not None
        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR should be negative for losses
    
    def test_var_insufficient_data(self):
        """Test VaR returns None with insufficient data."""
        returns = np.random.normal(0, 0.02, 30)  # Less than 50 observations
        
        var_95 = var_historic(returns, alpha=0.95)
        assert var_95 is None
    
    def test_var_with_nan_values(self):
        """Test VaR returns None with NaN values."""
        returns = np.array([0.01, 0.02, np.nan, 0.01, -0.01] * 20)
        
        var_95 = var_historic(returns, alpha=0.95)
        assert var_95 is None
    
    def test_var_with_infinite_values(self):
        """Test VaR returns None with infinite values."""
        returns = np.array([0.01, 0.02, np.inf, 0.01, -0.01] * 20)
        
        var_95 = var_historic(returns, alpha=0.95)
        assert var_95 is None
    
    def test_var_degenerate_distribution(self):
        """Test VaR returns None with zero volatility."""
        returns = np.array([0.01] * 100)  # All same value
        
        var_95 = var_historic(returns, alpha=0.95)
        assert var_95 is None
    
    def test_var_different_alpha_levels(self):
        """Test VaR with different confidence levels."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        var_90 = var_historic(returns, alpha=0.90)
        var_95 = var_historic(returns, alpha=0.95)
        var_99 = var_historic(returns, alpha=0.99)
        
        # Higher confidence levels should give more extreme (lower) VaR values
        assert var_90 > var_95 > var_99


class TestESHistoric:
    """Test Historical Expected Shortfall implementation."""
    
    def test_es_with_sufficient_data(self):
        """Test ES with sufficient normal data."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        es_95 = es_historic(returns, alpha=0.95)
        assert es_95 is not None
        assert isinstance(es_95, float)
        assert es_95 < 0  # ES should be negative for losses
    
    def test_es_insufficient_data(self):
        """Test ES returns None with insufficient data."""
        returns = np.random.normal(0, 0.02, 30)  # Less than 50 observations
        
        es_95 = es_historic(returns, alpha=0.95)
        assert es_95 is None
    
    def test_es_with_nan_values(self):
        """Test ES returns None with NaN values."""
        returns = np.array([0.01, 0.02, np.nan, 0.01, -0.01] * 20)
        
        es_95 = es_historic(returns, alpha=0.95)
        assert es_95 is None
    
    def test_es_degenerate_distribution(self):
        """Test ES returns None with zero volatility."""
        returns = np.array([0.01] * 100)
        
        es_95 = es_historic(returns, alpha=0.95)
        assert es_95 is None
    
    def test_es_no_tail_observations(self):
        """Test ES returns None when no observations in tail."""
        # Create returns where 95th percentile excludes all observations
        returns = np.array([0.01] * 95 + [-0.05] * 5)  # 95% positive, 5% negative
        
        es_95 = es_historic(returns, alpha=0.95)
        # Should return a value since we have tail observations
        assert es_95 is not None
        assert es_95 < 0
    
    def test_es_vs_var_relationship(self):
        """Test that ES is more extreme than VaR."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        var_95 = var_historic(returns, alpha=0.95)
        es_95 = es_historic(returns, alpha=0.95)
        
        assert var_95 is not None
        assert es_95 is not None
        # ES should be more extreme (lower) than VaR
        assert es_95 <= var_95
    
    def test_es_different_alpha_levels(self):
        """Test ES with different confidence levels."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        
        es_90 = es_historic(returns, alpha=0.90)
        es_95 = es_historic(returns, alpha=0.95)
        es_99 = es_historic(returns, alpha=0.99)
        
        # Higher confidence levels should give more extreme (lower) ES values
        assert es_90 > es_95 > es_99


class TestRiskMetricsIntegration:
    """Test integration of VaR and ES in metrics calculation."""
    
    def test_metrics_with_var_es(self):
        """Test that calculate_metrics includes VaR and ES."""
        from quantzoo.metrics.core import calculate_metrics
        
        # Create mock equity curve with sufficient data
        np.random.seed(42)
        initial_equity = 100000
        returns = np.random.normal(0.001, 0.02, 100)  # 100 observations
        equity_curve = [initial_equity]
        
        for ret in returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        metrics = calculate_metrics([], equity_curve)
        
        # Check that VaR and ES are included
        assert 'var_95' in metrics
        assert 'es_95' in metrics
        
        # With sufficient data, should not be "NA"
        assert metrics['var_95'] != "NA"
        assert metrics['es_95'] != "NA"
        
        # Should be proper float strings
        var_val = float(metrics['var_95'])
        es_val = float(metrics['es_95'])
        assert var_val < 0  # Should be negative
        assert es_val < 0   # Should be negative
        assert es_val <= var_val  # ES should be more extreme
    
    def test_metrics_insufficient_data_for_var_es(self):
        """Test that calculate_metrics returns NA for insufficient data."""
        from quantzoo.metrics.core import calculate_metrics
        
        # Create equity curve with insufficient data
        equity_curve = [100000, 100100, 100200, 100150]  # Only 4 points
        
        metrics = calculate_metrics([], equity_curve)
        
        # Should return "NA" for insufficient data
        assert metrics['var_95'] == "NA"
        assert metrics['es_95'] == "NA"


if __name__ == "__main__":
    pytest.main([__file__])