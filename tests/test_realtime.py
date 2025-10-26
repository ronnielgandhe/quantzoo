"""Tests for real-time data providers."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch
import pandas as pd

from quantzoo.rt.providers import ReplayProvider, create_provider, AlpacaProvider, PolygonProvider
from quantzoo.rt.replay import ReplayEngine


class TestReplayProvider:
    """Test ReplayProvider functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='15min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5],
            'close': [100.2, 101.2, 102.2, 103.2, 104.2, 105.2, 106.2, 107.2, 108.2, 109.2],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def teardown_method(self):
        """Cleanup test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_replay_provider_init(self):
        """Test ReplayProvider initialization."""
        provider = ReplayProvider(self.temp_file.name, speed_factor=0.1)
        assert provider.csv_path == self.temp_file.name
        assert provider.speed_factor == 0.1
        assert provider.data is None
        assert provider.current_index == 0
    
    def test_replay_provider_subscribe(self):
        """Test subscription to symbols."""
        provider = ReplayProvider(self.temp_file.name)
        provider.subscribe(['MNQ'])
        
        assert provider.symbols == ['MNQ']
        assert provider.data is not None
        assert len(provider.data) == 10
        assert provider.current_index == 0
    
    def test_replay_provider_next_bar(self):
        """Test getting next bar."""
        provider = ReplayProvider(self.temp_file.name, speed_factor=100.0)  # Fast for testing
        provider.subscribe(['MNQ'])
        
        # Get first bar
        bar = provider.next_bar()
        assert bar is not None
        assert bar['symbol'] == 'MNQ'
        assert bar['open'] == 100.0
        assert bar['high'] == 100.5
        assert bar['low'] == 99.5
        assert bar['close'] == 100.2
        assert bar['volume'] == 1000
        assert 'timestamp' in bar
        
        # Get second bar
        bar2 = provider.next_bar()
        assert bar2 is not None
        assert bar2['open'] == 101.0
        assert provider.current_index == 2
    
    def test_replay_provider_exhaustion(self):
        """Test provider returns None when data exhausted."""
        provider = ReplayProvider(self.temp_file.name, speed_factor=100.0)
        provider.subscribe(['MNQ'])
        
        # Consume all bars
        bars = []
        while True:
            bar = provider.next_bar()
            if bar is None:
                break
            bars.append(bar)
        
        assert len(bars) == 10
        assert provider.current_index == 10
        
        # Should return None after exhaustion
        assert provider.next_bar() is None
    
    def test_replay_provider_close(self):
        """Test provider cleanup."""
        provider = ReplayProvider(self.temp_file.name)
        provider.subscribe(['MNQ'])
        assert provider.data is not None
        
        provider.close()
        assert provider.data is None
        assert provider.current_index == 0
        assert provider.symbols == []


class TestProviderFactory:
    """Test provider factory function."""
    
    def test_create_replay_provider(self):
        """Test creating replay provider."""
        provider = create_provider('replay', csv_path='tests/data/mini_mnq_15m.csv')
        assert isinstance(provider, ReplayProvider)
        assert provider.csv_path == 'tests/data/mini_mnq_15m.csv'
    
    def test_create_alpaca_provider_fallback(self):
        """Test Alpaca provider fallback to replay."""
        with patch.dict(os.environ, {}, clear=True):
            # Should fallback to replay when no credentials
            provider = create_provider('alpaca')
            assert isinstance(provider, ReplayProvider)
    
    def test_create_polygon_provider_fallback(self):
        """Test Polygon provider fallback to replay."""
        with patch.dict(os.environ, {}, clear=True):
            # Should fallback to replay when no credentials
            provider = create_provider('polygon')
            assert isinstance(provider, ReplayProvider)
    
    def test_unknown_provider_fallback(self):
        """Test unknown provider fallback to replay."""
        provider = create_provider('unknown_provider')
        assert isinstance(provider, ReplayProvider)


class TestProviderInterfaces:
    """Test provider interface compliance."""
    
    def test_alpaca_provider_requires_credentials(self):
        """Test AlpacaProvider credential requirements."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Alpaca API credentials not found"):
                AlpacaProvider()
    
    def test_polygon_provider_requires_credentials(self):
        """Test PolygonProvider credential requirements."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Polygon API key not found"):
                PolygonProvider()
    
    def test_alpaca_provider_with_credentials(self):
        """Test AlpacaProvider with credentials."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            provider = AlpacaProvider()
            assert provider.api_key == 'test_key'
            assert provider.secret_key == 'test_secret'
    
    def test_polygon_provider_with_credentials(self):
        """Test PolygonProvider with credentials."""
        with patch.dict(os.environ, {'POLYGON_API_KEY': 'test_key'}):
            provider = PolygonProvider()
            assert provider.api_key == 'test_key'


class TestReplayEngine:
    """Test ReplayEngine functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='15min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5],
            'close': [100.2, 101.2, 102.2, 103.2, 104.2],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def teardown_method(self):
        """Cleanup test data."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_replay_engine_init(self):
        """Test ReplayEngine initialization."""
        engine = ReplayEngine(self.temp_file.name, speed_factor=0.1)
        assert engine.provider.csv_path == self.temp_file.name
        assert engine.provider.speed_factor == 0.1
        assert not engine.is_running
        assert engine.current_bar is None
        assert engine.bar_count == 0
    
    @pytest.mark.asyncio
    async def test_replay_engine_start_stop(self):
        """Test starting and stopping replay engine."""
        engine = ReplayEngine(self.temp_file.name, speed_factor=100.0)  # Fast for testing
        
        # Start engine in background
        task = asyncio.create_task(engine.start(['MNQ']))
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop engine
        engine.stop()
        
        # Wait for task to complete
        await task
        
        assert not engine.is_running
        assert engine.bar_count > 0
    
    def test_replay_engine_stats(self):
        """Test getting engine statistics."""
        engine = ReplayEngine(self.temp_file.name)
        stats = engine.get_stats()
        
        assert 'is_running' in stats
        assert 'bar_count' in stats
        assert 'latest_bar' in stats
        assert stats['is_running'] is False
        assert stats['bar_count'] == 0
        assert stats['latest_bar'] is None


if __name__ == "__main__":
    pytest.main([__file__])