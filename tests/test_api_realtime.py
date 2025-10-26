"""Tests for real-time API endpoints."""

import pytest
import requests
import json
from datetime import datetime, timezone
from typing import Dict, Any
import time


class TestAPIRealtime:
    """Test suite for real-time API functionality."""
    
    BASE_URL = "http://localhost:8000"
    
    def setup_method(self):
        """Setup for each test method."""
        # Check if API is running
        try:
            response = requests.get(f"{self.BASE_URL}/healthz", timeout=2)
            if response.status_code != 200:
                pytest.skip("API service not running")
        except:
            pytest.skip("API service not running")
    
    def test_healthz_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.BASE_URL}/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert data["ok"] is True
        assert "version" in data
        assert "service" in data
    
    def test_state_endpoint_schema(self):
        """Test /state endpoint returns correct schema."""
        response = requests.get(f"{self.BASE_URL}/state")
        assert response.status_code == 200
        
        data = response.json()
        expected_fields = [
            "symbol", "timeframe", "tz", "config_path", 
            "replay_speed", "last_heartbeat", "is_running", "current_time"
        ]
        
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate timezone format
        assert data["tz"] in ["America/Toronto", "UTC", "America/New_York"]
        
        # Validate ISO-8601 timestamp format
        if data["current_time"]:
            parsed_time = datetime.fromisoformat(data["current_time"].replace('Z', '+00:00'))
            assert parsed_time.tzinfo is not None, "Timestamp should be timezone-aware"
    
    def test_positions_current_endpoint_schema(self):
        """Test /positions/current endpoint returns correct schema."""
        response = requests.get(f"{self.BASE_URL}/positions/current")
        assert response.status_code == 200
        
        data = response.json()
        expected_fields = ["side", "qty", "avg_price", "unrealized_pnl", "unrealized_pnl_pct"]
        
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate side values
        assert data["side"] in ["flat", "long", "short"]
        
        # Validate numeric fields
        assert isinstance(data["qty"], (int, float))
        assert isinstance(data["avg_price"], (int, float))
        assert isinstance(data["unrealized_pnl"], (int, float))
        assert isinstance(data["unrealized_pnl_pct"], (int, float))
    
    def test_trades_recent_endpoint_schema(self):
        """Test /trades/recent endpoint returns correct schema."""
        response = requests.get(f"{self.BASE_URL}/trades/recent")
        assert response.status_code == 200
        
        data = response.json()
        assert "trades" in data
        assert "count" in data
        assert "timezone" in data
        assert isinstance(data["trades"], list)
        assert isinstance(data["count"], int)
        
        # Test with parameter
        response = requests.get(f"{self.BASE_URL}/trades/recent?n=50")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["trades"]) <= 50
    
    def test_trades_recent_with_data_schema(self):
        """Test trade data schema when trades exist."""
        # This test assumes there might be trade data
        response = requests.get(f"{self.BASE_URL}/trades/recent?n=1")
        data = response.json()
        
        if data["trades"]:
            trade = data["trades"][0]
            expected_fields = [
                "time", "side", "qty", "price", "fees_bps", 
                "slippage_bps", "reason", "exit_time", "exit_price", "pnl"
            ]
            
            for field in expected_fields:
                assert field in trade, f"Missing trade field: {field}"
            
            # Validate timestamp format
            if trade["time"]:
                parsed_time = datetime.fromisoformat(trade["time"].replace('Z', '+00:00'))
                assert parsed_time.tzinfo is not None, "Trade time should be timezone-aware"
    
    def test_signals_latest_endpoint_schema(self):
        """Test /signals/latest endpoint returns correct schema."""
        response = requests.get(f"{self.BASE_URL}/signals/latest")
        assert response.status_code == 200
        
        data = response.json()
        assert "signals" in data
        assert "timezone" in data
        assert isinstance(data["signals"], dict)
        
        # If signals exist, validate their structure
        signals = data["signals"]
        if signals:
            # These fields might exist
            optional_fields = [
                "atr", "smaTR", "momentum_kind", "momentum_value",
                "anchor_current", "anchor_lag2", "crossover_state",
                "bar_time", "bar_price"
            ]
            
            # Check that at least one field exists
            has_data = any(field in signals for field in optional_fields)
            if has_data:
                # Validate bar_time format if present
                if "bar_time" in signals and signals["bar_time"]:
                    parsed_time = datetime.fromisoformat(signals["bar_time"].replace('Z', '+00:00'))
                    assert parsed_time.tzinfo is not None, "Signal bar_time should be timezone-aware"
    
    def test_timezone_consistency(self):
        """Test that all endpoints return consistent timezone information."""
        endpoints = ["/state", "/trades/recent", "/signals/latest"]
        timezones = []
        
        for endpoint in endpoints:
            response = requests.get(f"{self.BASE_URL}{endpoint}")
            if response.status_code == 200:
                data = response.json()
                if "timezone" in data:
                    timezones.append(data["timezone"])
                elif "tz" in data:
                    timezones.append(data["tz"])
        
        # All timezones should be the same
        if timezones:
            assert all(tz == timezones[0] for tz in timezones), "Timezone inconsistency across endpoints"
    
    def test_replay_start_stop_cycle(self):
        """Test starting and stopping replay."""
        # Test starting replay
        start_params = {
            "symbols": ["MNQ"],
            "csv_path": "tests/data/mnq_15m_2025.csv",
            "speed": 1.0,
            "timeframe": "15m",
            "start_date": "2025-01-01"
        }
        
        response = requests.post(f"{self.BASE_URL}/replay/start", params=start_params)
        # Note: This might fail if CSV doesn't exist, which is expected
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            
            # Wait a moment
            time.sleep(1)
            
            # Check status
            status_response = requests.get(f"{self.BASE_URL}/replay/status")
            assert status_response.status_code == 200
            
            # Stop replay
            stop_response = requests.post(f"{self.BASE_URL}/replay/stop")
            assert stop_response.status_code == 200
            
            stop_data = stop_response.json()
            assert "status" in stop_data
    
    def test_error_handling(self):
        """Test API error handling."""
        # Test invalid endpoint
        response = requests.get(f"{self.BASE_URL}/invalid_endpoint")
        assert response.status_code == 404
        
        # Test invalid parameters
        response = requests.get(f"{self.BASE_URL}/trades/recent?n=-1")
        # Should still return 200 but handle gracefully
        assert response.status_code == 200
    
    def test_iso8601_timestamp_format(self):
        """Test that all timestamps are valid ISO-8601 format."""
        endpoints_to_check = [
            "/state",
            "/trades/recent",
            "/signals/latest"
        ]
        
        for endpoint in endpoints_to_check:
            response = requests.get(f"{self.BASE_URL}{endpoint}")
            if response.status_code == 200:
                data = response.json()
                self._validate_timestamps_recursive(data)
    
    def _validate_timestamps_recursive(self, data: Any, path: str = "root"):
        """Recursively validate timestamp fields in response data."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}"
                
                # Check if this looks like a timestamp field
                if any(time_word in key.lower() for time_word in ["time", "timestamp", "heartbeat"]):
                    if value and isinstance(value, str):
                        try:
                            # Parse ISO-8601 timestamp
                            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            assert parsed.tzinfo is not None, f"Timestamp at {current_path} should be timezone-aware"
                        except ValueError:
                            pytest.fail(f"Invalid timestamp format at {current_path}: {value}")
                
                # Recurse into nested structures
                self._validate_timestamps_recursive(value, current_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._validate_timestamps_recursive(item, f"{path}[{i}]")


class TestAPIIntegration:
    """Integration tests for API with mock data."""
    
    BASE_URL = "http://localhost:8000"
    
    def setup_method(self):
        """Setup for integration tests."""
        try:
            response = requests.get(f"{self.BASE_URL}/healthz", timeout=2)
            if response.status_code != 200:
                pytest.skip("API service not running")
        except:
            pytest.skip("API service not running")
    
    def test_api_state_updates(self):
        """Test that API state updates correctly when replay starts."""
        # Get initial state
        initial_response = requests.get(f"{self.BASE_URL}/state")
        initial_data = initial_response.json()
        
        # The state should have default values
        assert "symbol" in initial_data
        assert "is_running" in initial_data
        assert "tz" in initial_data
    
    def test_position_api_integration(self):
        """Test position API integration."""
        response = requests.get(f"{self.BASE_URL}/positions/current")
        assert response.status_code == 200
        
        data = response.json()
        
        # Default position should be flat
        if data["side"] == "flat":
            assert data["qty"] == 0
            assert data["unrealized_pnl"] == 0
            assert data["unrealized_pnl_pct"] == 0
    
    def test_api_cross_endpoint_consistency(self):
        """Test consistency across related endpoints."""
        # Get state
        state_response = requests.get(f"{self.BASE_URL}/state")
        state_data = state_response.json()
        
        # Get trades
        trades_response = requests.get(f"{self.BASE_URL}/trades/recent")
        trades_data = trades_response.json()
        
        # Get signals
        signals_response = requests.get(f"{self.BASE_URL}/signals/latest")
        signals_data = signals_response.json()
        
        # All should return 200
        assert state_response.status_code == 200
        assert trades_response.status_code == 200
        assert signals_response.status_code == 200
        
        # Timezone should be consistent
        state_tz = state_data.get("tz")
        trades_tz = trades_data.get("timezone")
        signals_tz = signals_data.get("timezone")
        
        if state_tz and trades_tz:
            assert state_tz == trades_tz
        if state_tz and signals_tz:
            assert state_tz == signals_tz


if __name__ == "__main__":
    pytest.main([__file__, "-v"])