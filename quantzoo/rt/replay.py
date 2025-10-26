"""Replay engine for simulated real-time trading."""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import pytz
from .providers import ReplayProvider


class ReplayEngine:
    """Engine for managing replay providers and simulated real-time execution."""
    
    def __init__(self, csv_path: str, speed_factor: float = 1.0, 
                 start_date: str = "2025-01-01", end_date: str = None,
                 timezone: str = "America/Toronto"):
        """Initialize replay engine.
        
        Args:
            csv_path: Path to CSV file with OHLCV data
            speed_factor: Speed multiplier for replay
            start_date: Start date for filtering data (YYYY-MM-DD)
            end_date: End date for filtering data (YYYY-MM-DD), defaults to today
            timezone: Timezone for timestamps
        """
        self.csv_path = csv_path
        self.speed_factor = speed_factor
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.timezone = timezone
        
        self.provider = ReplayProvider(
            csv_path, speed_factor, start_date, end_date, timezone
        )
        self.is_running = False
        self.current_bar: Optional[Dict[str, Any]] = None
        self.bar_count = 0
        self.heartbeat_count = 0
        self.heartbeat_interval = 10  # Publish heartbeat every N bars
        self.last_heartbeat = None
        
    async def start(self, symbols: List[str]):
        """Start the replay engine."""
        try:
            self.provider.subscribe(symbols)
            self.is_running = True
            self.bar_count = 0
            self.heartbeat_count = 0
            
            # Update heartbeat on start
            tz = pytz.timezone(self.timezone)
            self.last_heartbeat = datetime.now(tz).isoformat()
            
            print(f"Starting replay for symbols: {symbols}")
            print(f"Date range: {self.start_date} to {self.end_date}")
            print(f"Timezone: {self.timezone}")
            
            async for bar in self.provider.iter_bars(symbols):
                if not self.is_running:
                    break
                    
                self.current_bar = bar
                self.bar_count += 1
                self.heartbeat_count += 1
                
                # Publish heartbeat every N bars
                if self.heartbeat_count >= self.heartbeat_interval:
                    self.last_heartbeat = datetime.now(tz).isoformat()
                    self.heartbeat_count = 0
                
                print(f"Bar {self.bar_count}: {bar['symbol']} @ {bar['timestamp'][:19]} "
                      f"OHLC: {bar['open']:.2f}/{bar['high']:.2f}/{bar['low']:.2f}/{bar['close']:.2f}")
                
                # Allow other coroutines to run
                await asyncio.sleep(0.001)
                
        except Exception as e:
            print(f"Error in replay engine: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the replay engine."""
        self.is_running = False
        self.provider.close()
        
    def get_latest_bar(self) -> Optional[Dict[str, Any]]:
        """Get the latest bar data."""
        return self.current_bar
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replay statistics."""
        return {
            'is_running': self.is_running,
            'bar_count': self.bar_count,
            'latest_bar': self.current_bar,
            'last_heartbeat': self.last_heartbeat,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'timezone': self.timezone,
            'csv_path': self.csv_path,
            'speed_factor': self.speed_factor
        }