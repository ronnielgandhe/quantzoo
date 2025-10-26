"""Provider interfaces and adapters for real-time data."""

import os
import time
import asyncio
import websocket
import json
import threading
from abc import ABC, abstractmethod
from typing import Protocol, Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
from queue import Queue, Empty


class BaseProvider(Protocol):
    """Interface for real-time data providers."""
    
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols for real-time data."""
        ...
    
    def next_bar(self) -> Optional[Dict[str, Any]]:
        """Get next bar data. Returns None when no more data."""
        ...
    
    def close(self) -> None:
        """Close provider and clean up resources."""
        ...


class ReplayProvider:
    """CSV replay provider for simulated real-time data."""
    
    def __init__(self, csv_path: str, speed_factor: float = 1.0,
                 start_date: str = "2025-01-01", end_date: str = None,
                 timezone: str = "America/Toronto"):
        """Initialize replay provider.
        
        Args:
            csv_path: Path to CSV file with OHLCV data
            speed_factor: Speed multiplier (1.0 = real-time, 0.1 = 10x slower)
            start_date: Start date for filtering data (YYYY-MM-DD)
            end_date: End date for filtering data (YYYY-MM-DD), defaults to today
            timezone: Timezone for timestamps
        """
        self.csv_path = csv_path
        self.speed_factor = speed_factor
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.timezone = timezone
        self.data: Optional[pd.DataFrame] = None
        self.current_index = 0
        self.symbols: List[str] = []
        self.last_bar_time: Optional[float] = None
        
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols."""
        self.symbols = symbols
        self._load_data()
        
    def _load_data(self) -> None:
        """Load CSV data with date filtering and symbol validation."""
        try:
            self.data = pd.read_csv(self.csv_path)
            
            # Handle timestamp column
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            elif 'datetime' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['datetime'])
            else:
                # Create synthetic timestamps
                self.data['timestamp'] = pd.date_range(
                    start=self.start_date, periods=len(self.data), freq='15min'
                )
            
            # Validate symbol if present in data
            if 'symbol' in self.data.columns and self.symbols:
                csv_symbols = set(self.data['symbol'].unique())
                requested_symbols = set(self.symbols)
                
                missing_symbols = requested_symbols - csv_symbols
                if missing_symbols:
                    raise ValueError(
                        f"Requested symbols {missing_symbols} not found in CSV. "
                        f"Available symbols: {csv_symbols}"
                    )
                
                # Filter to requested symbols
                self.data = self.data[self.data['symbol'].isin(self.symbols)]
            
            # Apply date filtering
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date) + pd.Timedelta(days=1)  # Include end date
            
            date_mask = (self.data['timestamp'] >= start_dt) & (self.data['timestamp'] < end_dt)
            self.data = self.data[date_mask]
            
            if len(self.data) == 0:
                raise ValueError(
                    f"No data found in date range {self.start_date} to {self.end_date}. "
                    f"CSV contains data from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}"
                )
            
            # Sort by timestamp
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            self.current_index = 0
            
            print(f"Loaded {len(self.data)} bars from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV data from {self.csv_path}: {e}")
    
    def next_bar(self) -> Optional[Dict[str, Any]]:
        """Get next bar with timing simulation."""
        if self.data is None or self.current_index >= len(self.data):
            return None
            
        # Simulate real-time delays
        current_time = time.time()
        if self.last_bar_time is not None:
            time_since_last = current_time - self.last_bar_time
            expected_delay = 1.0 / self.speed_factor  # Base delay of 1 second
            if time_since_last < expected_delay:
                time.sleep(expected_delay - time_since_last)
        
        self.last_bar_time = time.time()
        
        row = self.data.iloc[self.current_index]
        self.current_index += 1
        
        # Determine symbol
        symbol = self.symbols[0] if self.symbols else "DEFAULT"
        if 'symbol' in row:
            symbol = row['symbol']
            
        bar = {
            'symbol': symbol,
            'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
            'open': float(row.get('open', row.get('Open', 0))),
            'high': float(row.get('high', row.get('High', 0))),
            'low': float(row.get('low', row.get('Low', 0))),
            'close': float(row.get('close', row.get('Close', 0))),
            'volume': int(row.get('volume', row.get('Volume', 0)))
        }
        
        return bar
    
    async def iter_bars(self, symbols: List[str]):
        """Async iterator for streaming bars."""
        self.subscribe(symbols)
        while True:
            bar = self.next_bar()
            if bar is None:
                break
            yield bar
            await asyncio.sleep(0.1)  # Small async delay
    
    def close(self) -> None:
        """Close provider."""
        self.data = None
        self.current_index = 0
        self.symbols = []


class AlphaVantageProvider:
    """Alpha Vantage real-time data provider."""
    
    def __init__(self, api_key: str = None):
        """Initialize Alpha Vantage provider."""
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.symbols: List[str] = []
        self.data_queue = Queue()
        self.is_running = False
        self.poll_thread = None
        
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key not found. "
                "Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = "https://www.alphavantage.co/query"
    
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols for real-time data."""
        self.symbols = symbols
        self.is_running = True
        self._start_polling()
    
    def _start_polling(self):
        """Start polling Alpha Vantage API for real-time data."""
        
        def poll_data():
            while self.is_running:
                try:
                    for symbol in self.symbols:
                        # Get intraday data (1min intervals)
                        params = {
                            'function': 'TIME_SERIES_INTRADAY',
                            'symbol': symbol,
                            'interval': '1min',
                            'apikey': self.api_key,
                            'outputsize': 'compact'
                        }
                        
                        response = requests.get(self.base_url, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            
                            if 'Time Series (1min)' in data:
                                time_series = data['Time Series (1min)']
                                # Get the latest bar
                                latest_time = max(time_series.keys())
                                latest_bar = time_series[latest_time]
                                
                                bar_data = {
                                    'symbol': symbol,
                                    'timestamp': datetime.fromisoformat(latest_time).replace(tzinfo=timezone.utc).isoformat(),
                                    'open': float(latest_bar['1. open']),
                                    'high': float(latest_bar['2. high']),
                                    'low': float(latest_bar['3. low']),
                                    'close': float(latest_bar['4. close']),
                                    'volume': int(latest_bar['5. volume'])
                                }
                                self.data_queue.put(bar_data)
                            
                            elif 'Note' in data:
                                print(f"Alpha Vantage rate limit: {data['Note']}")
                                time.sleep(60)  # Wait 1 minute on rate limit
                            
                            elif 'Error Message' in data:
                                print(f"Alpha Vantage error: {data['Error Message']}")
                        
                        time.sleep(12)  # Alpha Vantage free tier: 5 calls/minute
                    
                    time.sleep(60)  # Poll every minute
                    
                except Exception as e:
                    print(f"Error polling Alpha Vantage: {e}")
                    time.sleep(30)
        
        self.poll_thread = threading.Thread(target=poll_data)
        self.poll_thread.daemon = True
        self.poll_thread.start()
    
    def next_bar(self) -> Optional[Dict[str, Any]]:
        """Get next bar from the queue."""
        try:
            return self.data_queue.get(timeout=1)
        except Empty:
            return None
    
    async def iter_bars(self, symbols: List[str]):
        """Async iterator for streaming bars."""
        self.subscribe(symbols)
        while True:
            bar = self.next_bar()
            if bar:
                yield bar
            await asyncio.sleep(0.1)
    
    def close(self) -> None:
        """Close provider."""
        self.is_running = False
        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=1)


class PolygonProvider:
    """Polygon.io real-time data provider."""
    
    def __init__(self, api_key: str = None):
        """Initialize Polygon provider with real-time capabilities."""
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.symbols: List[str] = []
        self.ws = None
        self.data_queue = Queue()
        self.is_connected = False
        self.ws_thread = None
        
        if not self.api_key:
            raise ValueError(
                "Polygon API key not found. "
                "Set POLYGON_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = "https://api.polygon.io"
        self.ws_url = f"wss://socket.polygon.io/stocks"
    
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time data for symbols."""
        self.symbols = symbols
        self._connect_websocket()
    
    def _connect_websocket(self):
        """Connect to Polygon WebSocket for real-time data."""
        try:
            import websocket
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if isinstance(data, list):
                        for item in data:
                            self._process_message(item)
                    else:
                        self._process_message(data)
                except Exception as e:
                    print(f"Error processing message: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
                self.is_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                print("WebSocket connection closed")
                self.is_connected = False
            
            def on_open(ws):
                print("Connected to Polygon WebSocket")
                self.is_connected = True
                
                # Authenticate
                auth_message = {"action": "auth", "params": self.api_key}
                ws.send(json.dumps(auth_message))
                
                # Subscribe to symbols
                for symbol in self.symbols:
                    sub_message = {
                        "action": "subscribe", 
                        "params": f"AM.{symbol}"  # Aggregate per minute
                    }
                    ws.send(json.dumps(sub_message))
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Run WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except ImportError:
            print("websocket-client not installed. Install with: pip install websocket-client")
            self._use_rest_fallback()
    
    def _process_message(self, message: Dict):
        """Process incoming WebSocket message."""
        if message.get('ev') == 'AM':  # Aggregate per minute
            try:
                bar_data = {
                    'symbol': message.get('sym', ''),
                    'timestamp': datetime.fromtimestamp(message.get('e', 0) / 1000, tz=timezone.utc).isoformat(),
                    'open': float(message.get('o', 0)),
                    'high': float(message.get('h', 0)),
                    'low': float(message.get('l', 0)),
                    'close': float(message.get('c', 0)),
                    'volume': int(message.get('v', 0))
                }
                self.data_queue.put(bar_data)
            except Exception as e:
                print(f"Error processing bar data: {e}")
    
    def _use_rest_fallback(self):
        """Fallback to REST API polling when WebSocket unavailable."""
        print("Using Polygon REST API fallback (polling every 60 seconds)")
        
        def poll_data():
            while True:
                try:
                    for symbol in self.symbols:
                        # Get latest bar data
                        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}"
                        params = {"apikey": self.api_key, "limit": 1}
                        
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('results'):
                                result = data['results'][-1]  # Latest bar
                                bar_data = {
                                    'symbol': symbol,
                                    'timestamp': datetime.fromtimestamp(result['t'] / 1000, tz=timezone.utc).isoformat(),
                                    'open': float(result['o']),
                                    'high': float(result['h']),
                                    'low': float(result['l']),
                                    'close': float(result['c']),
                                    'volume': int(result['v'])
                                }
                                self.data_queue.put(bar_data)
                        
                        time.sleep(1)  # Rate limiting
                    
                    time.sleep(60)  # Poll every minute
                    
                except Exception as e:
                    print(f"Error in REST polling: {e}")
                    time.sleep(10)
        
        self.ws_thread = threading.Thread(target=poll_data)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def next_bar(self) -> Optional[Dict[str, Any]]:
        """Get next bar from the data queue."""
        try:
            return self.data_queue.get(timeout=1)
        except Empty:
            return None
    
    async def iter_bars(self, symbols: List[str]):
        """Async iterator for streaming bars."""
        self.subscribe(symbols)
        while True:
            bar = self.next_bar()
            if bar:
                yield bar
            await asyncio.sleep(0.1)
    
    def close(self) -> None:
        """Close provider and cleanup."""
        self.is_connected = False
        if self.ws:
            self.ws.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=1)


def create_provider(provider_type: str = "replay", **kwargs) -> BaseProvider:
    """Factory function to create providers with fallback to Replay."""
    provider_type = provider_type.lower()
    
    try:
        if provider_type == "polygon":
            return PolygonProvider(**kwargs)
        elif provider_type == "alpha_vantage":
            return AlphaVantageProvider(**kwargs)
        elif provider_type == "replay":
            csv_path = kwargs.get('csv_path', 'tests/data/mnq_15m_2025.csv')
            speed_factor = kwargs.get('speed_factor', 1.0)
            start_date = kwargs.get('start_date', '2025-01-01')
            end_date = kwargs.get('end_date', None)
            timezone = kwargs.get('timezone', 'America/Toronto')
            return ReplayProvider(csv_path, speed_factor, start_date, end_date, timezone)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
    except (ValueError, NotImplementedError) as e:
        print(f"Warning: {e}")
        print("Falling back to ReplayProvider...")
        csv_path = kwargs.get('csv_path', 'tests/data/mnq_15m_2025.csv')
        speed_factor = kwargs.get('speed_factor', 1.0)
        start_date = kwargs.get('start_date', '2025-01-01')
        end_date = kwargs.get('end_date', None)
        timezone = kwargs.get('timezone', 'America/Toronto')
        return ReplayProvider(csv_path, speed_factor, start_date, end_date, timezone)