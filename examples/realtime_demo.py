#!/usr/bin/env python3
"""
Real-Time Trading Demo

This script demonstrates how to use QuantZoo's real-time data infrastructure
to stream market data and run strategies against live feeds.

Usage:
    python examples/realtime_demo.py
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator

from quantzoo.rt.providers import get_provider
from quantzoo.strategies.mnq_808 import MNQ808

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeDemo:
    """Demo class for real-time strategy execution."""
    
    def __init__(self, symbol: str = "MNQ"):
        self.symbol = symbol
        self.strategy = MNQ808()
        self.strategy.on_start()
        self.position = 0
        self.equity = 100000.0
        
    async def run_replay_demo(self, data_file: str, speed: int = 10):
        """Run strategy against replayed historical data."""
        logger.info(f"Starting replay demo for {self.symbol}")
        logger.info(f"Data file: {data_file}, Speed: {speed}x")
        
        # Get replay provider
        provider = get_provider("replay", file_path=data_file)
        
        # Subscribe to data stream
        async for bar in provider.stream_bars(self.symbol, speed=speed):
            await self.process_bar(bar)
            
    async def run_live_demo(self, provider_name: str = "alpaca"):
        """Run strategy against live data (requires API keys)."""
        logger.info(f"Starting live demo for {self.symbol}")
        logger.info(f"Provider: {provider_name}")
        
        # Get live provider (requires environment variables for API keys)
        provider = get_provider(provider_name)
        
        # Subscribe to data stream
        async for bar in provider.stream_bars(self.symbol):
            await self.process_bar(bar)
            
    async def process_bar(self, bar):
        """Process incoming bar data with strategy."""
        # Update strategy with new bar
        signal = self.strategy.on_bar(bar)
        
        # Execute trades based on signal
        if signal == 1 and self.position <= 0:  # Buy signal
            self.position = 1
            logger.info(f"BUY at {bar['close']:.2f} | Time: {bar['timestamp']}")
            
        elif signal == -1 and self.position >= 0:  # Sell signal
            self.position = -1
            logger.info(f"SELL at {bar['close']:.2f} | Time: {bar['timestamp']}")
            
        elif signal == 0 and self.position != 0:  # Exit signal
            logger.info(f"EXIT at {bar['close']:.2f} | Time: {bar['timestamp']}")
            self.position = 0
            
        # Log periodic updates
        if bar.name % 100 == 0:  # Every 100 bars
            logger.info(f"Bar {bar.name}: Price={bar['close']:.2f}, Position={self.position}")


async def main():
    """Main demo function."""
    demo = RealTimeDemo("MNQ")
    
    print("QuantZoo Real-Time Demo")
    print("======================")
    print()
    print("Available demos:")
    print("1. Replay historical data")
    print("2. Live data stream (requires API keys)")
    print()
    
    choice = input("Select demo (1 or 2): ").strip()
    
    if choice == "1":
        # Replay demo using test data
        data_file = "tests/data/mini_mnq_15m.csv"
        speed = int(input("Replay speed (1-100, default=10): ") or "10")
        
        print(f"\nStarting replay demo...")
        print(f"File: {data_file}")
        print(f"Speed: {speed}x")
        print("Press Ctrl+C to stop\n")
        
        try:
            await demo.run_replay_demo(data_file, speed)
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
            
    elif choice == "2":
        # Live demo (requires API keys in environment)
        provider = input("Provider (alpaca/polygon, default=alpaca): ") or "alpaca"
        
        print(f"\nStarting live demo...")
        print(f"Provider: {provider}")
        print("Make sure API keys are set in environment variables")
        print("Press Ctrl+C to stop\n")
        
        try:
            await demo.run_live_demo(provider)
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure API keys are properly configured")
            
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())