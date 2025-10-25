#!/usr/bin/env python3
"""Create longer test data with multiple crossover signals."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_extended_test_data():
    """Create 300+ bars with multiple crossover signals for walk-forward testing."""
    
    start_time = datetime(2023, 1, 1, 8, 0)
    times = []
    data = []
    
    # Create 300 bars (about 10 trading days)
    for day in range(10):
        for hour in range(8, 17):  # 8 AM to 4 PM
            for minute in [0, 15, 30, 45]:
                if hour == 16 and minute > 30:  # Stop at 4:30 PM
                    break
                time_point = start_time + timedelta(days=day, hours=hour-8, minutes=minute)
                times.append(time_point)
    
    # Create price patterns with multiple crossovers
    base_price = 4200
    
    for i, time_point in enumerate(times):
        # Create cyclical pattern with trend + oscillations
        trend = i * 0.5  # Slow uptrend
        cycle = 50 * np.sin(i * 0.1) + 25 * np.sin(i * 0.3)  # Multiple cycles
        noise = np.random.uniform(-2, 2)
        
        price = base_price + trend + cycle + noise
        
        # Create OHLC around the price
        high = price + np.random.uniform(1, 3)
        low = price - np.random.uniform(1, 3) 
        open_price = price + np.random.uniform(-1, 1)
        close = price + np.random.uniform(-1, 1)
        
        # Ensure valid OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': time_point,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(1500 + np.random.uniform(-300, 500))
        })
    
    df = pd.DataFrame(data)
    df.to_csv('/Users/ronniel/quantzoo/tests/data/mini_mnq_15m.csv', index=False)
    
    print(f"Created extended test data with {len(df)} bars")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    
    # Show first and last few bars
    print("\nFirst 5 bars:")
    print(df[['time', 'close']].head())
    print("\nLast 5 bars:")
    print(df[['time', 'close']].tail())
    
    return df

if __name__ == "__main__":
    create_extended_test_data()