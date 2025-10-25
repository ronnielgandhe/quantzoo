#!/usr/bin/env python3
"""Create test data with pullbacks to verify strategy is working."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data_with_pullbacks():
    """Create test data that should trigger strategy signals."""
    
    # Create 50 bars of data with pullbacks
    start_time = datetime(2023, 1, 1, 8, 0)
    times = [start_time + timedelta(minutes=15*i) for i in range(50)]
    
    # Create price data with pullbacks
    base_price = 4200
    prices = []
    volumes = []
    
    for i in range(50):
        # Create trend with pullbacks
        if i < 10:
            # Initial uptrend
            price = base_price + i * 2
        elif i < 15:
            # Pullback
            price = base_price + 20 - (i - 10) * 1.5
        elif i < 25:
            # Another uptrend  
            price = base_price + 12.5 + (i - 15) * 3
        elif i < 30:
            # Another pullback
            price = base_price + 42.5 - (i - 25) * 2
        else:
            # Final uptrend
            price = base_price + 32.5 + (i - 30) * 1.5
            
        # Create OHLC around the trend price
        high = price + np.random.uniform(1, 3)
        low = price - np.random.uniform(1, 3)
        open_price = price + np.random.uniform(-1, 1)
        close = price + np.random.uniform(-1, 1)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        prices.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(1500 + np.random.uniform(-300, 500))
        })
    
    # Create DataFrame
    df = pd.DataFrame(prices)
    
    # Save to CSV
    df.to_csv('/Users/ronniel/quantzoo/test_pullback_data.csv', index=False)
    
    print("Created test data with pullbacks:")
    print(df.head(10))
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    
    return df

if __name__ == "__main__":
    create_test_data_with_pullbacks()