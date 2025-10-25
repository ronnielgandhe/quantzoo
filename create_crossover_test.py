#!/usr/bin/env python3
"""Create specific test data to verify crossover signals."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_crossover_test_data():
    """Create data that should definitely trigger signals."""
    
    # Create 30 bars during session hours
    times = []
    start_time = datetime(2023, 1, 1, 8, 0)
    for i in range(30):
        times.append(start_time + timedelta(minutes=15*i))
    
    # Create specific price pattern to trigger signals
    data = []
    
    # Bars 0-10: Establish anchor around 4210
    for i in range(11):
        price = 4210 + i * 0.5 + np.random.uniform(-0.5, 0.5)
        high = price + np.random.uniform(1, 2)
        low = price - np.random.uniform(1, 2)
        open_price = price + np.random.uniform(-0.5, 0.5)
        close = price + np.random.uniform(-0.5, 0.5)
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': 1500
        })
    
    # Bars 11-15: Drop below anchor (should be around 4215 by now)
    for i in range(11, 16):
        price = 4205 - (i-11) * 2  # Drop to around 4197
        high = price + 1
        low = price - 1
        open_price = price + 0.5
        close = price - 0.5
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2), 
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': 1500
        })
    
    # Bars 16-20: Cross back above anchor (should trigger long signal)
    for i in range(16, 21):
        price = 4197 + (i-16) * 5  # Rise back to 4217
        high = price + 1
        low = price - 1
        open_price = price - 0.5
        close = price + 0.5
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': 1500
        })
    
    # Bars 21-25: Continue up then drop back down
    for i in range(21, 26):
        price = 4217 + (i-21) * 2  # Up to 4225
        high = price + 1
        low = price - 1
        open_price = price + 0.5
        close = price - 0.5
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2), 
            'close': round(close, 2),
            'volume': 1500
        })
    
    # Bars 26-29: Drop below anchor again (should trigger short signal)
    for i in range(26, 30):
        price = 4225 - (i-26) * 4  # Drop to around 4213
        high = price + 1
        low = price - 1
        open_price = price + 0.5
        close = price - 0.5
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': times[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': 1500
        })
    
    df = pd.DataFrame(data)
    df.to_csv('/Users/ronniel/quantzoo/crossover_test_data.csv', index=False)
    
    print("Created crossover test data:")
    print(df[['time', 'close']].head(15))
    print("...")
    print(df[['time', 'close']].tail(10))
    
    return df

if __name__ == "__main__":
    create_crossover_test_data()