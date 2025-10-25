#!/usr/bin/env python3
"""Create sample news data for testing regime hybrid strategy."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_news_data():
    """Create sample news data aligned with price data."""
    
    # Load existing price data to align timestamps
    try:
        price_data = pd.read_csv('/Users/ronniel/quantzoo/tests/data/mini_mnq_15m.csv')
        price_times = pd.to_datetime(price_data['time'])
        
        start_time = price_times.min()
        end_time = price_times.max()
    except:
        # Fallback if price data not available
        start_time = datetime(2023, 1, 1, 8, 0)
        end_time = datetime(2023, 1, 10, 16, 30)
    
    # Generate news headlines with sentiment
    positive_headlines = [
        "Fed signals dovish stance on rates",
        "Tech earnings beat expectations across sector",
        "Strong jobs report shows economic resilience",
        "Market rally continues on positive sentiment",
        "GDP growth exceeds analyst forecasts",
        "Corporate profits surge in latest quarter",
        "Consumer confidence reaches new highs",
        "Bull market gains momentum on optimism",
        "Manufacturing data shows strong expansion",
        "Innovation drive boosts market outlook"
    ]
    
    negative_headlines = [
        "Fed hints at aggressive rate hikes ahead",
        "Tech stocks tumble on weak guidance",
        "Employment data misses expectations badly",
        "Market decline accelerates on concerns",
        "GDP contraction worse than anticipated",
        "Corporate earnings disappoint investors",
        "Consumer sentiment falls to new lows",
        "Bear market fears grip trading floors",
        "Manufacturing slump deepens further",
        "Recession risks weigh on markets"
    ]
    
    neutral_headlines = [
        "Federal Reserve meets to discuss policy",
        "Trading volume remains steady today",
        "Economic indicators show mixed signals",
        "Analysts update sector recommendations",
        "Central bank reviews current stance",
        "Market participants await key data",
        "Quarterly earnings season continues",
        "Regulatory changes under review",
        "Industry leaders discuss outlook",
        "Seasonal patterns emerge in trading"
    ]
    
    # Create news entries
    news_data = []
    current_time = start_time
    
    while current_time <= end_time:
        # Generate 0-3 news items per hour randomly
        num_news = np.random.poisson(1.5)
        
        for _ in range(min(num_news, 3)):
            # Random offset within the hour
            offset_minutes = np.random.randint(0, 60)
            news_time = current_time + timedelta(minutes=offset_minutes)
            
            # Choose sentiment based on some pattern (trend + noise)
            hours_elapsed = (current_time - start_time).total_seconds() / 3600
            trend_sentiment = 0.3 * np.sin(hours_elapsed * 0.1) + 0.1 * (hours_elapsed / 100)
            noise = np.random.normal(0, 0.5)
            sentiment_score = trend_sentiment + noise
            
            # Select headline based on sentiment
            if sentiment_score > 0.3:
                headline = np.random.choice(positive_headlines)
            elif sentiment_score < -0.3:
                headline = np.random.choice(negative_headlines)
            else:
                headline = np.random.choice(neutral_headlines)
            
            news_data.append({
                'timestamp': news_time,
                'headline': headline
            })
        
        current_time += timedelta(hours=1)
    
    # Create DataFrame and save
    news_df = pd.DataFrame(news_data)
    news_df = news_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    news_df.to_csv('/Users/ronniel/quantzoo/tests/data/sample_news.csv', index=False)
    
    print(f"Created {len(news_df)} news items")
    print(f"Date range: {news_df['timestamp'].min()} to {news_df['timestamp'].max()}")
    print("\nSample headlines:")
    print(news_df.head(10)[['timestamp', 'headline']])
    
    return news_df

if __name__ == "__main__":
    create_news_data()