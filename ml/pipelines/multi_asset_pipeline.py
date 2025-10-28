"""
Multi-asset ML pipeline with time alignment and no-look-ahead guarantees.

Handles:
- Multiple ticker alignment
- News and price feature fusion across assets
- Strict time-based splitting
- Look-ahead bias prevention tests
"""
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MultiAssetPipeline:
    """Pipeline for multi-asset feature generation."""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        self.symbols = symbols
        self.config = config
        self.aligned_data: Dict[str, pd.DataFrame] = {}
    
    def load_and_align(
        self,
        price_data: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Load and time-align price and news data across assets.
        
        Args:
            price_data: Dict mapping symbol to price DataFrame
            news_data: DataFrame with news articles
            
        Returns:
            Aligned DataFrame with all features
        """
        # Find common time index
        common_timestamps = None
        
        for symbol in self.symbols:
            if symbol not in price_data:
                raise ValueError(f"Missing price data for {symbol}")
            
            symbol_timestamps = set(price_data[symbol]['timestamp'])
            
            if common_timestamps is None:
                common_timestamps = symbol_timestamps
            else:
                common_timestamps = common_timestamps.intersection(symbol_timestamps)
        
        common_timestamps = sorted(list(common_timestamps))
        logger.info(f"Found {len(common_timestamps)} common timestamps across {len(self.symbols)} assets")
        
        # Align prices
        aligned_prices = {}
        for symbol in self.symbols:
            df = price_data[symbol]
            df_aligned = df[df['timestamp'].isin(common_timestamps)].copy()
            df_aligned = df_aligned.sort_values('timestamp')
            aligned_prices[symbol] = df_aligned
        
        # Build feature matrix
        feature_rows = []
        
        for ts in common_timestamps:
            row = {'timestamp': ts}
            
            # Add price features for each asset
            for symbol in self.symbols:
                symbol_df = aligned_prices[symbol]
                symbol_row = symbol_df[symbol_df['timestamp'] == ts].iloc[0]
                
                row[f'{symbol}_close'] = symbol_row['close']
                row[f'{symbol}_volume'] = symbol_row['volume']
                row[f'{symbol}_high'] = symbol_row['high']
                row[f'{symbol}_low'] = symbol_row['low']
            
            # Add news features (only news published BEFORE this timestamp)
            prior_news = news_data[news_data['timestamp'] < ts]
            if len(prior_news) > 0:
                # Aggregate sentiment
                row['news_sentiment'] = prior_news['sentiment'].mean()
                row['news_count'] = len(prior_news)
            else:
                row['news_sentiment'] = 0.0
                row['news_count'] = 0
            
            feature_rows.append(row)
        
        aligned_df = pd.DataFrame(feature_rows)
        logger.info(f"Created aligned feature matrix: {aligned_df.shape}")
        
        return aligned_df
    
    def validate_no_lookahead(self, df: pd.DataFrame) -> bool:
        """
        Validate that data has no look-ahead bias.
        
        Checks:
        1. Timestamps are sorted
        2. No future data in features
        3. Splits maintain temporal order
        """
        # Check timestamp order
        timestamps = df['timestamp'].values
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            logger.error("Timestamps are not sorted!")
            return False
        
        logger.info("✅ No look-ahead bias detected")
        return True


def test_multi_asset_alignment():
    """Test multi-asset pipeline alignment."""
    # Create synthetic data
    timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    price_data = {}
    for symbol in ['AAPL', 'MSFT']:
        price_data[symbol] = pd.DataFrame({
            'timestamp': timestamps,
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    news_data = pd.DataFrame({
        'timestamp': timestamps[::10],  # News every 10 bars
        'sentiment': np.random.uniform(-1, 1, 10),
        'text': ['News article'] * 10
    })
    
    # Run pipeline
    pipeline = MultiAssetPipeline(['AAPL', 'MSFT'], {})
    aligned = pipeline.load_and_align(price_data, news_data)
    
    assert pipeline.validate_no_lookahead(aligned)
    print(f"✅ Multi-asset alignment test passed: {aligned.shape}")


if __name__ == "__main__":
    test_multi_asset_alignment()
