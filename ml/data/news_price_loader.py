"""
Build paired sequences: tokenized news text aligned with price windows and regime labels.

Ensures no look-ahead bias by only using news published before the price window.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class NewsPriceSample:
    """Single sample of aligned news and price data."""
    news_text: str
    price_window: np.ndarray  # Shape: (seq_len, features)
    label: int  # 0=down, 1=up
    timestamp: pd.Timestamp
    symbol: str
    metadata: Dict[str, Any]


class NewsPriceDataset(Dataset):
    """
    Dataset that aligns news articles with price windows.
    
    Ensures no look-ahead bias by only using news published before the price window.
    """
    
    def __init__(
        self,
        news_df: pd.DataFrame,
        price_df: pd.DataFrame,
        price_window_bars: int = 20,
        prediction_horizon_bars: int = 5,
        min_return_threshold: float = 0.001,
        max_news_age_hours: int = 24,
    ):
        """
        Initialize dataset.
        
        Args:
            news_df: DataFrame with columns [timestamp, symbol, text, sentiment]
            price_df: DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
            price_window_bars: Number of bars to use as input features
            prediction_horizon_bars: Number of bars ahead to predict
            min_return_threshold: Minimum return to classify as up/down
            max_news_age_hours: Maximum age of news to consider relevant
        """
        self.news_df = news_df.copy()
        self.price_df = price_df.copy()
        self.price_window_bars = price_window_bars
        self.prediction_horizon_bars = prediction_horizon_bars
        self.min_return_threshold = min_return_threshold
        self.max_news_age_hours = max_news_age_hours
        
        # Ensure timestamps are datetime
        self.news_df['timestamp'] = pd.to_datetime(self.news_df['timestamp'])
        self.price_df['timestamp'] = pd.to_datetime(self.price_df['timestamp'])
        
        # Sort by timestamp
        self.news_df = self.news_df.sort_values('timestamp')
        self.price_df = self.price_df.sort_values(['symbol', 'timestamp'])
        
        self.samples: List[NewsPriceSample] = []
        self._build_samples()
    
    def _build_samples(self) -> None:
        """Build aligned samples ensuring no look-ahead bias."""
        logger.info("Building news-price aligned samples...")
        
        symbols = self.price_df['symbol'].unique()
        
        for symbol in symbols:
            symbol_news = self.news_df[self.news_df['symbol'] == symbol].copy()
            symbol_prices = self.price_df[self.price_df['symbol'] == symbol].copy()
            
            if len(symbol_prices) < self.price_window_bars + self.prediction_horizon_bars:
                logger.warning(f"Insufficient price data for {symbol}, skipping")
                continue
            
            # For each news article
            for _, news_row in symbol_news.iterrows():
                news_time = news_row['timestamp']
                
                # Find price bars after news (no look-ahead)
                future_prices = symbol_prices[symbol_prices['timestamp'] > news_time]
                
                if len(future_prices) < self.price_window_bars + self.prediction_horizon_bars:
                    continue
                
                # Take price window starting after news
                window_start = future_prices.iloc[:self.price_window_bars]
                prediction_bar = future_prices.iloc[self.price_window_bars + self.prediction_horizon_bars - 1]
                
                # Calculate label from future return
                entry_price = window_start['close'].iloc[0]
                exit_price = prediction_bar['close']
                future_return = (exit_price - entry_price) / entry_price
                
                # Classify: 1 if up, 0 if down
                if abs(future_return) < self.min_return_threshold:
                    continue  # Skip neutral samples
                
                label = 1 if future_return > 0 else 0
                
                # Build feature array from price window
                price_features = self._extract_price_features(window_start)
                
                sample = NewsPriceSample(
                    news_text=news_row['text'],
                    price_window=price_features,
                    label=label,
                    timestamp=news_time,
                    symbol=symbol,
                    metadata={
                        'future_return': float(future_return),
                        'sentiment': float(news_row.get('sentiment', 0.0)),
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                    }
                )
                
                self.samples.append(sample)
        
        logger.info(f"Built {len(self.samples)} valid samples")
    
    def _extract_price_features(self, price_window: pd.DataFrame) -> np.ndarray:
        """
        Extract normalized price features from window.
        
        Returns:
            Array of shape (seq_len, num_features)
        """
        features = []
        
        # Normalize prices by first close
        base_price = price_window['close'].iloc[0]
        
        for _, row in price_window.iterrows():
            bar_features = [
                (row['open'] - base_price) / base_price,
                (row['high'] - base_price) / base_price,
                (row['low'] - base_price) / base_price,
                (row['close'] - base_price) / base_price,
                np.log1p(row['volume']) / 10.0,  # Log volume normalized
            ]
            features.append(bar_features)
        
        return np.array(features, dtype=np.float32)
    
    def split_by_time(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple['NewsPriceDataset', 'NewsPriceDataset', 'NewsPriceDataset']:
        """
        Split dataset by time to prevent look-ahead bias.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        # Sort samples by time
        sorted_samples = sorted(self.samples, key=lambda x: x.timestamp)
        
        n = len(sorted_samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_samples = sorted_samples[:train_end]
        val_samples = sorted_samples[train_end:val_end]
        test_samples = sorted_samples[val_end:]
        
        # Create new dataset objects
        train_ds = NewsPriceDataset.__new__(NewsPriceDataset)
        val_ds = NewsPriceDataset.__new__(NewsPriceDataset)
        test_ds = NewsPriceDataset.__new__(NewsPriceDataset)
        
        train_ds.samples = train_samples
        train_ds.price_window_bars = self.price_window_bars
        val_ds.samples = val_samples
        val_ds.price_window_bars = self.price_window_bars
        test_ds.samples = test_samples
        test_ds.price_window_bars = self.price_window_bars
        
        logger.info(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        return train_ds, val_ds, test_ds
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> NewsPriceSample:
        return self.samples[idx]


def load_news_price_data(path: str, config: Dict[str, Any]) -> Tuple[NewsPriceDataset, NewsPriceDataset]:
    """
    Load news-price dataset from parquet file.
    
    Args:
        path: Path to parquet file
        config: Configuration dictionary with keys:
            - price_window_bars: Number of price bars in input window
            - prediction_horizon_bars: Number of bars ahead to predict
            - train_ratio: Fraction for training
            - val_ratio: Fraction for validation
            
    Returns:
        (train_dataset, val_dataset)
    """
    df = pd.read_parquet(path)
    
    # Split into news and price dataframes
    news_cols = ['news_timestamp', 'news_text', 'news_sentiment', 'symbol']
    price_cols = ['price_timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    
    news_df = df[news_cols].drop_duplicates().rename(columns={
        'news_timestamp': 'timestamp',
        'news_text': 'text',
        'news_sentiment': 'sentiment',
    })
    
    price_df = df[price_cols].drop_duplicates().rename(columns={
        'price_timestamp': 'timestamp',
    })
    
    dataset = NewsPriceDataset(
        news_df=news_df,
        price_df=price_df,
        price_window_bars=config.get('price_window_bars', 20),
        prediction_horizon_bars=config.get('prediction_horizon_bars', 5),
    )
    
    train_ds, val_ds, _ = dataset.split_by_time(
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
    )
    
    return train_ds, val_ds


def create_synthetic_sample_data(output_path: Path, n_samples: int = 100, seed: int = 42) -> None:
    """
    Create synthetic news-price dataset for testing.
    
    Args:
        output_path: Where to save parquet file
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = pd.Timestamp('2024-01-01')
    
    records = []
    
    for i in range(n_samples):
        symbol = np.random.choice(symbols)
        
        # Generate news timestamp
        news_time = start_date + pd.Timedelta(hours=i * 6)
        
        # Generate synthetic news
        sentiment = np.random.uniform(-1, 1)
        if sentiment > 0:
            text = f"Positive earnings report for {symbol}. Strong growth expected. Market sentiment bullish."
        else:
            text = f"Concerns about {symbol} market position. Analysts cautious. Bearish outlook prevails."
        
        # Generate 30 price bars after news
        base_price = 100.0 + np.random.uniform(-10, 10)
        
        for bar_idx in range(30):
            price_time = news_time + pd.Timedelta(hours=bar_idx + 1)
            
            # Random walk with slight trend based on sentiment
            drift = sentiment * 0.001
            returns = drift + np.random.normal(0, 0.01)
            
            open_price = base_price
            close_price = base_price * (1 + returns)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.lognormal(10, 1))
            
            records.append({
                'news_timestamp': news_time,
                'news_text': text,
                'news_sentiment': sentiment,
                'symbol': symbol,
                'price_timestamp': price_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            })
            
            base_price = close_price
    
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Created synthetic dataset at {output_path} with {len(records)} records")
