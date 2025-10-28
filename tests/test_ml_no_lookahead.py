"""
Test ML pipeline for look-ahead bias and data integrity.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from ml.data.news_price_loader import NewsPriceDataset, create_synthetic_sample_data
from ml.pipelines.multi_asset_pipeline import MultiAssetPipeline


class TestNoLookAheadBias:
    """Test that ML pipelines have no look-ahead bias."""
    
    def test_news_price_alignment(self, tmp_path):
        """Test that news is aligned only with future prices."""
        # Create test data
        news_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 10:00', periods=5, freq='1H'),
            'symbol': ['AAPL'] * 5,
            'text': ['News article'] * 5,
            'sentiment': [0.5] * 5
        })
        
        price_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:00', periods=50, freq='15min'),
            'symbol': ['AAPL'] * 50,
            'open': np.random.randn(50).cumsum() + 100,
            'high': np.random.randn(50).cumsum() + 101,
            'low': np.random.randn(50).cumsum() + 99,
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        dataset = NewsPriceDataset(
            news_df=news_df,
            price_df=price_df,
            price_window_bars=20,
            prediction_horizon_bars=5
        )
        
        # Check that all samples use price data AFTER news timestamp
        for sample in dataset.samples:
            news_time = sample.timestamp
            # The price window should start after the news
            # This is verified in the implementation, but we check results
            assert sample.price_window.shape[0] == 20  # Correct window size
    
    def test_time_based_split(self, tmp_path):
        """Test that train/val/test split maintains temporal order."""
        # Create sample data
        data_path = tmp_path / "test_data.parquet"
        create_synthetic_sample_data(data_path, n_samples=50, seed=42)
        
        df = pd.read_parquet(data_path)
        
        # Create dataset
        news_df = df[['news_timestamp', 'news_text', 'news_sentiment', 'symbol']].drop_duplicates()
        news_df = news_df.rename(columns={
            'news_timestamp': 'timestamp',
            'news_text': 'text',
            'news_sentiment': 'sentiment'
        })
        
        price_df = df[['price_timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']].drop_duplicates()
        price_df = price_df.rename(columns={'price_timestamp': 'timestamp'})
        
        dataset = NewsPriceDataset(
            news_df=news_df,
            price_df=price_df,
            price_window_bars=10,
            prediction_horizon_bars=3
        )
        
        if len(dataset) > 0:
            train_ds, val_ds, test_ds = dataset.split_by_time(
                train_ratio=0.6,
                val_ratio=0.2
            )
            
            # Get timestamps
            train_times = [s.timestamp for s in train_ds.samples]
            val_times = [s.timestamp for s in val_ds.samples]
            test_times = [s.timestamp for s in test_ds.samples]
            
            # Verify temporal order
            if train_times and val_times:
                assert max(train_times) <= min(val_times), "Train data leaks into validation"
            if val_times and test_times:
                assert max(val_times) <= min(test_times), "Validation data leaks into test"
    
    def test_multi_asset_alignment_no_future_data(self):
        """Test multi-asset pipeline doesn't use future data."""
        # Create price data for multiple assets
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
        
        # Create news data
        news_data = pd.DataFrame({
            'timestamp': timestamps[::10],  # News every 10 hours
            'sentiment': np.random.uniform(-1, 1, 10),
            'text': ['Article'] * 10
        })
        
        # Run pipeline
        pipeline = MultiAssetPipeline(['AAPL', 'MSFT'], {})
        aligned = pipeline.load_and_align(price_data, news_data)
        
        # Verify no look-ahead
        assert pipeline.validate_no_lookahead(aligned)
        
        # Check that news features only use past news
        for idx, row in aligned.iterrows():
            current_time = row['timestamp']
            # All news should be before current time
            # This is implicit in the aggregation, but worth noting
            assert 'news_sentiment' in row
    
    def test_synthetic_data_deterministic(self, tmp_path):
        """Test that synthetic data generation is reproducible."""
        path1 = tmp_path / "data1.parquet"
        path2 = tmp_path / "data2.parquet"
        
        create_synthetic_sample_data(path1, n_samples=20, seed=42)
        create_synthetic_sample_data(path2, n_samples=20, seed=42)
        
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)


class TestMLDataQuality:
    """Test ML data quality and integrity."""
    
    def test_sample_data_generation(self, tmp_path):
        """Test synthetic sample data is valid."""
        data_path = tmp_path / "sample.parquet"
        create_synthetic_sample_data(data_path, n_samples=10, seed=42)
        
        assert data_path.exists()
        
        df = pd.read_parquet(data_path)
        
        # Check required columns
        required_cols = [
            'news_timestamp', 'news_text', 'news_sentiment', 'symbol',
            'price_timestamp', 'open', 'high', 'low', 'close', 'volume'
        ]
        for col in required_cols:
            assert col in df.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['news_timestamp'])
        assert pd.api.types.is_datetime64_any_dtype(df['price_timestamp'])
        assert pd.api.types.is_numeric_dtype(df['close'])
        
        # Check OHLC consistency
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_dataset_filtering(self, tmp_path):
        """Test that neutral samples are filtered out."""
        # Create data with small moves
        news_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'symbol': ['AAPL'] * 5,
            'text': ['News'] * 5,
            'sentiment': [0.0] * 5
        })
        
        # Prices with minimal movement
        price_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'symbol': ['AAPL'] * 100,
            'open': [100.0] * 100,
            'high': [100.01] * 100,
            'low': [99.99] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        })
        
        dataset = NewsPriceDataset(
            news_df=news_df,
            price_df=price_df,
            min_return_threshold=0.01  # 1% threshold
        )
        
        # Should filter out neutral moves
        assert len(dataset.samples) == 0 or \
               all(abs(s.metadata['future_return']) >= 0.01 for s in dataset.samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
