"""Data loading and processing utilities for OHLCV data."""

from typing import Optional
import pandas as pd
import numpy as np
from datetime import time


def load_csv_ohlcv(path: str, tz: str, timeframe: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    
    Args:
        path: Path to CSV file with columns: time, open, high, low, close, volume
        tz: Timezone string (e.g., "US/Eastern")
        timeframe: Timeframe string (e.g., "15m", "1h", "1d")
        
    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    # Load CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["time", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])
    
    # Set timezone if specified
    if tz:
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(tz)
        else:
            df["time"] = df["time"].dt.tz_convert(tz)
    
    # Set time as index
    df = df.set_index("time")
    
    # Sort by time
    df = df.sort_index()
    
    # Validate OHLC relationships
    invalid_ohlc = (
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["high"] < df["low"])
    )
    
    if invalid_ohlc.any():
        n_invalid = invalid_ohlc.sum()
        print(f"Warning: {n_invalid} bars have invalid OHLC relationships")
    
    return df


def filter_session(
    df: pd.DataFrame, 
    start: str, 
    end: str,
    timezone: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only bars within specified session hours.
    
    Args:
        df: DataFrame with datetime index
        start: Session start time as string (e.g., "08:00")
        end: Session end time as string (e.g., "16:30")
        timezone: Optional timezone to convert to before filtering
        
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    # Convert timezone if specified
    if timezone and df_filtered.index.tz != timezone:
        df_filtered.index = df_filtered.index.tz_convert(timezone)
    
    # Parse time strings
    start_time = time.fromisoformat(start)
    end_time = time.fromisoformat(end)
    
    # Filter by time
    mask = (
        (df_filtered.index.time >= start_time) &
        (df_filtered.index.time <= end_time)
    )
    
    return df_filtered[mask]


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range for each bar.
    
    Args:
        df: DataFrame with high, low, close columns
        
    Returns:
        Series with True Range values
    """
    high_low = df["high"] - df["low"]
    high_close_prev = np.abs(df["high"] - df["close"].shift(1))
    low_close_prev = np.abs(df["low"] - df["close"].shift(1))
    
    true_range = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
    return pd.Series(true_range, index=df.index, name="true_range")


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period for ATR calculation
        
    Returns:
        Series with ATR values
    """
    tr = calculate_true_range(df)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Input series
        period: Lookback period
        
    Returns:
        Series with SMA values
    """
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        series: Price series (typically close)
        period: Lookback period
        
    Returns:
        Series with RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_mfi(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate Money Flow Index.
    
    Args:
        df: DataFrame with high, low, close, volume columns
        period: Lookback period
        
    Returns:
        Series with MFI values
    """
    # Calculate typical price (HLC3)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    
    # Calculate raw money flow
    rmf = tp * df["volume"]
    
    # Separate positive and negative money flows
    tp_diff = tp.diff()
    positive_mf = rmf.where(tp_diff > 0, 0).rolling(window=period).sum()
    negative_mf = rmf.where(tp_diff < 0, 0).rolling(window=period).sum()
    
    # Calculate MFI
    mfr = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi


def load_news_csv(path: str, text_col: str = "headline", time_col: str = "timestamp", tz: Optional[str] = None) -> pd.DataFrame:
    """
    Load news data from CSV file.
    
    Args:
        path: Path to CSV file with news data
        text_col: Column name containing text/headlines
        time_col: Column name containing timestamps
        tz: Timezone string (e.g., "US/Eastern")
        
    Returns:
        DataFrame with datetime index and text column
    """
    # Load CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV")
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in CSV")
    
    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Set timezone if specified
    if tz:
        if df[time_col].dt.tz is None:
            df[time_col] = df[time_col].dt.tz_localize(tz)
        else:
            df[time_col] = df[time_col].dt.tz_convert(tz)
    
    # Set time as index and rename text column
    df = df.set_index(time_col)
    df = df.rename(columns={text_col: "text"})
    
    # Sort by time
    df = df.sort_index()
    
    return df[["text"]]


def join_news_prices(news_df: pd.DataFrame, prices_df: pd.DataFrame, window: str = "30min") -> pd.DataFrame:
    """
    Join news data with price data by aggregating news within time windows.
    
    Args:
        news_df: DataFrame with datetime index and 'text' column
        prices_df: DataFrame with datetime index and OHLCV columns
        window: Time window for news aggregation (e.g., "30min", "1h")
        
    Returns:
        DataFrame with price data and aggregated news features
    """
    # Create copy of prices to avoid modifying original
    result_df = prices_df.copy()
    
    # Align news to price bar timestamps
    aligned_news = []
    
    for bar_time in prices_df.index:
        # Define window boundaries
        window_start = bar_time - pd.Timedelta(window)
        window_end = bar_time
        
        # Get news within window
        window_news = news_df[
            (news_df.index > window_start) & 
            (news_df.index <= window_end)
        ]
        
        # Aggregate news features
        if len(window_news) > 0:
            # Count of news items
            news_count = len(window_news)
            
            # Concatenate all headlines
            combined_text = " ".join(window_news["text"].fillna("").astype(str))
            
            # Simple sentiment scoring (placeholder for more sophisticated analysis)
            positive_words = ["up", "rise", "gain", "bull", "positive", "growth", "beat", "strong"]
            negative_words = ["down", "fall", "drop", "bear", "negative", "decline", "miss", "weak"]
            
            text_lower = combined_text.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            # Net sentiment
            net_sentiment = positive_score - negative_score
            sentiment_ratio = (positive_score + 1) / (negative_score + 1)  # Add 1 to avoid division by zero
            
        else:
            news_count = 0
            combined_text = ""
            net_sentiment = 0
            sentiment_ratio = 1.0
        
        aligned_news.append({
            "news_count": news_count,
            "news_text": combined_text,
            "news_sentiment": net_sentiment,
            "news_sentiment_ratio": sentiment_ratio
        })
    
    # Convert to DataFrame and join with prices
    news_features_df = pd.DataFrame(aligned_news, index=prices_df.index)
    result_df = pd.concat([result_df, news_features_df], axis=1)
    
    return result_df