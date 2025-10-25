# Regime Hybrid Strategy

## What

News+price hybrid regime detection model that combines sentiment analysis of news headlines with price-based technical indicators. Classifies market regime as risk-on or risk-off and trades accordingly with overlay strategy.

## Data

**Requirements:**
- 15-minute OHLCV price data
- News data with headlines and timestamps
- Minimum 500 bars for training

**Price Format:**
```csv
time,open,high,low,close,volume
2023-01-01 08:00:00,4200.50,4205.25,4198.75,4203.00,1500
```

**News Format:**
```csv
timestamp,headline
2023-01-01 08:05:00,"Fed signals dovish stance on rates"
2023-01-01 08:12:00,"Tech earnings beat expectations"
```

## Method

**Text Processing:**
- Default: TF-IDF vectorization with logistic regression
- Optional: Hugging Face transformers with embedding cache
- Fallback path ensures robustness without external dependencies

**Price Features:**
- Returns (1, 5, 15 bar lookbacks)
- Z-score of returns
- Average True Range (ATR)
- Momentum indicators

**Regime Classification:**
- Combine text sentiment with price features
- Binary classification: risk-on (1) vs risk-off (0)
- Logistic regression or neural network classifier

**Trading Overlay:**
- Long positions during risk-on regime
- Flat or short positions during risk-off regime
- Same risk management as MNQ 808 (stops, trailing)

## Config

**Files:**
- `configs/regime_hybrid_conservative.yaml` - Conservative parameters
- `configs/regime_hybrid_tfidf.yaml` - TF-IDF only mode

**Key Parameters:**
- `text_mode: "tfidf"` - Text processing method
- `lookback: 20` - Historical window for features
- `news_window: "30min"` - News aggregation window
- `clf: "logreg"` - Classifier type
- `price_features: ["returns", "zscore", "atr"]`

## Metrics

**Expected Performance:**
- Sharpe Ratio: 1.0-1.6
- Maximum Drawdown: <6%
- Information Ratio: >0.8
- Regime Accuracy: >60%

## Limitations

- News quality dependent on data source
- Text processing may miss context/sarcasm
- Regime changes lag actual market shifts
- Requires aligned news and price timestamps
- Performance varies with news coverage density
- TF-IDF approach limited to simple sentiment

## Reproduce

```bash
# Create sample news data
python3 create_news_data.py

# Run conservative config
qz run -c configs/regime_hybrid_conservative.yaml -s 42

# Generate report with regime predictions
qz report -r <run_id>
```

**Expected Output:**
- Regime predictions in artifacts/predictions.csv
- Feature importance analysis in report
- Sharpe ratio >1.0 on conservative settings