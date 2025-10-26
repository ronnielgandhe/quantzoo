# Data Format Guide

This document explains the expected CSV schema for QuantZoo and how to bring your own 2025 MNQ data.

## CSV Schema

QuantZoo expects CSV files with the following columns:

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Bar timestamp in ISO-8601 format or parseable datetime |
| `open` | float | Opening price for the bar |
| `high` | float | Highest price during the bar |
| `low` | float | Lowest price during the bar |
| `close` | float | Closing price for the bar |
| `volume` | int | Trading volume for the bar |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Instrument symbol (e.g., "MNQ") |
| `datetime` | datetime | Alternative to `timestamp` |

### Example CSV Format

```csv
timestamp,open,high,low,close,volume,symbol
2025-01-01 08:00:00,4238.50,4242.25,4236.75,4240.00,1250,MNQ
2025-01-01 08:15:00,4240.00,4245.50,4238.25,4244.75,1180,MNQ
2025-01-01 08:30:00,4244.75,4248.00,4243.50,4246.25,1320,MNQ
```

## 2025 Data Defaults

The dashboard and API are configured with 2025 defaults:

- **Start Date**: `2025-01-01`
- **End Date**: Current date (auto-updated)
- **Timeframe**: `15m` (15-minute bars)
- **Timezone**: `America/Toronto`
- **Default File**: `tests/data/mnq_15m_2025.csv`

## Bringing Your Own MNQ Data

### 1. Data Requirements

For Mini NASDAQ-100 (MNQ) futures:

- **Minimum timeframe**: 15-minute bars
- **Date range**: January 1, 2025 onwards
- **Trading hours**: Generally 6:00 PM - 5:00 PM ET (Sunday-Friday)
- **Contract**: Use front month or continuous contract
- **Currency**: USD

### 2. Data Sources

Popular sources for MNQ data:

- **Interactive Brokers** (TWS/API)
- **TradingView** (export feature)
- **Quandl/Alpha Vantage** (API)
- **Yahoo Finance** (limited intraday history)
- **IEX Cloud** (futures data)

### 3. File Placement

Place your CSV file in one of these locations:

```
quantzoo/
├── tests/data/mnq_15m_2025.csv     # Default location
├── data/your_mnq_data.csv          # Custom location
└── configs/custom_config.yaml      # Point to your file
```

### 4. Configuration

Update your config file to point to your data:

```yaml
data:
  path: "data/your_mnq_data.csv"
  timeframe: "15m"
  start_date: "2025-01-01"
  end_date: null
  timezone: "America/Toronto"
```

## Data Validation

The replay engine validates:

1. **Symbol alignment**: CSV symbol matches requested symbol
2. **Date range**: Data exists within requested start/end dates
3. **Required columns**: All OHLCV columns are present
4. **Chronological order**: Data is sorted by timestamp

### Error Messages

- `"Requested symbols not found in CSV"`: Symbol mismatch
- `"No data found in date range"`: Date filtering removed all data
- `"Missing required columns"`: OHLCV columns missing

## Timezone Handling

All timestamps are converted to the configured timezone:

- **Input**: Any timezone or timezone-naive
- **Processing**: Converted to `America/Toronto` (default)
- **Output**: ISO-8601 strings with timezone info
- **Charts**: X-axis labeled in target timezone

## Performance Tips

1. **Pre-filter data**: Only include needed date ranges in CSV
2. **Optimize timeframe**: Use native bar interval (avoid resampling)
3. **Clean data**: Remove gaps, holidays, and invalid bars
4. **File size**: Large files (>100MB) may impact startup time

## Troubleshooting

### Common Issues

**Empty chart/no data**:
- Check date range in config vs. CSV data
- Verify timestamp format is parseable
- Ensure timezone conversion is correct

**Symbol errors**:
- Match symbol exactly (case-sensitive)
- Remove spaces or special characters
- Use consistent naming (MNQ vs NQH25)

**Performance issues**:
- Reduce date range for testing
- Use smaller files during development
- Increase replay speed for faster testing

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed data loading and filtering steps.