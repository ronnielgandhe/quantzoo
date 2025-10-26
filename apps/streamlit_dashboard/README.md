# QuantZoo Streamlit Dashboard

This is the real-time dashboard for monitoring QuantZoo trading strategies.

## Features

- **Real-time Data**: Live streaming of market data and strategy metrics
- **Interactive Charts**: Equity curves, price charts, and performance visualization
- **Position Monitoring**: Track open positions and portfolio allocation
- **Trade History**: View recent trades and execution details
- **Strategy Controls**: Start/stop strategies and adjust parameters

## Quick Start

1. **Start the FastAPI service** (required for live data):
   ```bash
   uvicorn quantzoo.rt.api:app --reload
   ```

2. **Run the dashboard**:
   ```bash
   streamlit run apps/streamlit_dashboard/app.py
   ```

3. **Configure and run**:
   - Select a strategy configuration file
   - Choose symbols to monitor
   - Set replay speed for backtesting
   - Click "Start" to begin live monitoring

## Requirements

```
streamlit>=1.20.0
plotly>=5.0.0
requests>=2.25.0
pandas>=1.3.0
numpy>=1.20.0
```

## API Integration

The dashboard connects to the QuantZoo FastAPI service running on `localhost:8000`. Ensure the API service is running before starting the dashboard.

Key endpoints used:
- `GET /healthz` - Service health check
- `GET /metrics/latest` - Latest portfolio metrics
- `POST /replay/start` - Start strategy replay
- `POST /replay/stop` - Stop strategy replay
- `GET /replay/status` - Get replay status

## Usage Tips

- Use faster replay speeds (2x-10x) for quick backtesting
- Monitor the equity curve for real-time performance
- Check positions tab for current portfolio allocation
- Use trades tab to analyze execution quality

## Troubleshooting

**Dashboard shows "FastAPI Service Offline":**
- Ensure the API service is running: `uvicorn quantzoo.rt.api:app --reload`
- Check that port 8000 is not in use by another service

**No data appearing:**
- Verify the CSV path exists and contains valid OHLCV data
- Check that symbols match the data in your CSV file
- Ensure replay is started via the "Start" button