# QuantZoo Hugging Face Spaces Deployment

This directory contains the setup for deploying QuantZoo as a Hugging Face Spaces application.

## Deployment Instructions

### Option 1: Streamlit on Spaces

1. **Create a new Space on Hugging Face:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Set space name (e.g., "quantzoo-demo")

2. **Upload files:**
   - Copy `../streamlit_app/app.py` to your Space repository
   - Copy `requirements.txt` to your Space repository
   - Copy the `quantzoo/` package directory to your Space

3. **Configuration:**
   - Ensure `requirements.txt` includes all dependencies
   - Set Python version to 3.8+ in Space settings
   - Configure secrets if needed (not required for this demo)

### Option 2: Gradio Interface

For a Gradio-based interface, create an `app.py` with:

```python
import gradio as gr
import pandas as pd
from quantzoo import run_backtest_simple

def backtest_interface(price_file, strategy, atr_mult, lookback):
    # Process uploaded file and run backtest
    results = run_backtest_simple(price_file, strategy, atr_mult, lookback)
    return results['metrics'], results['chart']

iface = gr.Interface(
    fn=backtest_interface,
    inputs=[
        gr.File(label="Price Data CSV"),
        gr.Dropdown(["MNQ 808", "Regime Hybrid"], label="Strategy"),
        gr.Number(value=1.5, label="ATR Multiplier"),
        gr.Number(value=10, label="Lookback Period")
    ],
    outputs=[
        gr.JSON(label="Metrics"),
        gr.Plot(label="Equity Curve")
    ],
    title="QuantZoo Strategy Backtesting",
    description="Upload price data and backtest systematic trading strategies"
)

iface.launch()
```

## Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.28.0
gradio>=3.40.0
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.1.0
pydantic>=1.10.0
```

## File Structure for Spaces

```
your-space-repo/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Space description
└── quantzoo/             # QuantZoo package
    ├── __init__.py
    ├── strategies/
    ├── backtest/
    └── ...
```

## Environment Variables

No special environment variables required for the basic demo.

## Notes

- Spaces have computational limits, so optimize for smaller datasets
- Consider caching results to improve performance
- Include sample data files for users to test
- Add clear instructions in the Space README

## Sample Data

Include sample CSV files in your Space:
- `sample_prices.csv` - Example OHLCV data
- `sample_news.csv` - Example news headlines

## Performance Optimization

For Spaces deployment:
- Limit data size to < 1000 bars for faster processing
- Use `@st.cache_data` decorators in Streamlit
- Optimize model training frequency
- Consider pre-trained models for news processing