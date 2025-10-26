# QuantZoo Examples

This directory contains comprehensive examples demonstrating QuantZoo's features.

## Available Examples

### realtime_demo.py
Demonstrates real-time data streaming and strategy execution:
- Historical data replay at configurable speeds
- Live data streaming with multiple providers
- Real-time strategy signal generation
- Position management and logging

**Usage:**
```bash
python examples/realtime_demo.py
```

### portfolio_demo.py  
Demonstrates portfolio backtesting with multiple allocation methods:
- Multi-strategy portfolio construction
- Equal weight, volatility targeting, and risk parity allocation
- Rebalancing mechanics and transaction costs
- Performance comparison across methods

**Usage:**
```bash
python examples/portfolio_demo.py
```

## Running the Examples

### Prerequisites
Make sure QuantZoo is installed in development mode:
```bash
pip install -e .
```

### Real-Time Demo
```bash
# Interactive demo with menu
python examples/realtime_demo.py

# Or run directly:
# Replay historical data
python -c "
import asyncio
from examples.realtime_demo import RealTimeDemo
demo = RealTimeDemo('MNQ')
asyncio.run(demo.run_replay_demo('tests/data/mini_mnq_15m.csv', speed=10))
"
```

### Portfolio Demo
```bash
# Full portfolio comparison
python examples/portfolio_demo.py

# Or import and customize:
python -c "
from examples.portfolio_demo import run_portfolio_demo
results = run_portfolio_demo()
print('Best Sharpe:', max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio']))
"
```

## Example Outputs

### Real-Time Demo Output
```
QuantZoo Real-Time Demo
======================

Starting replay demo for MNQ
Data file: tests/data/mini_mnq_15m.csv, Speed: 10x
BUY at 15234.50 | Time: 2023-01-03 09:45:00
Bar 100: Price=15245.25, Position=1
SELL at 15267.75 | Time: 2023-01-03 11:30:00
EXIT at 15255.00 | Time: 2023-01-03 12:15:00
```

### Portfolio Demo Output
```
==================================================
Running Equal Weight Portfolio
==================================================
Total Return: 12.45%
Sharpe Ratio: 1.234
Max Drawdown: -5.67%
Win Rate: 58.3%
Total Trades: 1,247

Strategy Allocations:
  MNQ808: 33.3%
  Momentum: 33.3%
  VolBreakout: 33.3%

============================================================
Portfolio Comparison
============================================================
                Total Return  Sharpe Ratio  Max Drawdown  Win Rate  Total Trades
Equal Weight         0.124         1.234        -0.057     0.583          1247
Vol Target (15%)     0.138         1.456        -0.045     0.591          1156  
Risk Parity          0.131         1.387        -0.039     0.578          1203
```

## Customization

### Adding Your Own Strategy
```python
# In portfolio_demo.py, add to strategies list:
strategies.append({
    'name': 'MyStrategy',
    'class': my_strategy.MyStrategy,
    'config': load_strategy_config('configs/my_strategy.yaml')
})
```

### Custom Allocation Method
```python
from quantzoo.portfolio.alloc import BaseAllocator

class MyAllocator(BaseAllocator):
    def allocate(self, returns, **kwargs):
        # Your allocation logic
        return weights

# Use in portfolio
allocator = MyAllocator()
engine = PortfolioEngine(strategies=strategies, allocator=allocator, ...)
```

### Real-Time Provider Configuration
```python
# For live data, set environment variables:
import os
os.environ['ALPACA_API_KEY'] = 'your_key'
os.environ['ALPACA_API_SECRET'] = 'your_secret'

# Then use live provider
await demo.run_live_demo('alpaca')
```

## Integration with CLI

These examples complement the CLI interface:

```bash
# Run the same portfolio via CLI
qz run-portfolio -c configs/portfolio_example.yaml -s 42

# Compare with example results
qz leaderboard

# Generate detailed report
qz report -r <run_id>
```

## Advanced Usage

### Streaming Integration
```python
# Combine with FastAPI streaming service
# Terminal 1: Start API
uvicorn quantzoo.rt.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start data replay
qz ingest-replay -p tests/data/mini_mnq_15m.csv -s MNQ --speed 5

# Terminal 3: Run real-time demo
python examples/realtime_demo.py
```

### Dashboard Integration
```python
# Launch Streamlit dashboard to visualize results
streamlit run apps/streamlit_dashboard/app.py

# Navigate to http://localhost:8501 to see real-time charts
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Make sure QuantZoo is installed in development mode
   pip install -e .
   ```

2. **Missing Data Files**
   ```bash
   # Generate test data if missing
   python create_test_data.py
   ```

3. **API Key Errors**
   ```bash
   # Set environment variables for live data
   export ALPACA_API_KEY="your_key"
   export ALPACA_API_SECRET="your_secret"
   ```

4. **Missing Dependencies**
   ```bash
   # Install optional dependencies
   pip install matplotlib streamlit
   ```

For more help, see the main [README.md](../README.md) and [Integration Guide](../INTEGRATION_GUIDE.md).