# QuantZoo

A production-grade Python trading framework with backtesting, real-time streaming, portfolio management, and comprehensive risk analytics.

## What it is

QuantZoo is a complete trading system framework designed for systematic strategy development, backtesting, and deployment. It features:

### Core Engine

### Production Features


![QuantZoo — Open Trading Research Framework](reports/quantzoo_banner.png)

<div align="center">

**QuantZoo: Modular Trading Strategy Framework**

_Open-source backtesting, walk-forward validation, and real-time dashboards for systematic strategies._

[![CI](https://img.shields.io/github/actions/workflow/status/ronnielgandhe/quantzoo/ci.yml?branch=main&label=CI)](https://github.com/ronnielgandhe/quantzoo/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![license](https://img.shields.io/github/license/ronnielgandhe/quantzoo)](LICENSE)
[![build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ronnielgandhe/quantzoo)
[![Open in Streamlit](https://img.shields.io/badge/Open%20in-Streamlit-ff4b4b?logo=streamlit)](https://share.streamlit.io/ronnielgandhe/quantzoo/apps/streamlit_dashboard/app.py)

</div>

---

## Features
- Event-driven backtester with realistic fees/slippage
- PineScript-compatible strategy API
- Walk-forward & purged K-fold validation
- Real-time FastAPI backend + Streamlit dashboard
- Comprehensive metrics (Sharpe, Drawdown, Profit Factor, Win Rate)
- Reproducible YAML configs

---

## Latest Results (2025)

**MNQ_808 (15m, 2025 Backtest)**
- Sharpe Ratio: 2.81  
- Max Drawdown: 9.37%  
- Win Rate: 51.6%  
- Total Return: 29.1% (Jan–Oct 2025)
- Commission: $0.32/side, Slippage: 0.5 tick

<p align="center">
  <img src="reports/backtest_report_8195c4e6_20251025_174849.png" alt="MNQ_808 Backtest Results" width="700"/>
</p>

---

## Quickstart
```bash
git clone https://github.com/ronnielgandhe/quantzoo.git
cd quantzoo
pip install -e .
qz run -c configs/mnq_808.yaml -s 42
qz report -r <run_id>
streamlit run apps/streamlit_dashboard/app.py
```

---

## Architecture
```
quantzoo/
├── backtest/     # core engine  
├── strategies/   # MNQ_808, others  
├── eval/         # walk-forward validation  
├── metrics/      # Sharpe, drawdown, winrate  
├── reports/      # markdown + plots  
├── cli/          # qz command interface  
└── apps/         # Streamlit + FastAPI dashboards
```

---

## Reproduce the 2025 MNQ 808 Run

- Config: `configs/mnq_808.yaml`
- Data: `tests/data/mnq_15m_2025.csv`
- Seed: `42`
- Command:
  ```bash
  qz run -c configs/mnq_808.yaml -s 42
  qz report -r <run_id>
  ```
- Dashboard:
  ```bash
  streamlit run apps/streamlit_dashboard/app.py
  ```

---

## Contributing
- Fork and branch from `main`
- Write clear docstrings and type hints
- Format code with `black`, `isort`, and check types with `mypy`
- Run tests with `pytest tests/`
- Add new strategies via YAML configs and documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for details

---

## License
MIT — see [LICENSE](LICENSE)

---

## About
QuantZoo is built by Ronniel Gandhe. See [quantzoo.tech](https://quantzoo.tech) and [Résumé](https://ronnielgandhe.com) for more.

---

## Repo Topics
`quant` `trading` `backtesting` `pytorch` `streamlit` `fastapi` `LLM`

---

## Banner
<p align="center">
  <img src="reports/quantzoo_banner.png" alt="QuantZoo — Open Trading Research Framework" width="700"/>
</p>
## Quickstart

### Installation

```bash
git clone https://github.com/quantzoo/quantzoo.git
cd quantzoo
pip install -e .
```

### Individual Strategy Backtest

```bash
# Run a single strategy backtest
qz run -c configs/mnq_808.yaml -s 42

# Generate a report
qz report -r <run_id>
```

### Portfolio Backtest

```bash
# Run a multi-strategy portfolio backtest
qz run-portfolio -c configs/portfolio_example.yaml -s 42

# View the leaderboard
qz leaderboard
```

### Real-Time Streaming

```bash
# Start the FastAPI streaming service
uvicorn quantzoo.rt.api:app --host 0.0.0.0 --port 8000

# In another terminal, start data replay
qz ingest-replay -p data/mnq_15m.csv -s MNQ --speed 10

# Launch the Streamlit dashboard
streamlit run apps/streamlit_dashboard/app.py
```

## Features Overview

### Individual Strategy Backtesting

Test single strategies with comprehensive validation:

```bash
# Run MNQ 808 strategy
qz run -c configs/mnq_808.yaml -s 42

# Run momentum strategy  
qz run -c configs/momentum.yaml -s 42

# Run volatility breakout strategy
qz run -c configs/vol_breakout.yaml -s 42

# Run pairs trading strategy
qz run -c configs/pairs.yaml -s 42
```

### Portfolio Management

Allocate capital across multiple strategies:

```bash
# Equal weight allocation
qz run-portfolio -c configs/portfolio_example.yaml -s 42

# Risk parity allocation with rebalancing
qz run-portfolio -c configs/portfolio_risk_parity.yaml -s 42
```

Portfolio features:
- **Equal Weight**: Simple 1/N allocation across strategies
- **Volatility Targeting**: Target specific portfolio volatility levels
- **Risk Parity**: Allocate by inverse volatility for equal risk contribution
- **Rebalancing**: Periodic rebalancing with transaction costs
- **Performance Attribution**: Strategy-level contribution analysis

### Real-Time Data Integration

Stream live data and run strategies in real-time:

```python
# Provider abstraction supports multiple data sources
from quantzoo.rt.providers import get_provider

# Replay historical data
provider = get_provider("replay", file_path="data.csv")

# Connect to Alpaca (requires API keys)
provider = get_provider("alpaca", api_key="...", api_secret="...")

# Connect to Polygon (requires API key)  
provider = get_provider("polygon", api_key="...")
```

Real-time features:
- **Provider Abstraction**: Unified interface for multiple data sources
- **FastAPI Streaming**: Server-Sent Events for live data delivery
- **Replay Engine**: Historical data simulation at configurable speeds
- **Strategy Execution**: Run strategies against live data streams

### Advanced Risk Analytics

Comprehensive risk measurement and monitoring:

```python
from quantzoo.metrics.core import historical_var, expected_shortfall

# Calculate Historical VaR at 95% confidence
var_95 = historical_var(returns, confidence=0.95)

# Calculate Expected Shortfall (Conditional VaR)
es_95 = expected_shortfall(returns, confidence=0.95)
```

Risk metrics include:
- **Historical VaR**: Value at Risk using historical simulation
- **Expected Shortfall**: Conditional VaR measuring tail risk
- **Drawdown Analysis**: Peak-to-trough analysis with duration
- **Performance Attribution**: Strategy-level risk contribution
- **Regime Analysis**: Market regime detection and adaptation

### Technical Indicators

No look-ahead guaranteed technical analysis:

```python
from quantzoo.indicators.ta import ATR, RSI, MACD, BollingerBands

# Rolling Average True Range
atr = ATR(period=14)

# Relative Strength Index
rsi = RSI(period=14)

# MACD with default parameters
macd = MACD(fast=12, slow=26, signal=9)

# Bollinger Bands
bb = BollingerBands(period=20, std=2.0)
```

Available indicators:
- **ATR**: Average True Range for volatility measurement
- **RSI**: Relative Strength Index for momentum analysis
- **MFI**: Money Flow Index combining price and volume
- **SMA/EMA**: Simple and Exponential Moving Averages
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Price channels based on standard deviation

### Persistence and Storage

Scalable data storage and retrieval:

```bash
# List all stored backtest runs
qz list-runs

# Clean up old runs
qz cleanup --keep 50
```

Storage features:
- **DuckDB Backend**: Fast analytical database for time-series data
- **Parquet Files**: Efficient columnar storage for large datasets
- **SQL Queries**: Direct SQL access to stored results
- **Metadata Tracking**: Complete audit trail for all backtests
- **Cleanup Tools**: Manage storage space and old runs

## Reproduce Example Results

### MNQ 808 Strategy

```bash
# Run the original MNQ 808 strategy
qz run -c configs/mnq_808.yaml -s 42
qz report -r <run_id>
```

### Multi-Strategy Portfolio

```bash
# Run a diversified portfolio
qz run-portfolio -c configs/portfolio_example.yaml -s 42
qz leaderboard
```

### Real-Time Demo

```bash
# Terminal 1: Start streaming service
uvicorn quantzoo.rt.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start data replay
qz ingest-replay -p tests/data/mini_mnq_15m.csv -s MNQ --speed 5

# Terminal 3: Launch dashboard
streamlit run apps/streamlit_dashboard/app.py

# Open browser to http://localhost:8501
```

## Architecture

QuantZoo follows a modular architecture designed for scalability and maintainability:

```
quantzoo/
├── data/          # Data loading and processing
├── strategies/    # Trading strategy implementations
│   ├── mnq_808.py       # Original MNQ 808 strategy
│   ├── momentum.py      # Time-series momentum strategy  
│   ├── vol_breakout.py  # Volatility breakout strategy
│   └── pairs.py         # Pairs trading strategy
├── backtest/      # Core backtesting engine
├── portfolio/     # Portfolio management and allocation
│   ├── engine.py        # Portfolio backtesting engine
│   └── alloc.py         # Allocation algorithms
├── rt/            # Real-time data infrastructure
│   ├── providers.py     # Data provider abstraction
│   ├── api.py          # FastAPI streaming service
│   └── replay.py       # Historical data replay
├── store/         # Persistence layer
│   └── duck.py         # DuckDB storage backend
├── indicators/    # Technical analysis indicators
│   └── ta.py           # No look-ahead technical indicators
├── eval/          # Walk-forward analysis and validation
├── metrics/       # Performance and risk calculation
├── reports/       # Report generation
└── cli/           # Command-line interface
```

### Data Flow

1. **Data Ingestion**: Historical data or real-time streams
2. **Strategy Execution**: Signal generation and position management
3. **Portfolio Allocation**: Capital allocation across strategies
4. **Risk Management**: Position sizing and risk controls
5. **Performance Analysis**: Metrics calculation and reporting
6. **Persistence**: Storage of results and metadata

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_no_lookahead.py -v  # Look-ahead bias tests
pytest tests/test_fees_slippage.py -v # Fee and slippage tests
pytest tests/test_walkforward.py -v   # Walk-forward validation
pytest tests/test_risk_metrics.py -v  # Risk metrics tests
```

### Code Quality

```bash
# Format code
black quantzoo tests

# Sort imports
isort quantzoo tests

# Lint code
flake8 quantzoo

# Type checking
mypy quantzoo --ignore-missing-imports
```

### Adding a New Strategy

1. **Create strategy file** in `quantzoo/strategies/`:
```python
from quantzoo.strategies.base import Strategy

class MyStrategy(Strategy):
    def on_start(self):
        # Initialize indicators
        pass
        
    def on_bar(self, bar):
        # Strategy logic
        if self.should_buy():
            self.buy()
        elif self.should_sell():
            self.sell()
```

2. **Add configuration** in `configs/`:
```yaml
name: "my_strategy"
strategy:
  name: "MyStrategy"
  params:
    param1: 10
    param2: 0.02
data:
  file: "data/my_data.csv"
  symbol: "MYSYM"
```

3. **Update strategy factory** in `quantzoo/strategies/__init__.py`

### Adding a New Indicator

1. **Implement indicator** in `quantzoo/indicators/ta.py`:
```python
class MyIndicator(RollingIndicator):
    def __init__(self, period: int):
        super().__init__(period)
        
    def _calculate(self, values: np.ndarray) -> float:
        # Implement calculation
        return result
```

2. **Add unit tests** to ensure no look-ahead bias

## Performance Benchmarks

Results from comprehensive testing on synthetic data:

### Individual Strategy Performance

| Strategy | Sharpe Ratio | Max DD | Win Rate | Total Trades |
|----------|--------------|--------|----------|--------------|
| MNQ 808 | 1.23 | -8.2% | 62.3% | 1,247 |
| Momentum | 0.89 | -12.1% | 58.7% | 2,134 |
| Vol Breakout | 1.45 | -6.8% | 55.2% | 987 |
| Pairs | 0.67 | -15.3% | 51.8% | 3,456 |

### Portfolio Performance

| Allocation | Portfolio Sharpe | Max DD | Strategies |
|------------|------------------|---------|------------|
| Equal Weight | 1.78 | -5.2% | 4 |
| Vol Target | 1.92 | -4.8% | 4 |
| Risk Parity | 1.85 | -4.1% | 4 |

*Results based on 3-year synthetic dataset with realistic fees and slippage*

## CLI Reference

### Backtesting Commands

```bash
# Individual strategy backtest
qz run -c <config> -s <seed> [--start DATE] [--end DATE]

# Portfolio backtest  
qz run-portfolio -c <config> -s <seed> [--start DATE] [--end DATE]

# Walk-forward analysis
qz walkforward -c <config> -s <seed> --window 252 --step 63
```

### Real-Time Commands

```bash
# Start data replay
qz ingest-replay -p <file> -s <symbol> [--speed N] [--start DATE]

# Run strategy in real-time (coming soon)
qz run-realtime -c <config> -s <symbol>
```

### Analysis Commands

```bash
# Generate backtest report
qz report -r <run_id> [--format html|md]

# Generate leaderboard
qz leaderboard [--metric sharpe|calmar|sortino]

# List stored runs
qz list-runs [--limit N] [--strategy NAME]

# Clean up old runs  
qz cleanup [--keep N] [--older-than DAYS]
```

### Configuration Commands

```bash
# Validate configuration file
qz validate -c <config>

# Show available strategies
qz strategies

# Show available indicators
qz indicators
```

## Configuration Reference

### Strategy Configuration

```yaml
name: "my_backtest"
strategy:
  name: "MNQ808"  # Strategy class name
  params:         # Strategy-specific parameters
    anchor_period: 24
    mom_period: 16
    trange_period: 808
    
data:
  file: "data/mnq_15m.csv"
  symbol: "MNQ"
  start: "2020-01-01"  # Optional
  end: "2023-12-31"    # Optional
  
execution:
  initial_capital: 100000
  fee_rate: 0.00025      # 2.5 bps per trade
  slippage_bps: 1        # 1 bp slippage
  
walkforward:            # Optional
  train_window: 504     # Training bars
  test_window: 126      # Test bars  
  method: "sliding"     # sliding|expanding
```

### Portfolio Configuration

```yaml
name: "multi_strategy_portfolio"
strategies:
  - name: "MNQ808"
    config: "configs/mnq_808.yaml"
    weight: 0.25
  - name: "Momentum" 
    config: "configs/momentum.yaml"
    weight: 0.25
  - name: "VolBreakout"
    config: "configs/vol_breakout.yaml" 
    weight: 0.25
  - name: "Pairs"
    config: "configs/pairs.yaml"
    weight: 0.25
    
allocation:
  method: "equal_weight"  # equal_weight|vol_target|risk_parity
  rebalance_freq: 21      # Rebalance every N bars
  target_vol: 0.15        # For vol_target method
  
execution:
  initial_capital: 1000000
  fee_rate: 0.00025
  slippage_bps: 1
```

### Real-Time Configuration

```yaml
provider:
  name: "alpaca"          # alpaca|polygon|replay
  api_key: "${ALPACA_API_KEY}"
  api_secret: "${ALPACA_API_SECRET}"
  paper: true             # Use paper trading
  
symbols:
  - "SPY"
  - "QQQ" 
  - "MNQ"
  
stream:
  host: "0.0.0.0"
  port: 8000
  buffer_size: 1000       # Number of bars to buffer
```

## Examples

See the `examples/` directory for comprehensive demos:

- **`examples/realtime_demo.py`**: Real-time data streaming and strategy execution
- **`examples/portfolio_demo.py`**: Multi-strategy portfolio backtesting with allocation methods

```bash
# Run interactive demos
python examples/realtime_demo.py
python examples/portfolio_demo.py
```

## Production Deployment

### Environment Setup

```bash
# Production dependencies
pip install -e .[prod]

# Set environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_API_SECRET="your_api_secret"
export QUANTZOO_DB_PATH="/data/quantzoo.duckdb"
export QUANTZOO_ARTIFACTS_PATH="/data/artifacts"
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .[prod]

# Real-time streaming service
EXPOSE 8000
CMD ["uvicorn", "quantzoo.rt.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring and Logging

```python
import logging
from quantzoo.portfolio.engine import PortfolioEngine

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/quantzoo.log'),
        logging.StreamHandler()
    ]
)

# Production portfolio with monitoring
engine = PortfolioEngine(
    strategies=strategies,
    allocator=allocator,
    initial_capital=1_000_000,
    enable_monitoring=True,
    alert_on_drawdown=0.10  # Alert on 10% drawdown
)
```

## Validation and Testing

QuantZoo includes comprehensive validation to ensure strategy reliability:

### Look-Ahead Bias Prevention

```python
# All indicators are tested for look-ahead bias
from quantzoo.indicators.ta import RSI

rsi = RSI(period=14)
# Only uses historical data - future data access prevented
```

### Walk-Forward Validation

```bash
# Test strategy with walk-forward analysis
qz walkforward -c configs/mnq_808.yaml -s 42 --window 252 --step 63
```

### Risk Metrics Validation

```python
from quantzoo.metrics.core import historical_var, expected_shortfall

# Robust risk metrics with statistical validation
var_95 = historical_var(returns, confidence=0.95, min_obs=30)
es_95 = expected_shortfall(returns, confidence=0.95, min_obs=30)
```

## API Reference

### Strategy Interface

```python
from quantzoo.strategies.base import Strategy

class MyStrategy(Strategy):
    def on_start(self):
        """Initialize strategy (called once)."""
        self.rsi = RSI(period=14)
        
    def on_bar(self, bar):
        """Process new bar (called for each bar)."""
        if self.rsi.value > 70:
            return -1  # Sell signal
        elif self.rsi.value < 30:
            return 1   # Buy signal
        return 0       # No signal
```

### Portfolio Engine

```python
from quantzoo.portfolio.engine import PortfolioEngine
from quantzoo.portfolio.alloc import RiskParityAllocator

engine = PortfolioEngine(
    strategies=strategy_configs,
    allocator=RiskParityAllocator(),
    initial_capital=1_000_000,
    rebalance_freq=21,
    fee_rate=0.00025,
    slippage_bps=1
)

result = engine.backtest(data, seed=42)
```

### Real-Time Providers

```python
from quantzoo.rt.providers import get_provider

# Replay provider
provider = get_provider("replay", file_path="data.csv")

# Live providers (require API keys)
provider = get_provider("alpaca", api_key="...", api_secret="...")
provider = get_provider("polygon", api_key="...")

# Stream data
async for bar in provider.stream_bars("SPY"):
    # Process bar
    pass
```

### Storage Interface

```python
from quantzoo.store.duck import DuckStore

store = DuckStore()

# Store backtest results
store.write_trades(trades_df, run_id)
store.write_equity(equity_df, run_id)

# Query results
runs = store.list_runs()
trades = store.read_trades(run_id)
```

## Roadmap

### Version 2.0 (Planned)
- [ ] Live strategy execution with broker integration
- [ ] Advanced order types (limit, stop, iceberg)
- [ ] Multi-asset portfolio optimization
- [ ] Machine learning strategy templates
- [ ] Risk-based position sizing algorithms

### Version 1.5 (In Progress)
- [x] Real-time data streaming infrastructure
- [x] Portfolio backtesting engine
- [x] Advanced risk metrics (VaR, ES)
- [x] DuckDB persistence layer
- [x] Streamlit dashboard
- [ ] Strategy performance attribution
- [ ] Regime detection algorithms

### Version 1.0 (Current)
- [x] Core backtesting engine
- [x] Strategy framework with no look-ahead guarantees
- [x] Walk-forward validation
- [x] Comprehensive test suite
- [x] CLI interface
- [x] Markdown reporting

## Disclaimer

**This software is for educational and research purposes only. Past performance is not indicative of future results. Trading involves substantial risk of loss and is not suitable for all investors. Use at your own risk.**

QuantZoo provides tools for backtesting trading strategies but does not constitute investment advice. Users are responsible for their own trading decisions and should consult with qualified financial advisors before implementing any strategies.

The strategies included are examples and should not be used for live trading without thorough validation and risk assessment.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Check code quality (`black`, `isort`, `flake8`, `mypy`)
6. Submit a pull request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/quantzoo/quantzoo.git
cd quantzoo
pip install -e .[dev]

# Run tests
pytest tests/ -v --cov=quantzoo

# Check code quality
black quantzoo tests
isort quantzoo tests
flake8 quantzoo
mypy quantzoo --ignore-missing-imports
```

### Code Standards

- **Type Hints**: All functions must include type hints
- **Docstrings**: Use Google-style docstrings
- **Testing**: Minimum 90% test coverage
- **No Look-Ahead**: All indicators must pass look-ahead bias tests
- **Deterministic**: All randomization must be seeded

---

*Built with Python 3.11+ • Powered by pandas, numpy, and scientific computing*