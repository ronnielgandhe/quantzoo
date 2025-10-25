# QuantZoo

A lean, credible Python trading strategy backtesting framework with comprehensive validation and no look-ahead bias.

## What it is

QuantZoo is a modular backtesting framework designed for systematic trading strategy development and validation. It features:

- **Modular Strategy System**: Easy-to-implement strategy interface with Pine Script compatibility
- **Realistic Execution**: Configurable fees, slippage, and market impact modeling
- **Walk-Forward Analysis**: Sliding and expanding window validation with purged K-fold cross-validation
- **No Look-Ahead Bias**: Comprehensive testing to prevent future data leakage
- **Comprehensive Metrics**: Sharpe ratio, maximum drawdown, win rate, profit factor, and more
- **CLI Interface**: Simple command-line tools for running backtests and generating reports
- **Deterministic Results**: Reproducible backtests with seed-based randomization

## Quickstart

### Installation

```bash
git clone https://github.com/quantzoo/quantzoo.git
cd quantzoo
pip install -e .
```

### Run a Backtest

```bash
qz run -c configs/mnq_808.yaml -s 42
```

### Generate a Report

```bash
qz report -r <run_id>
```

The run ID is displayed after running a backtest and can also be found in `artifacts/metrics.json`.

## Reproduce

To reproduce the MNQ 808 strategy results:

1. **Clone and install** the repository as shown above
2. **Run the backtest** with the provided configuration:
   ```bash
   qz run -c configs/mnq_808.yaml -s 42
   ```
3. **Generate the report** using the returned run ID:
   ```bash
   qz report -r <run_id>
   ```

The backtest will execute on the provided synthetic MNQ 15-minute data and generate:
- Performance metrics in `artifacts/`
- Markdown report with equity curve in `reports/`

## Methodology

### Strategy Implementation

The MNQ 808 strategy is a precise port of the Pine Script v6 code, implementing:

- **Indicator Calculation**: SMA of True Range, ATR, MFI/RSI momentum
- **Signal Generation**: Anchor-based crossover system with session filtering
- **Risk Management**: One-bar adverse exit rule and legacy tick-based stops
- **Position Sizing**: Fixed contract quantity with configurable parameters

### Validation Framework

QuantZoo implements multiple layers of validation:

1. **Look-Ahead Prevention**: Indicators only access historical data
2. **Walk-Forward Analysis**: Time-series aware train/test splits
3. **Purged K-Fold**: Embargo periods prevent label leakage
4. **Realistic Execution**: Fees and slippage applied to all trades
5. **Deterministic Testing**: Seed-based randomization for reproducibility

### Performance Metrics

All metrics are calculated with proper annualization and market assumptions:

- **Sharpe Ratio**: Annualized using √6048 for 15-minute bars (252 trading days × 24 bars/day)
- **Maximum Drawdown**: Peak-to-trough equity decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Exposure**: Percentage of time in position

## What is Live vs WIP

### ✅ Live (Working)

- Core backtesting engine with fees and slippage
- MNQ 808 strategy implementation
- Walk-forward analysis with sliding/expanding windows
- Comprehensive test suite with look-ahead validation
- CLI interface for running backtests and generating reports
- Markdown report generation with equity curves
- CI/CD pipeline with automated testing

### 🚧 Work in Progress

- Additional strategy templates and indicators
- Real-time data integration
- Portfolio-level backtesting
- Advanced risk metrics (VaR, Expected Shortfall)
- Web-based dashboard
- Database persistence for large-scale analysis

## Results

Results from the MNQ 808 strategy on synthetic data (seed=42):

| Metric | Value |
|--------|-------|
| Total Return | TBD% |
| Sharpe Ratio | TBD |
| Maximum Drawdown | TBD% |
| Win Rate | TBD% |
| Total Trades | TBD |

*Results will be populated after running the acceptance tests*

## Architecture

```
quantzoo/
├── data/          # Data loading and processing
├── strategies/    # Trading strategy implementations
├── backtest/      # Core backtesting engine
├── eval/          # Walk-forward analysis and validation
├── metrics/       # Performance calculation utilities
├── reports/       # Report generation
└── cli/           # Command-line interface
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black quantzoo tests
isort quantzoo tests
flake8 quantzoo
mypy quantzoo
```

### Adding a New Strategy

1. Create a new file in `quantzoo/strategies/`
2. Implement the `Strategy` protocol with `on_start()` and `on_bar()` methods
3. Add configuration in `configs/`
4. Update CLI to recognize the new strategy

## Disclaimer

**This software is for educational and research purposes only. Past performance is not indicative of future results. Trading involves substantial risk of loss and is not suitable for all investors. Use at your own risk.**

QuantZoo provides tools for backtesting trading strategies but does not constitute investment advice. Users are responsible for their own trading decisions and should consult with qualified financial advisors before implementing any strategies.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code is properly formatted
5. Submit a pull request

---

*Built with Python 3.11+ • Powered by pandas, numpy, and scientific computing*