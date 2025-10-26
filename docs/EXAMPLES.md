# EXAMPLES.md

## MNQ_808 Example Walkthrough

1. **Config**: `configs/mnq_808.yaml`
2. **Data**: `tests/data/mnq_15m_2025.csv`
3. **Seed**: `42`
4. **Run Backtest**:
   ```bash
   qz run -c configs/mnq_808.yaml -s 42
   qz report -r <run_id>
   ```
5. **View Results**:
   - Sharpe Ratio: 2.81
   - Max Drawdown: 9.37%
   - Win Rate: 51.6%
   - Total Return: 29.1%
   - Commission: $0.32/side, Slippage: 0.5 tick
6. **Visualize**:
   ```bash
   streamlit run apps/streamlit_dashboard/app.py
   ```
7. **Interpretation**:
   - Consistent outperformance with controlled drawdown.
   - Robust to slippage and commission assumptions.
