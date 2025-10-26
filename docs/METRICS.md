# METRICS.md

## Metric Definitions

| Metric         | Formula / Description                                  |
|----------------|-------------------------------------------------------|
| Sharpe Ratio   | $(\frac{\text{mean}(R_p - R_f)}{\text{std}(R_p - R_f)})$ |
| Max Drawdown   | Largest peak-to-trough decline                        |
| Win Rate       | $\frac{\text{Winning Trades}}{\text{Total Trades}}$    |
| Profit Factor  | $\frac{\text{Gross Profit}}{\text{Gross Loss}}$        |
| Total Return   | $\frac{\text{Final Balance} - \text{Start}}{\text{Start}}$ |
| Commission     | Per contract, per side (entry/exit)                   |
| Slippage       | Simulated tick impact per trade                       |

## Example Calculation
```
Sharpe Ratio = (mean(strategy returns) - risk-free rate) / std(strategy returns)
```
