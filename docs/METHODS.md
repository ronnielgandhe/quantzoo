# METHODS.md

## Methodology

QuantZoo implements robust validation and backtesting techniques:

- **Walk-forward analysis**: Expanding and sliding window validation for realistic out-of-sample testing.
- **Purged K-fold splits**: Prevents look-ahead bias by purging overlapping samples.
- **No look-ahead bias**: All metrics and signals are computed strictly on past data.
- **Deterministic seeds**: Ensures reproducibility for all runs.

## Validation Example
```
train: Jan–Jun 2025
validate: Jul–Oct 2025
purge: 1 bar between splits
```

## References
- Marcos López de Prado, "Advances in Financial Machine Learning"
- QuantZoo documentation
