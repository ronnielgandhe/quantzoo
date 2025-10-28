# README Update Additions

Add these sections to the main README.md:

## 🧠 Machine Learning & Deep Learning

QuantZoo now supports hybrid transformer models combining news sentiment and price patterns.

### Quick Start: Train a Model

```bash
# Generate synthetic sample data
python scripts/generate_sample_data.py

# Train hybrid transformer model
python ml/train_transformer.py --config configs/example_transformer.yaml

# Check generated artifacts
ls -la artifacts/models/news_price_hybrid_v1/
# - model.pt (weights)
# - checkpoint.pt (full checkpoint)
# - metrics.json (evaluation metrics)
# - model_card.json (metadata)
# - model_card.md (documentation)
```

### Model Cards

Every trained model generates a comprehensive model card with:
- Dataset provenance
- Training metrics and evaluation
- Intended use and limitations
- Explainability summary
- Reproducibility information

See `docs/model_card_template.md` for the full template.

---

## 📊 Live Trading & Broker Integration

**⚠️ WARNING**: Live trading can place real orders with real money. Always test in paper mode first.

### Paper Trading (Safe)

```python
from connectors.brokers import PaperBroker, Order, OrderSide, OrderType

# Initialize paper broker
broker = PaperBroker({
    'initial_cash': 100000,
    'slippage_bps': 5,
    'commission_per_share': 0.005,
    'dry_run': True
})

# Update market price
broker.update_market_price('AAPL', 150.0)

# Place order
order = Order(
    symbol='AAPL',
    side=OrderSide.BUY,
    quantity=10,
    order_type=OrderType.MARKET
)

order_id = broker.place_order(order)

# Check position
positions = broker.get_positions()
balance = broker.get_account_balance()
```

### Supported Brokers

- **Paper Broker**: Simulated trading (safe for testing)
- **Alpaca**: Paper and live trading
- **Interactive Brokers**: Paper and live trading via IB Gateway

### Safety Features

Before ANY live trading:

1. **Kill Switch**: Emergency stop all trading
   ```bash
   curl -X POST http://localhost:8888/kill-switch/activate \
     -H "Authorization: Bearer $SAFETY_API_TOKEN" \
     -d '{"reason": "Emergency stop", "close_positions": true, "operator": "Admin"}'
   ```

2. **Environment Check**: `QUANTZOO_ENV=production` required for live

3. **Dry-Run Default**: All connectors default to paper mode

4. **Two-Step Confirmation**: Live orders require explicit approval

See `ops/RUNBOOK.md` for complete safety procedures.

---

## 🔧 Monitoring & Operations

### Start Monitoring Stack

```bash
# Start Prometheus + Grafana
docker-compose -f ops/docker-compose.monitor.yml up -d

# Access dashboards
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana (admin/admin)
```

### Health Checks

```bash
# API health
curl http://localhost:8001/healthz

# Safety API health
curl http://localhost:8888/health

# Current safety status
curl http://localhost:8888/status
```

### Emergency Procedures

See `ops/RUNBOOK.md` for:
- Kill switch activation
- Emergency position closure
- Common failure modes
- Credential rotation
- System restart procedures

---

## 🐳 Docker Deployment

### Build Production Image

```bash
docker build -f docker/Dockerfile.inference -t quantzoo:latest .
```

### Run Container

```bash
docker run -d \
  --name quantzoo-api \
  -p 8000:8000 \
  -e QUANTZOO_ENV=development \
  -e SAFETY_API_TOKEN=your_token_here \
  quantzoo:latest
```

---

## 🧪 Testing

### Run Full Test Suite

```bash
# All tests
pytest -v

# With coverage
pytest --cov=quantzoo --cov-report=html

# Specific test suites
pytest tests/test_brokers.py -v
pytest tests/test_ml_no_lookahead.py -v
```

### Regression Tests

```bash
# Ensure no breaking changes
pytest tests/test_walkforward.py -v
```

---

## 📈 Leaderboard & Model Hub

### Generate Leaderboard

```bash
python tools/export_leaderboard.py \
  --input artifacts/results/ \
  --output examples/leaderboard_2025.md
```

### Hugging Face Integration

Models can be published to Hugging Face Hub (manual process):

1. Train model and generate model card
2. Review artifacts in `artifacts/models/`
3. Create HF Hub repository
4. Upload model files manually
5. Add model card as README

**Note**: No automatic upload. All publishing requires human review.

---

## 🔐 Security & Credentials

### Environment Variables

Create `.env` file (gitignored):

```bash
# Broker credentials
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here

# Safety API
SAFETY_API_TOKEN=generate_secure_random_token

# Environment (NEVER set to 'production' without approval)
QUANTZOO_ENV=development
```

### Credential Rotation

Rotate API keys every 90 days:

1. Generate new keys in broker portal
2. Update `.env` file
3. Restart services
4. Verify functionality
5. Revoke old keys after 24 hours

See `ops/RUNBOOK.md` for detailed procedures.

---

## 📚 Additional Documentation

- **Operations**: `ops/RUNBOOK.md` - Emergency procedures, monitoring, deployment
- **Model Cards**: `docs/model_card_template.md` - ML model documentation template
- **PR Checklist**: `pr_checklist.md` - Required approvals for production features
- **Implementation**: `IMPLEMENTATION_SUMMARY.md` - Complete feature overview

---

## ⚖️ Legal & Compliance

**DISCLAIMER**: QuantZoo is for research and educational purposes. 

Before using for live trading:
- Obtain proper regulatory licenses
- Get compliance approval
- Implement risk controls
- Have legal review terms
- Understand liability
- Follow all applicable regulations

See `pr_checklist.md` for required approvals before production deployment.
