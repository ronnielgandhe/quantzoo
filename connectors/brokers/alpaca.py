"""
Alpaca broker connector.

⚠️ WARNING: This connector can place REAL ORDERS with REAL MONEY.
Only enable live trading after:
1. Setting QUANTZOO_ENV=production
2. Configuring credentials properly
3. Testing extensively in paper mode
4. Getting compliance approval
5. Implementing proper risk limits

DO NOT use without understanding the risks.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import logging

from .base import (
    BrokerInterface, Order, Position, Fill,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker connector.
    
    Configuration:
        - api_key: Alpaca API key (from environment: ALPACA_API_KEY)
        - api_secret: Alpaca API secret (from environment: ALPACA_API_SECRET)
        - base_url: API base URL (paper: https://paper-api.alpaca.markets, live: https://api.alpaca.markets)
        - dry_run: If True, use paper trading endpoint (default: True)
    
    ⚠️ SAFETY: This class will REFUSE to place orders unless:
    - QUANTZOO_ENV=production (for live trading)
    - STOP_ALL_TRADING=False
    - dry_run=True (for paper) OR explicit confirmation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca connector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get credentials from environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
        
        # Determine endpoint
        if self.dry_run:
            self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
            logger.info("📝 Alpaca connector in PAPER TRADING mode")
        else:
            self.base_url = config.get('base_url', 'https://api.alpaca.markets')
            logger.warning("⚠️  ⚠️  ⚠️  ALPACA LIVE TRADING MODE ⚠️  ⚠️  ⚠️")
            logger.warning("Real orders will be placed with real money!")
            
            # Extra safety check
            if self.env != 'production':
                raise RuntimeError(
                    f"Live trading requires QUANTZOO_ENV=production. Current: {self.env}"
                )
        
        # Initialize Alpaca client (stubbed - requires alpaca-trade-api package)
        try:
            import alpaca_trade_api as tradeapi
            self.client = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url
            )
            logger.info(f"Alpaca client initialized: {self.base_url}")
        except ImportError:
            logger.error("alpaca-trade-api package not installed. Install with: pip install alpaca-trade-api")
            raise
    
    def _require_confirmation(self, action: str) -> bool:
        """
        Require human confirmation for critical actions.
        
        Args:
            action: Description of action requiring confirmation
            
        Returns:
            True if confirmed, False otherwise
        """
        if self.dry_run:
            # Paper trading doesn't require confirmation
            return True
        
        logger.warning(f"⚠️  CONFIRMATION REQUIRED: {action}")
        logger.warning("This is LIVE TRADING and will use REAL MONEY.")
        logger.warning("To proceed, you must manually approve this action.")
        
        # In production, this would integrate with a confirmation system
        # For safety, we return False by default
        return False
    
    def place_order(self, order: Order) -> str:
        """
        Place order via Alpaca API.
        
        Args:
            order: Order to place
            
        Returns:
            Order ID
        """
        # Safety checks
        if not self._safety_check():
            raise RuntimeError("Safety check failed. Order not placed.")
        
        # Live trading requires confirmation
        if not self.dry_run:
            confirmed = self._require_confirmation(
                f"Place {order.side.value} order: {order.symbol} x {order.quantity}"
            )
            if not confirmed:
                logger.error("Order not confirmed. Cancelling.")
                raise RuntimeError("Order requires manual confirmation for live trading.")
        
        # Map order type
        alpaca_order_type = order.order_type.value.lower()
        alpaca_side = order.side.value.lower()
        
        # Place order
        try:
            alpaca_order = self.client.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=alpaca_side,
                type=alpaca_order_type,
                time_in_force=order.time_in_force,
                limit_price=order.price if order.order_type == OrderType.LIMIT else None,
                stop_price=order.stop_price if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] else None,
                client_order_id=order.client_order_id
            )
            
            logger.info(f"Order placed: {alpaca_order.id} - {order.symbol} {order.side.value} {order.quantity}")
            return alpaca_order.id
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Alpaca API."""
        try:
            self.client.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca."""
        try:
            alpaca_order = self.client.get_order(order_id)
            
            # Map Alpaca status to our status
            status_map = {
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'cancelled': OrderStatus.CANCELLED,
                'expired': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'pending_new': OrderStatus.PENDING,
                'accepted': OrderStatus.PENDING,
                'new': OrderStatus.PENDING,
            }
            
            return status_map.get(alpaca_order.status, OrderStatus.PENDING)
        
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            raise
    
    def get_positions(self) -> List[Position]:
        """Get positions from Alpaca."""
        try:
            alpaca_positions = self.client.list_positions()
            
            positions = []
            for ap in alpaca_positions:
                position = Position(
                    symbol=ap.symbol,
                    quantity=float(ap.qty),
                    avg_entry_price=float(ap.avg_entry_price),
                    current_price=float(ap.current_price),
                    unrealized_pnl=float(ap.unrealized_pl),
                    realized_pnl=0.0,  # Alpaca doesn't provide this in position object
                    market_value=float(ap.market_value)
                )
                positions.append(position)
            
            return positions
        
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account information from Alpaca."""
        try:
            account = self.client.get_account()
            
            return {
                'cash': float(account.cash),
                'equity': float(account.equity),
                'market_value': float(account.long_market_value) + float(account.short_market_value),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
            }
        
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise
    
    def get_fills(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None) -> List[Fill]:
        """
        Get fills from Alpaca.
        
        Note: Alpaca returns fills as part of order objects, so we query orders.
        """
        try:
            # Query orders
            orders = self.client.list_orders(
                status='filled',
                limit=100,
                after=start_date.isoformat() if start_date else None
            )
            
            fills = []
            for order in orders:
                if symbol and order.symbol != symbol:
                    continue
                
                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=OrderSide.BUY if order.side == 'buy' else OrderSide.SELL,
                    quantity=float(order.filled_qty),
                    price=float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    commission=0.0,  # Alpaca is commission-free for stocks
                    timestamp=datetime.fromisoformat(order.filled_at.replace('Z', '+00:00'))
                )
                fills.append(fill)
            
            return fills
        
        except Exception as e:
            logger.error(f"Failed to get fills: {e}")
            raise


# Configuration template for Alpaca
ALPACA_CONFIG_TEMPLATE = """
# Alpaca Broker Configuration

## Required Environment Variables

Set these in your .env file (NEVER commit to git):

```bash
# Alpaca API credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here

# Environment setting
QUANTZOO_ENV=development  # Set to 'production' for live trading
```

## Configuration Dictionary

```python
alpaca_config = {
    'dry_run': True,  # Use paper trading (REQUIRED for testing)
    'base_url': 'https://paper-api.alpaca.markets',  # Paper trading URL
    
    # For LIVE trading (requires compliance approval):
    # 'dry_run': False,
    # 'base_url': 'https://api.alpaca.markets',
}
```

## Safety Checklist for Live Trading

Before enabling live trading (dry_run=False):

- [ ] Set QUANTZOO_ENV=production in environment
- [ ] Test extensively in paper trading mode
- [ ] Implement position size limits
- [ ] Configure stop-loss mechanisms  
- [ ] Set up monitoring and alerts
- [ ] Get compliance approval
- [ ] Have emergency shutdown procedure ready
- [ ] Verify API credentials are correct
- [ ] Understand commission structure and fees
- [ ] Know market hours and order routing

## Manual Steps

1. Create Alpaca account at https://alpaca.markets
2. Get API keys from dashboard (use paper trading initially)
3. Set environment variables in .env file
4. Test with dry_run=True first
5. Monitor for at least 1 week in paper mode
6. Get approval before switching to live

## Support

- Alpaca Docs: https://alpaca.markets/docs/
- Alpaca API: https://github.com/alpacahq/alpaca-trade-api-python
"""
