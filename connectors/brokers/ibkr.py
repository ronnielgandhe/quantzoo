"""
Interactive Brokers (IBKR) connector.

⚠️ WARNING: This connector can place REAL ORDERS with REAL MONEY.
Only enable live trading after extensive testing and compliance approval.

Requires ib_insync package and running IB Gateway or TWS.
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


class IBKRBroker(BrokerInterface):
    """
    Interactive Brokers connector via ib_insync.
    
    Configuration:
        - host: IB Gateway/TWS host (default: 127.0.0.1)
        - port: IB Gateway port (paper: 7497, live: 7496)
        - client_id: Client ID for connection (default: 1)
        - account: Account number (optional)
        - dry_run: If True, use paper trading port (default: True)
    
    ⚠️ CRITICAL SAFETY:
    - Requires IB Gateway or TWS running locally
    - Port 7497 = PAPER TRADING
    - Port 7496 = LIVE TRADING
    - Double-check port configuration before running
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IBKR connector."""
        super().__init__(config)
        
        self.host = config.get('host', '127.0.0.1')
        self.client_id = config.get('client_id', 1)
        self.account = config.get('account', None)
        
        # Determine port based on dry_run
        if self.dry_run:
            self.port = config.get('port', 7497)  # Paper trading
            logger.info("📝 IBKR connector in PAPER TRADING mode (port 7497)")
        else:
            self.port = config.get('port', 7496)  # Live trading
            logger.warning("⚠️  ⚠️  ⚠️  IBKR LIVE TRADING MODE (port 7496) ⚠️  ⚠️  ⚠️")
            logger.warning("Real orders will be placed with real money!")
            
            if self.env != 'production':
                raise RuntimeError(
                    f"Live trading requires QUANTZOO_ENV=production. Current: {self.env}"
                )
        
        # Initialize ib_insync client
        try:
            from ib_insync import IB, Stock, MarketOrder, LimitOrder
            self.ib = IB()
            self.Stock = Stock
            self.MarketOrder = MarketOrder
            self.LimitOrder = LimitOrder
            
            logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            
            if self.ib.isConnected():
                logger.info("✅ Connected to IBKR")
                # Get account if not specified
                if not self.account:
                    accounts = self.ib.managedAccounts()
                    if accounts:
                        self.account = accounts[0]
                        logger.info(f"Using account: {self.account}")
            else:
                raise ConnectionError("Failed to connect to IBKR")
        
        except ImportError:
            logger.error("ib_insync package not installed. Install with: pip install ib_insync")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise
    
    def place_order(self, order: Order) -> str:
        """
        Place order via IBKR.
        
        Args:
            order: Order to place
            
        Returns:
            Order ID (permId from IBKR)
        """
        # Safety checks
        if not self._safety_check():
            raise RuntimeError("Safety check failed. Order not placed.")
        
        # Live trading requires confirmation
        if not self.dry_run:
            logger.warning(f"⚠️  PLACING LIVE ORDER: {order.symbol} {order.side.value} {order.quantity}")
            logger.warning("This will execute with REAL MONEY. Implement confirmation system before production use.")
        
        # Create contract
        contract = self.Stock(order.symbol, 'SMART', 'USD')
        
        # Create IBKR order
        if order.order_type == OrderType.MARKET:
            ib_order = self.MarketOrder(order.side.value, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                raise ValueError("Limit order requires price")
            ib_order = self.LimitOrder(order.side.value, order.quantity, order.price)
        else:
            raise ValueError(f"Order type {order.order_type} not yet supported")
        
        # Place order
        try:
            trade = self.ib.placeOrder(contract, ib_order)
            
            # Wait for acknowledgement
            self.ib.sleep(1)
            
            order_id = str(trade.order.permId)
            logger.info(f"Order placed: {order_id} - {order.symbol} {order.side.value} {order.quantity}")
            
            return order_id
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        try:
            # Find trade by permId
            for trade in self.ib.trades():
                if str(trade.order.permId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order cancelled: {order_id}")
                    return True
            
            logger.warning(f"Order not found: {order_id}")
            return False
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        try:
            for trade in self.ib.trades():
                if str(trade.order.permId) == order_id:
                    status = trade.orderStatus.status
                    
                    # Map IBKR status to our status
                    status_map = {
                        'Filled': OrderStatus.FILLED,
                        'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
                        'Cancelled': OrderStatus.CANCELLED,
                        'PendingCancel': OrderStatus.PENDING,
                        'Rejected': OrderStatus.REJECTED,
                        'Submitted': OrderStatus.PENDING,
                        'PreSubmitted': OrderStatus.PENDING,
                    }
                    
                    return status_map.get(status, OrderStatus.PENDING)
            
            raise ValueError(f"Order not found: {order_id}")
        
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            raise
    
    def get_positions(self) -> List[Position]:
        """Get positions from IBKR."""
        try:
            positions = []
            
            for ib_position in self.ib.positions():
                if self.account and ib_position.account != self.account:
                    continue
                
                # Get current price
                contract = ib_position.contract
                ticker = self.ib.reqTicker(contract)
                current_price = ticker.marketPrice() if ticker else ib_position.avgCost
                
                position = Position(
                    symbol=contract.symbol,
                    quantity=float(ib_position.position),
                    avg_entry_price=float(ib_position.avgCost),
                    current_price=float(current_price),
                    unrealized_pnl=float(ib_position.unrealizedPNL),
                    realized_pnl=0.0,  # Not available in position object
                    market_value=float(ib_position.position * current_price)
                )
                positions.append(position)
            
            return positions
        
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance from IBKR."""
        try:
            account_values = self.ib.accountValues(account=self.account)
            
            balance_dict = {}
            for av in account_values:
                if av.tag == 'CashBalance' and av.currency == 'USD':
                    balance_dict['cash'] = float(av.value)
                elif av.tag == 'NetLiquidation' and av.currency == 'USD':
                    balance_dict['equity'] = float(av.value)
                elif av.tag == 'GrossPositionValue' and av.currency == 'USD':
                    balance_dict['market_value'] = float(av.value)
                elif av.tag == 'UnrealizedPnL' and av.currency == 'USD':
                    balance_dict['unrealized_pnl'] = float(av.value)
            
            return balance_dict
        
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise
    
    def get_fills(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None) -> List[Fill]:
        """
        Get fills from IBKR.
        
        Note: IBKR fill history is limited. For complete history, query executions separately.
        """
        try:
            fills = []
            
            for trade in self.ib.trades():
                for fill_obj in trade.fills:
                    if symbol and trade.contract.symbol != symbol:
                        continue
                    
                    fill_time = datetime.fromisoformat(fill_obj.time)
                    if start_date and fill_time < start_date:
                        continue
                    
                    fill = Fill(
                        order_id=str(trade.order.permId),
                        symbol=trade.contract.symbol,
                        side=OrderSide.BUY if fill_obj.execution.side == 'BOT' else OrderSide.SELL,
                        quantity=float(fill_obj.execution.shares),
                        price=float(fill_obj.execution.price),
                        commission=float(fill_obj.commissionReport.commission) if fill_obj.commissionReport else 0.0,
                        timestamp=fill_time
                    )
                    fills.append(fill)
            
            return fills
        
        except Exception as e:
            logger.error(f"Failed to get fills: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")


# Configuration template
IBKR_CONFIG_TEMPLATE = """
# Interactive Brokers Configuration

## Prerequisites

1. Install IB Gateway or Trader Workstation (TWS)
2. Configure API access in IB Gateway/TWS settings
3. Enable socket client connections
4. Install ib_insync: pip install ib_insync

## Configuration

```python
ibkr_config = {
    'host': '127.0.0.1',
    'port': 7497,  # Paper trading port (7496 for live)
    'client_id': 1,
    'account': None,  # Will use first available account
    'dry_run': True,  # REQUIRED for paper trading
}
```

## Port Configuration

⚠️ **CRITICAL**: Verify port before running!

- **7497**: Paper trading (demo account)
- **7496**: Live trading (real money)

## Safety Checklist for Live Trading

Before using port 7496 (live trading):

- [ ] Set QUANTZOO_ENV=production
- [ ] Test extensively with port 7497 (paper)
- [ ] Verify account number is correct
- [ ] Implement position size limits
- [ ] Configure stop-losses
- [ ] Set up monitoring
- [ ] Get compliance approval
- [ ] Have kill switch ready
- [ ] Understand IBKR fee structure
- [ ] Know margin requirements

## Manual Steps

1. Download IB Gateway from IBKR website
2. Install and configure API access
3. Set socket port (7497 for paper)
4. Start IB Gateway and login
5. Test connection with dry_run=True
6. Monitor for stability
7. Get approval before live trading

## Resources

- IB Gateway: https://www.interactivebrokers.com/en/index.php?f=16457
- ib_insync: https://github.com/erdewit/ib_insync
- API Docs: https://interactivebrokers.github.io/tws-api/
"""
