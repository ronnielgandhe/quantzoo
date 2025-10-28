"""
Paper broker for simulated trading with realistic slippage and commissions.

This broker simulates order execution without connecting to real markets.
Useful for testing strategies in a safe environment.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging
from collections import defaultdict

from .base import (
    BrokerInterface, Order, Position, Fill,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class PaperBroker(BrokerInterface):
    """
    Simulated broker with realistic execution modeling.
    
    Features:
    - Simulated fills with configurable slippage
    - Commission accounting
    - Position tracking
    - Order book simulation (optional)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize paper broker.
        
        Config options:
            - initial_cash: Starting cash balance (default: 100000)
            - slippage_bps: Slippage in basis points (default: 5)
            - commission_per_share: Commission per share (default: 0.005)
            - min_commission: Minimum commission per order (default: 1.0)
        """
        super().__init__(config)
        
        self.initial_cash = config.get('initial_cash', 100000.0)
        self.cash = self.initial_cash
        self.slippage_bps = config.get('slippage_bps', 5.0)
        self.commission_per_share = config.get('commission_per_share', 0.005)
        self.min_commission = config.get('min_commission', 1.0)
        
        # Track orders, positions, fills
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []
        
        # Current market prices (must be updated externally)
        self.market_prices: Dict[str, float] = {}
        
        logger.info(f"Paper broker initialized with ${self.initial_cash:,.2f}")
    
    def update_market_price(self, symbol: str, price: float) -> None:
        """Update current market price for a symbol."""
        self.market_prices[symbol] = price
    
    def _calculate_slippage(self, price: float, side: OrderSide) -> float:
        """Calculate slippage based on order side."""
        slippage = price * (self.slippage_bps / 10000.0)
        
        # Slippage is unfavorable: higher for buys, lower for sells
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage
    
    def _calculate_commission(self, quantity: float) -> float:
        """Calculate commission for order."""
        commission = quantity * self.commission_per_share
        return max(commission, self.min_commission)
    
    def place_order(self, order: Order) -> str:
        """
        Place a simulated order.
        
        Args:
            order: Order to place
            
        Returns:
            Order ID
        """
        # Safety check
        if not self._safety_check():
            raise RuntimeError("Safety check failed. Order not placed.")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Get current market price
        if order.symbol not in self.market_prices:
            raise ValueError(f"No market price available for {order.symbol}")
        
        market_price = self.market_prices[order.symbol]
        
        # Determine fill price based on order type
        if order.order_type == OrderType.MARKET:
            fill_price = self._calculate_slippage(market_price, order.side)
            status = OrderStatus.FILLED
        elif order.order_type == OrderType.LIMIT:
            # Simplified: assume limit orders fill at limit price if market price is favorable
            if order.price is None:
                raise ValueError("Limit order requires price")
            
            if order.side == OrderSide.BUY and market_price <= order.price:
                fill_price = order.price
                status = OrderStatus.FILLED
            elif order.side == OrderSide.SELL and market_price >= order.price:
                fill_price = order.price
                status = OrderStatus.FILLED
            else:
                fill_price = None
                status = OrderStatus.PENDING
        else:
            # Other order types not yet implemented
            fill_price = None
            status = OrderStatus.PENDING
        
        # Store order
        self.orders[order_id] = {
            'order': order,
            'status': status,
            'fill_price': fill_price,
            'timestamp': datetime.now()
        }
        
        # Execute if filled
        if status == OrderStatus.FILLED and fill_price is not None:
            self._execute_fill(order_id, order, fill_price)
        
        logger.info(f"Order placed: {order_id} - {order.symbol} {order.side.value} "
                   f"{order.quantity} @ {fill_price if fill_price else 'pending'}")
        
        return order_id
    
    def _execute_fill(self, order_id: str, order: Order, fill_price: float) -> None:
        """Execute order fill and update positions."""
        # Calculate commission
        commission = self._calculate_commission(order.quantity)
        
        # Calculate total cost/proceeds
        total_value = fill_price * order.quantity
        
        if order.side == OrderSide.BUY:
            total_cost = total_value + commission
            
            # Check if enough cash
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for order {order_id}. Required: ${total_cost:.2f}, Available: ${self.cash:.2f}")
                self.orders[order_id]['status'] = OrderStatus.REJECTED
                return
            
            self.cash -= total_cost
        else:  # SELL
            total_proceeds = total_value - commission
            self.cash += total_proceeds
        
        # Update position
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=0,
                avg_entry_price=0,
                current_price=fill_price,
                unrealized_pnl=0,
                realized_pnl=0,
                market_value=0
            )
        
        position = self.positions[order.symbol]
        
        # Calculate new position
        if order.side == OrderSide.BUY:
            new_quantity = position.quantity + order.quantity
            if position.quantity >= 0:
                # Adding to long position or opening new long
                position.avg_entry_price = (
                    (position.avg_entry_price * position.quantity + fill_price * order.quantity) / new_quantity
                )
            else:
                # Covering short position
                if new_quantity >= 0:
                    # Realized PnL from covering short
                    covered_quantity = min(abs(position.quantity), order.quantity)
                    realized_pnl = (position.avg_entry_price - fill_price) * covered_quantity
                    position.realized_pnl += realized_pnl
            
            position.quantity = new_quantity
        else:  # SELL
            new_quantity = position.quantity - order.quantity
            if position.quantity <= 0:
                # Adding to short position or opening new short
                position.avg_entry_price = (
                    (position.avg_entry_price * abs(position.quantity) + fill_price * order.quantity) / abs(new_quantity)
                )
            else:
                # Closing long position
                if new_quantity <= 0:
                    # Realized PnL from closing long
                    closed_quantity = min(position.quantity, order.quantity)
                    realized_pnl = (fill_price - position.avg_entry_price) * closed_quantity
                    position.realized_pnl += realized_pnl
            
            position.quantity = new_quantity
        
        position.current_price = fill_price
        self._update_position_pnl(position)
        
        # Record fill
        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now()
        )
        self.fills.append(fill)
        
        logger.info(f"Fill executed: {order.symbol} {order.quantity} @ ${fill_price:.2f}, "
                   f"commission: ${commission:.2f}")
    
    def _update_position_pnl(self, position: Position) -> None:
        """Update position P&L based on current price."""
        if position.quantity > 0:
            # Long position
            position.unrealized_pnl = (position.current_price - position.avg_entry_price) * position.quantity
            position.market_value = position.current_price * position.quantity
        elif position.quantity < 0:
            # Short position
            position.unrealized_pnl = (position.avg_entry_price - position.current_price) * abs(position.quantity)
            position.market_value = position.current_price * abs(position.quantity)
        else:
            position.unrealized_pnl = 0
            position.market_value = 0
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return False
        
        order_info = self.orders[order_id]
        if order_info['status'] == OrderStatus.PENDING:
            order_info['status'] = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")
        
        return self.orders[order_id]['status']
    
    def get_positions(self) -> List[Position]:
        """Get all positions."""
        # Update prices
        for symbol, position in self.positions.items():
            if symbol in self.market_prices:
                position.current_price = self.market_prices[symbol]
                self._update_position_pnl(position)
        
        return list(self.positions.values())
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance."""
        positions = self.get_positions()
        
        total_market_value = sum(p.market_value for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_realized_pnl = sum(p.realized_pnl for p in positions)
        
        equity = self.cash + total_market_value
        
        return {
            'cash': self.cash,
            'equity': equity,
            'market_value': total_market_value,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'initial_cash': self.initial_cash,
            'return_pct': ((equity - self.initial_cash) / self.initial_cash) * 100
        }
    
    def get_fills(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None) -> List[Fill]:
        """Get order fills."""
        fills = self.fills
        
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        
        if start_date:
            fills = [f for f in fills if f.timestamp >= start_date]
        
        return fills
