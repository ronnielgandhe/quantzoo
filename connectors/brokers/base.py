"""
Base broker interface for paper and live trading connectors.

All broker implementations must inherit from this base class and implement
the required safety checks.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side: BUY or SELL."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    client_order_id: Optional[str] = None


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float


@dataclass
class Fill:
    """Order fill representation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime


class BrokerInterface(ABC):
    """
    Base interface for all broker connectors.
    
    ⚠️ CRITICAL SAFETY REQUIREMENTS:
    - All implementations MUST check STOP_ALL_TRADING flag before placing orders
    - All implementations MUST check QUANTZOO_ENV environment variable
    - All implementations MUST implement two-step confirmation for live orders
    - All implementations MUST log every order attempt
    """
    
    # Global kill switch - can be set by safety API
    STOP_ALL_TRADING = False
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker connector.
        
        Args:
            config: Configuration dictionary with broker-specific settings
        """
        self.config = config
        self.env = os.getenv('QUANTZOO_ENV', 'development')
        self.dry_run = config.get('dry_run', True)
        
        # Safety checks on initialization
        if not self.dry_run and self.env != 'production':
            logger.warning(
                f"⚠️  Live trading enabled but QUANTZOO_ENV={self.env}. "
                f"Set QUANTZOO_ENV=production for live trading."
            )
        
        logger.info(f"Initialized {self.__class__.__name__} (env={self.env}, dry_run={self.dry_run})")
    
    def _safety_check(self) -> bool:
        """
        Perform safety checks before order placement.
        
        Returns:
            True if safe to place order, False otherwise
        """
        # Check global kill switch
        if BrokerInterface.STOP_ALL_TRADING:
            logger.error("❌ STOP_ALL_TRADING is enabled. Order rejected.")
            return False
        
        # Check environment
        if not self.dry_run and self.env != 'production':
            logger.error(f"❌ Live trading requires QUANTZOO_ENV=production (current: {self.env})")
            return False
        
        # Additional checks can be added here
        return True
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        Place an order.
        
        Args:
            order: Order to place
            
        Returns:
            Order ID
            
        Raises:
            Exception if order placement fails
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of positions
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance.
        
        Returns:
            Dictionary with balance information
        """
        pass
    
    @abstractmethod
    def get_fills(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None) -> List[Fill]:
        """
        Get order fills.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by date (optional)
            
        Returns:
            List of fills
        """
        pass
    
    def close_all_positions(self) -> List[str]:
        """
        Emergency function to close all positions.
        
        Returns:
            List of order IDs for closing orders
        """
        logger.warning("⚠️  CLOSING ALL POSITIONS")
        
        positions = self.get_positions()
        order_ids = []
        
        for position in positions:
            if position.quantity == 0:
                continue
            
            # Determine side to close
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            
            close_order = Order(
                symbol=position.symbol,
                side=side,
                quantity=abs(position.quantity),
                order_type=OrderType.MARKET
            )
            
            try:
                order_id = self.place_order(close_order)
                order_ids.append(order_id)
                logger.info(f"Closing position: {position.symbol} x {position.quantity}")
            except Exception as e:
                logger.error(f"Failed to close position {position.symbol}: {e}")
        
        return order_ids
