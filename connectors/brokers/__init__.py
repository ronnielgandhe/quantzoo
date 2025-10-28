"""
Broker connectors for paper and live trading.

⚠️ WARNING: Live trading connectors can place real orders with real money.
Always test in paper mode first and get compliance approval before live trading.
"""
from .base import (
    BrokerInterface,
    Order,
    Position,
    Fill,
    OrderSide,
    OrderType,
    OrderStatus,
)
from .paper import PaperBroker

__all__ = [
    'BrokerInterface',
    'Order',
    'Position',
    'Fill',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'PaperBroker',
]

# Optional imports for live connectors
try:
    from .alpaca import AlpacaBroker
    __all__.append('AlpacaBroker')
except ImportError:
    pass

try:
    from .ibkr import IBKRBroker
    __all__.append('IBKRBroker')
except ImportError:
    pass
