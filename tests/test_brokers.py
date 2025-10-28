"""
Integration tests for broker connectors and safety systems.
"""
import pytest
import time
from datetime import datetime
from connectors.brokers import (
    PaperBroker,
    BrokerInterface,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
)


class TestPaperBroker:
    """Test paper broker execution."""
    
    def test_initialization(self):
        """Test broker initializes correctly."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        assert broker.cash == 100000
        assert len(broker.positions) == 0
        assert len(broker.fills) == 0
    
    def test_market_order_buy(self):
        """Test market buy order."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        broker.update_market_price('AAPL', 150.0)
        
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        order_id = broker.place_order(order)
        assert order_id is not None
        
        # Check order status
        status = broker.get_order_status(order_id)
        assert status == OrderStatus.FILLED
        
        # Check position
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == 'AAPL'
        assert positions[0].quantity == 10
        
        # Check cash reduced
        assert broker.cash < 100000
    
    def test_market_order_sell(self):
        """Test market sell order."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        broker.update_market_price('AAPL', 150.0)
        
        # First buy
        buy_order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        broker.place_order(buy_order)
        
        # Then sell
        sell_order = Order(
            symbol='AAPL',
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.MARKET
        )
        broker.place_order(sell_order)
        
        # Check position reduced
        positions = broker.get_positions()
        assert positions[0].quantity == 5
    
    def test_slippage_applied(self):
        """Test slippage is applied to orders."""
        broker = PaperBroker({
            'initial_cash': 100000,
            'slippage_bps': 10,  # 0.1%
            'dry_run': True
        })
        broker.update_market_price('AAPL', 100.0)
        
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        broker.place_order(order)
        
        # Check fill price includes slippage
        fills = broker.get_fills()
        assert len(fills) == 1
        # Buy should have unfavorable slippage (higher price)
        assert fills[0].price > 100.0
    
    def test_commission_charged(self):
        """Test commissions are charged."""
        broker = PaperBroker({
            'initial_cash': 100000,
            'commission_per_share': 0.01,
            'dry_run': True
        })
        broker.update_market_price('AAPL', 100.0)
        
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        broker.place_order(order)
        
        fills = broker.get_fills()
        assert fills[0].commission > 0
    
    def test_insufficient_cash(self):
        """Test order rejection with insufficient cash."""
        broker = PaperBroker({'initial_cash': 100, 'dry_run': True})
        broker.update_market_price('AAPL', 150.0)
        
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=100,  # Would cost 15000+
            order_type=OrderType.MARKET
        )
        
        order_id = broker.place_order(order)
        status = broker.get_order_status(order_id)
        assert status == OrderStatus.REJECTED
    
    def test_account_balance(self):
        """Test account balance calculation."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        broker.update_market_price('AAPL', 100.0)
        
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        broker.place_order(order)
        
        balance = broker.get_account_balance()
        assert 'cash' in balance
        assert 'equity' in balance
        assert 'market_value' in balance
        assert balance['equity'] > 0


class TestSafetyChecks:
    """Test safety mechanisms."""
    
    def test_kill_switch_blocks_orders(self):
        """Test that kill switch prevents order placement."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        broker.update_market_price('AAPL', 150.0)
        
        # Activate kill switch
        BrokerInterface.STOP_ALL_TRADING = True
        
        order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        # Should raise exception
        with pytest.raises(RuntimeError, match="Safety check failed"):
            broker.place_order(order)
        
        # Reset kill switch
        BrokerInterface.STOP_ALL_TRADING = False
    
    def test_safety_check_passes_in_safe_mode(self):
        """Test safety check passes in dry-run mode."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        assert broker._safety_check() is True
    
    def test_close_all_positions(self):
        """Test emergency position closure."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        broker.update_market_price('AAPL', 150.0)
        broker.update_market_price('MSFT', 250.0)
        
        # Open positions
        for symbol in ['AAPL', 'MSFT']:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET
            )
            broker.place_order(order)
        
        # Close all
        order_ids = broker.close_all_positions()
        assert len(order_ids) == 2
        
        # Check positions are closed
        time.sleep(0.1)  # Allow for execution
        positions = broker.get_positions()
        for pos in positions:
            assert pos.quantity == 0


class TestMultiAsset:
    """Test multi-asset position tracking."""
    
    def test_multiple_symbols(self):
        """Test trading multiple symbols."""
        broker = PaperBroker({'initial_cash': 100000, 'dry_run': True})
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        for i, symbol in enumerate(symbols):
            broker.update_market_price(symbol, 100.0 + i * 50)
            
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=5,
                order_type=OrderType.MARKET
            )
            broker.place_order(order)
        
        positions = broker.get_positions()
        assert len(positions) == 3
        assert set(p.symbol for p in positions) == set(symbols)


def test_broker_interface_abstract():
    """Test that BrokerInterface cannot be instantiated."""
    with pytest.raises(TypeError):
        BrokerInterface({})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
