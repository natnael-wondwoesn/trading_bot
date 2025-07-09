#!/usr/bin/env python3
"""
Abstract Exchange Interface
Standardizes exchange implementations for MEXC, Bybit, and future exchanges
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class ExchangeInfo:
    """Exchange information structure"""

    name: str
    supports_spot: bool
    supports_futures: bool
    supports_websocket: bool
    base_url: str
    websocket_url: str
    rate_limits: Dict[str, int]


@dataclass
class MarketData:
    """Standardized market data structure"""

    symbol: str
    price: float
    volume: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: str
    exchange: str


@dataclass
class OrderResult:
    """Standardized order result structure"""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    commission: float
    timestamp: str
    exchange: str


@dataclass
class Balance:
    """Standardized balance structure"""

    asset: str
    free: float
    locked: float
    total: float


class BaseExchangeClient(ABC):
    """Abstract base class for all exchange clients"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange_info = self._get_exchange_info()

    @abstractmethod
    def _get_exchange_info(self) -> ExchangeInfo:
        """Get exchange-specific information"""
        pass

    # Market Data Methods
    @abstractmethod
    async def get_server_time(self) -> Dict:
        """Get server time"""
        pass

    @abstractmethod
    async def get_exchange_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> MarketData:
        """Get 24hr ticker data for a symbol"""
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get current order book"""
        pass

    @abstractmethod
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 100
    ) -> pd.DataFrame:
        """Get klines/candlestick data"""
        pass

    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        pass

    # Account Methods
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass

    @abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances for all assets"""
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Balance:
        """Get balance for specific asset"""
        pass

    # Trading Methods
    @abstractmethod
    async def create_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> OrderResult:
        """Create market order"""
        pass

    @abstractmethod
    async def create_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Create limit order"""
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order"""
        pass

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> OrderResult:
        """Get order details"""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[OrderResult]:
        """Get open orders"""
        pass

    @abstractmethod
    async def get_order_history(
        self, symbol: str = None, limit: int = 50
    ) -> List[OrderResult]:
        """Get order history"""
        pass

    # Connection Management
    @abstractmethod
    async def close(self):
        """Close connections and cleanup"""
        pass

    # Helper Methods
    def get_exchange_name(self) -> str:
        """Get exchange name"""
        return self.exchange_info.name

    def supports_websocket(self) -> bool:
        """Check if exchange supports WebSocket"""
        return self.exchange_info.supports_websocket

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for the exchange"""
        # Default implementation - can be overridden
        return symbol.upper()

    def get_min_order_size(self, symbol: str) -> float:
        """Get minimum order size for symbol"""
        # Default implementation - should be overridden
        return 0.001

    def calculate_fees(
        self, quantity: float, price: float, is_maker: bool = False
    ) -> float:
        """Calculate trading fees"""
        # Default implementation - should be overridden
        return quantity * price * (0.001 if is_maker else 0.001)


class BaseDataFeed(ABC):
    """Abstract base class for all data feeds"""

    def __init__(self, client: BaseExchangeClient):
        self.client = client
        self.running = False
        self.subscriptions = {}
        self.callbacks = {}

    @abstractmethod
    async def start(self, symbols: List[str], interval: str = "1m"):
        """Start the data feed"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the data feed"""
        pass

    @abstractmethod
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates"""
        pass

    @abstractmethod
    async def subscribe_klines(self, symbol: str, interval: str):
        """Subscribe to kline updates"""
        pass

    @abstractmethod
    async def subscribe_trades(self, symbol: str):
        """Subscribe to trade updates"""
        pass

    def subscribe_callback(self, symbol: str, event_type: str, callback):
        """Subscribe to callback for events"""
        key = f"{symbol}_{event_type}"
        self.callbacks[key] = callback

    def unsubscribe_callback(self, symbol: str, event_type: str):
        """Unsubscribe from callback"""
        key = f"{symbol}_{event_type}"
        if key in self.callbacks:
            del self.callbacks[key]

    async def trigger_callback(self, symbol: str, event_type: str, data: Dict):
        """Trigger registered callbacks"""
        key = f"{symbol}_{event_type}"
        if key in self.callbacks:
            try:
                callback = self.callbacks[key]
                if hasattr(callback, "__call__"):
                    if hasattr(callback, "__await__"):
                        await callback(symbol, data)
                    else:
                        callback(symbol, data)
            except Exception as e:
                print(f"Error in callback {key}: {e}")


class BaseTradeExecutor(ABC):
    """Abstract base class for trade executors"""

    def __init__(self, client: BaseExchangeClient):
        self.client = client

    @abstractmethod
    async def execute_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> OrderResult:
        """Execute market order"""
        pass

    @abstractmethod
    async def execute_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Execute limit order"""
        pass

    @abstractmethod
    async def place_stop_loss(
        self, symbol: str, side: str, quantity: float, stop_price: float
    ) -> OrderResult:
        """Place stop loss order"""
        pass

    async def calculate_order_size(
        self, symbol: str, amount_usd: float, current_price: float
    ) -> float:
        """Calculate order size based on USD amount"""
        min_size = self.client.get_min_order_size(symbol)
        calculated_size = amount_usd / current_price
        return max(min_size, calculated_size)

    async def validate_order(
        self, symbol: str, side: str, quantity: float, price: float = None
    ) -> Tuple[bool, str]:
        """Validate order parameters"""
        try:
            # Check minimum order size
            min_size = self.client.get_min_order_size(symbol)
            if quantity < min_size:
                return False, f"Order size {quantity} below minimum {min_size}"

            # Check account balance
            balances = await self.client.get_balances()

            if side.upper() == "BUY":
                # Check quote asset balance (usually USDT)
                quote_asset = "USDT"  # Simplified - should be extracted from symbol
                if quote_asset in balances:
                    required_balance = quantity * (price or 0)
                    if balances[quote_asset].free < required_balance:
                        return False, f"Insufficient {quote_asset} balance"
            else:
                # Check base asset balance
                base_asset = symbol.replace("USDT", "")  # Simplified
                if base_asset in balances:
                    if balances[base_asset].free < quantity:
                        return False, f"Insufficient {base_asset} balance"

            return True, "Order validation passed"

        except Exception as e:
            return False, f"Validation error: {str(e)}"


# Exchange-specific implementations would inherit from these base classes
# For example:
# class MEXCClient(BaseExchangeClient): ...
# class BybitClient(BaseExchangeClient): ...
# class MEXCDataFeed(BaseDataFeed): ...
# class BybitDataFeed(BaseDataFeed): ...
