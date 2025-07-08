import hmac
import hashlib
import time
import json
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class MEXCClient:
    """MEXC API client for spot trading"""

    BASE_URL = "https://api.mexc.com"
    WS_URL = "wss://wbs.mexc.com/ws"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = None
        self.ws_session = None
        self.ws_connection = None

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature"""
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def _request(
        self, method: str, endpoint: str, params: Dict = None, signed: bool = False
    ) -> Dict:
        """Make HTTP request to MEXC API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {"X-MEXC-APIKEY": self.api_key, "Content-Type": "application/json"}

        if signed:
            timestamp = int(time.time() * 1000)
            params = params or {}
            params["timestamp"] = timestamp
            params["recvWindow"] = 5000

            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params["signature"] = signature

        try:
            async with self.session.request(
                method, url, params=params, headers=headers
            ) as response:
                data = await response.json()
                if response.status != 200:
                    logger.error(f"API Error: {data}")
                    raise Exception(f"API Error: {data}")
                return data
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    # Market Data Methods
    async def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol information"""
        return await self._request("GET", "/api/v3/exchangeInfo")

    async def get_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker price change statistics"""
        params = {"symbol": symbol}
        return await self._request("GET", "/api/v3/ticker/24hr", params)

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get current order book"""
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", "/api/v3/depth", params)

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", "/api/v3/trades", params)

    async def get_klines(
        self, symbol: str, interval: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """Get candlestick data"""
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        data = await self._request("GET", "/api/v3/klines", params)

        # Convert to DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        for col in ["trades"]:
            df[col] = df[col].astype(int)

        df.set_index("timestamp", inplace=True)
        df.attrs["pair"] = symbol

        return df[["open", "high", "low", "close", "volume"]]

    # Account Methods
    async def get_account(self) -> Dict:
        """Get current account information"""
        return await self._request("GET", "/api/v3/account", signed=True)

    async def get_balance(self, asset: str = None) -> Dict:
        """Get account balance for specific asset or all assets"""
        account = await self.get_account()
        balances = {
            b["asset"]: {"free": float(b["free"]), "locked": float(b["locked"])}
            for b in account["balances"]
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        }

        if asset:
            return balances.get(asset, {"free": 0, "locked": 0})
        return balances

    # Trading Methods
    async def create_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        order_type: str,  # LIMIT, MARKET, etc
        quantity: float,
        price: float = None,
        time_in_force: str = "GTC",
    ) -> Dict:
        """Create a new order"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timeInForce": time_in_force,
        }

        if order_type == "LIMIT" and price:
            params["price"] = price

        return await self._request("POST", "/api/v3/order", params, signed=True)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an active order"""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("DELETE", "/api/v3/order", params, signed=True)

    async def get_order(self, symbol: str, order_id: str) -> Dict:
        """Get order details"""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("GET", "/api/v3/order", params, signed=True)

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/api/v3/openOrders", params, signed=True)

    async def get_all_orders(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get all orders for a symbol"""
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", "/api/v3/allOrders", params, signed=True)

    # WebSocket Methods
    async def start_websocket(self):
        """Start WebSocket connection"""
        self.ws_session = aiohttp.ClientSession()
        self.ws_connection = await self.ws_session.ws_connect(self.WS_URL)
        logger.info("WebSocket connected")

    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates"""
        if not self.ws_connection:
            await self.start_websocket()

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.miniTicker.v3.api@{symbol}"],
        }
        await self.ws_connection.send_json(subscribe_msg)
        logger.info(f"Subscribed to ticker updates for {symbol}")

    async def subscribe_orderbook(self, symbol: str, level: int = 20):
        """Subscribe to order book updates"""
        if not self.ws_connection:
            await self.start_websocket()

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.limit.depth.v3.api@{symbol}@{level}"],
        }
        await self.ws_connection.send_json(subscribe_msg)
        logger.info(f"Subscribed to order book updates for {symbol}")

    async def subscribe_trades(self, symbol: str):
        """Subscribe to trade updates"""
        if not self.ws_connection:
            await self.start_websocket()

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.deals.v3.api@{symbol}"],
        }
        await self.ws_connection.send_json(subscribe_msg)
        logger.info(f"Subscribed to trade updates for {symbol}")

    async def subscribe_klines(self, symbol: str, interval: str = "1m"):
        """Subscribe to kline/candlestick updates"""
        if not self.ws_connection:
            await self.start_websocket()

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.kline.v3.api@{symbol}@{interval}"],
        }
        await self.ws_connection.send_json(subscribe_msg)
        logger.info(f"Subscribed to {interval} kline updates for {symbol}")

    async def read_websocket(self):
        """Read messages from WebSocket"""
        async for msg in self.ws_connection:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                yield data
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {msg.data}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.info("WebSocket closed")
                break

    async def close(self):
        """Close all connections"""
        if self.ws_connection:
            await self.ws_connection.close()
        if self.ws_session:
            await self.ws_session.close()
        if self.session:
            await self.session.close()


class MEXCTradeExecutor:
    """Trade execution handler for MEXC"""

    def __init__(self, client: MEXCClient):
        self.client = client

    async def execute_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Dict:
        """Execute market order"""
        try:
            order = await self.client.create_order(
                symbol=symbol, side=side, order_type="MARKET", quantity=quantity
            )
            logger.info(f"Market order executed: {order}")
            return order
        except Exception as e:
            logger.error(f"Failed to execute market order: {str(e)}")
            raise

    async def execute_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Dict:
        """Execute limit order"""
        try:
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                price=price,
            )
            logger.info(f"Limit order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Failed to place limit order: {str(e)}")
            raise

    async def place_stop_loss(
        self, symbol: str, quantity: float, stop_price: float, side: str = "SELL"
    ) -> Dict:
        """Place stop loss order"""
        # MEXC doesn't support stop-loss orders directly in spot trading
        # You would need to monitor prices and execute market orders
        logger.warning(
            "Stop-loss orders need to be monitored manually for MEXC spot trading"
        )
        return {"message": "Stop-loss monitoring required"}

    async def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        risk_percent: float,
        stop_loss_price: float,
        entry_price: float,
    ) -> float:
        """Calculate position size based on risk management"""
        # Get symbol info for lot size and precision
        exchange_info = await self.client.get_exchange_info()
        symbol_info = next(
            (s for s in exchange_info["symbols"] if s["symbol"] == symbol), None
        )

        if not symbol_info:
            raise ValueError(f"Symbol {symbol} not found")

        # Calculate position size
        risk_amount = account_balance * risk_percent
        price_difference = abs(entry_price - stop_loss_price)
        position_value = risk_amount / (price_difference / entry_price)
        quantity = position_value / entry_price

        # Apply lot size rules
        min_qty = float(symbol_info["baseAssetPrecision"])
        step_size = float(symbol_info["quotePrecision"])

        # Round to appropriate precision
        quantity = round(quantity / step_size) * step_size
        quantity = max(quantity, min_qty)

        return quantity
