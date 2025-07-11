#!/usr/bin/env python3
"""
Bybit API Client
Implementation of Bybit V5 API for trading bot integration
https://bybit-exchange.github.io/docs/v5/intro
"""

import asyncio
import logging
import time
import hashlib
import hmac
from urllib.parse import urlencode
from typing import Dict, List, Optional
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class BybitClient:
    """Bybit V5 API client for market data and trading operations"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Bybit V5 API endpoints
        if testnet:
            self.BASE_URL = "https://api-testnet.bybit.com"
        else:
            self.BASE_URL = "https://api.bybit.com"

        self.session = None

    async def __aenter__(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, timestamp: str, params: str) -> str:
        """Generate HMAC signature for Bybit API"""
        param_str = f"{timestamp}{self.api_key}5000{params}"
        return hmac.new(
            self.api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False,
        category: str = "spot",
    ) -> Dict:
        """Make HTTP request to Bybit API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if signed:
            timestamp = str(int(time.time() * 1000))
            headers["X-BAPI-API-KEY"] = self.api_key
            headers["X-BAPI-TIMESTAMP"] = timestamp
            headers["X-BAPI-RECV-WINDOW"] = "5000"

            # Prepare parameters
            params = params or {}
            if category is not None:
                params["category"] = category

            # Create query string for signature
            query_string = urlencode(sorted(params.items())) if params else ""

            # Generate signature
            signature = self._generate_signature(timestamp, query_string)
            headers["X-BAPI-SIGN"] = signature

        try:
            if method == "GET":
                async with self.session.get(
                    url, params=params, headers=headers
                ) as response:
                    data = await response.json()
            elif method == "POST":
                async with self.session.post(
                    url, json=params, headers=headers
                ) as response:
                    data = await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if data.get("retCode") != 0:
                logger.error(f"Bybit API Error: {data}")
                raise Exception(f"Bybit API Error: {data}")

            return data

        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    # Market Data Methods
    async def get_server_time(self) -> Dict:
        """Get server time"""
        return await self._request("GET", "/v5/market/time")

    async def get_instruments_info(
        self, category: str = "spot", symbol: str = None
    ) -> Dict:
        """Get instruments information"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/market/instruments-info", params)

    async def get_ticker(self, symbol: str, category: str = "spot") -> Dict:
        """Get 24hr ticker price change statistics"""
        params = {"category": category, "symbol": symbol}
        response = await self._request("GET", "/v5/market/tickers", params)

        # Extract the specific ticker data
        tickers = response.get("result", {}).get("list", [])
        if tickers:
            ticker = tickers[0]
            # Convert to MEXC-like format for compatibility
            return {
                "symbol": ticker["symbol"],
                "lastPrice": ticker["lastPrice"],
                "priceChangePercent": ticker["price24hPcnt"],
                "volume": ticker["volume24h"],
                "turnover": ticker["turnover24h"],
                "highPrice": ticker["highPrice24h"],
                "lowPrice": ticker["lowPrice24h"],
                "openPrice": ticker["prevPrice24h"],
            }
        return {}

    async def get_orderbook(
        self, symbol: str, limit: int = 25, category: str = "spot"
    ) -> Dict:
        """Get current order book"""
        params = {"category": category, "symbol": symbol, "limit": limit}
        return await self._request("GET", "/v5/market/orderbook", params)

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 200, category: str = "spot"
    ) -> pd.DataFrame:
        """Get klines/candlestick data"""
        # Bybit interval mapping
        interval_map = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }

        bybit_interval = interval_map.get(interval, "1")
        params = {
            "category": category,
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit,
        }

        response = await self._request("GET", "/v5/market/kline", params)

        # Convert to pandas DataFrame
        klines_data = response.get("result", {}).get("list", [])
        if not klines_data:
            return pd.DataFrame()

        # Bybit returns: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        df = pd.DataFrame(
            klines_data,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )

        # Convert data types
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # Sort by timestamp (Bybit returns newest first, we want oldest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Set timestamp as index and return only OHLCV columns like MEXC
        df.set_index("timestamp", inplace=True)
        df.attrs["pair"] = symbol

        logger.info(f"Klines response structure for {symbol}: {len(df)} rows")
        if len(df) > 0:
            logger.info(
                f"First row structure: {len(df.columns)} columns: {df.iloc[0].tolist()}"
            )

        return df[["open", "high", "low", "close", "volume"]]

    async def get_recent_trades(
        self, symbol: str, limit: int = 60, category: str = "spot"
    ) -> Dict:
        """Get recent trades"""
        params = {"category": category, "symbol": symbol, "limit": limit}
        return await self._request("GET", "/v5/market/recent-trade", params)

    # Account Methods
    async def get_account_info(self) -> Dict:
        """Get account information"""
        return await self._request(
            "GET", "/v5/account/info", signed=True, category=None
        )

    async def get_wallet_balance(
        self, account_type: str = "UNIFIED", coin: str = None
    ) -> Dict:
        """Get wallet balance"""
        params = {"accountType": account_type}
        if coin:
            params["coin"] = coin
        return await self._request(
            "GET", "/v5/account/wallet-balance", params, signed=True, category=None
        )

    async def get_balance(self, account_type: str = "UNIFIED") -> Dict:
        """Get balance for all assets (compatible with MEXC format)"""
        response = await self.get_wallet_balance(account_type)

        # Convert to MEXC-like format
        balances = {}
        wallet_list = response.get("result", {}).get("list", [])

        for wallet in wallet_list:
            coins = wallet.get("coin", [])
            for coin in coins:
                asset = coin["coin"]
                balances[asset] = {
                    "free": float(coin["availableToWithdraw"]),
                    "locked": float(coin["locked"]),
                }

        return balances

    # Trading Methods
    async def create_order(
        self,
        symbol: str,
        side: str,  # Buy or Sell
        order_type: str,  # Market, Limit
        quantity: str,
        price: str = None,
        time_in_force: str = "GTC",
        category: str = "spot",
    ) -> Dict:
        """Create a new order"""
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": quantity,
            "timeInForce": time_in_force,
        }

        if order_type == "Limit" and price:
            params["price"] = price

        return await self._request(
            "POST", "/v5/order/create", params, signed=True, category=category
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str = None,
        order_link_id: str = None,
        category: str = "spot",
    ) -> Dict:
        """Cancel an active order"""
        params = {"category": category, "symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif order_link_id:
            params["orderLinkId"] = order_link_id
        else:
            raise ValueError("Either order_id or order_link_id must be provided")

        return await self._request(
            "POST", "/v5/order/cancel", params, signed=True, category=category
        )

    async def get_order(
        self,
        symbol: str,
        order_id: str = None,
        order_link_id: str = None,
        category: str = "spot",
    ) -> Dict:
        """Get order details"""
        params = {"category": category, "symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif order_link_id:
            params["orderLinkId"] = order_link_id

        return await self._request(
            "GET", "/v5/order/realtime", params, signed=True, category=category
        )

    async def get_open_orders(
        self, symbol: str = None, category: str = "spot"
    ) -> List[Dict]:
        """Get all open orders"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol

        response = await self._request(
            "GET", "/v5/order/realtime", params, signed=True, category=category
        )
        return response.get("result", {}).get("list", [])

    async def get_order_history(
        self, symbol: str = None, limit: int = 50, category: str = "spot"
    ) -> List[Dict]:
        """Get order history"""
        params = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol

        response = await self._request(
            "GET", "/v5/order/history", params, signed=True, category=category
        )
        return response.get("result", {}).get("list", [])

    async def get_execution_history(
        self, symbol: str = None, limit: int = 50, category: str = "spot"
    ) -> List[Dict]:
        """Get execution history (filled orders)"""
        params = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol

        response = await self._request(
            "GET", "/v5/execution/list", params, signed=True, category=category
        )
        return response.get("result", {}).get("list", [])

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()


class BybitTradeExecutor:
    """Trade execution handler for Bybit"""

    def __init__(self, client: BybitClient):
        self.client = client

    async def execute_market_order(
        self, symbol: str, side: str, quantity: str, category: str = "spot"
    ) -> Dict:
        """Execute market order"""
        try:
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                order_type="Market",
                quantity=quantity,
                category=category,
            )
            logger.info(f"Market order executed: {order}")
            return order
        except Exception as e:
            logger.error(f"Failed to execute market order: {str(e)}")
            raise

    async def execute_limit_order(
        self, symbol: str, side: str, quantity: str, price: str, category: str = "spot"
    ) -> Dict:
        """Execute limit order"""
        try:
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                order_type="Limit",
                quantity=quantity,
                price=price,
                category=category,
            )
            logger.info(f"Limit order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Failed to place limit order: {str(e)}")
            raise

    async def place_stop_loss(
        self,
        symbol: str,
        quantity: str,
        stop_price: str,
        side: str = "Sell",
        category: str = "spot",
    ) -> Dict:
        """Place stop loss order"""
        try:
            # Bybit supports conditional orders for stop-loss
            params = {
                "category": category,
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": quantity,
                "stopLoss": stop_price,
                "timeInForce": "GTC",
            }

            order = await self.client._request(
                "POST", "/v5/order/create", params, signed=True, category=category
            )
            logger.info(f"Stop-loss order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Failed to place stop-loss order: {str(e)}")
            raise
