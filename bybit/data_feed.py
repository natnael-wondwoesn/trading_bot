#!/usr/bin/env python3
"""
Bybit Data Feed
Real-time market data feed for Bybit exchange using WebSocket connections
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, List, Callable, Optional
from datetime import datetime
import pandas as pd

from .bybit_client import BybitClient

logger = logging.getLogger(__name__)


class BybitDataFeed:
    """Real-time data feed for Bybit exchange"""

    def __init__(self, client: BybitClient):
        self.client = client
        self.ws_url = (
            "wss://stream.bybit.com/v5/public/spot"
            if not client.testnet
            else "wss://stream-testnet.bybit.com/v5/public/spot"
        )
        self.private_ws_url = (
            "wss://stream.bybit.com/v5/private"
            if not client.testnet
            else "wss://stream-testnet.bybit.com/v5/private"
        )

        self.websocket = None
        self.private_websocket = None
        self.subscriptions = {}
        self.callbacks = {}
        self.running = False
        self.heartbeat_interval = 20  # Bybit requires ping every 20 seconds

    async def start(self, symbols: List[str], interval: str = "1m"):
        """Start the data feed for specified symbols"""
        if self.running:
            return

        self.running = True
        logger.info(f"Starting Bybit data feed for symbols: {symbols}")

        try:
            # Connect to public WebSocket
            await self._connect_public_ws()

            # Subscribe to ticker and kline updates
            for symbol in symbols:
                await self._subscribe_ticker(symbol)
                await self._subscribe_kline(symbol, interval)

            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())

            # Start message processing loop
            asyncio.create_task(self._process_messages())

            logger.info("Bybit data feed started successfully")

        except Exception as e:
            logger.error(f"Failed to start Bybit data feed: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop the data feed"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping Bybit data feed...")

        if self.websocket:
            await self.websocket.close()

        if self.private_websocket:
            await self.private_websocket.close()

        logger.info("Bybit data feed stopped")

    async def _connect_public_ws(self):
        """Connect to Bybit public WebSocket"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("Connected to Bybit public WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to Bybit WebSocket: {e}")
            raise

    async def _connect_private_ws(self):
        """Connect to Bybit private WebSocket for account updates"""
        try:
            self.private_websocket = await websockets.connect(self.private_ws_url)

            # Authenticate private WebSocket
            auth_message = await self._get_auth_message()
            await self.private_websocket.send(json.dumps(auth_message))

            logger.info("Connected to Bybit private WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to Bybit private WebSocket: {e}")
            raise

    async def _get_auth_message(self) -> Dict:
        """Generate authentication message for private WebSocket"""
        import time
        import hmac
        import hashlib

        timestamp = int(time.time() * 1000)
        signature = hmac.new(
            self.client.api_secret.encode("utf-8"),
            f"GET/realtime{timestamp}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {"op": "auth", "args": [self.client.api_key, timestamp, signature]}

    async def _subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates for a symbol"""
        subscription = {"op": "subscribe", "args": [f"tickers.{symbol}"]}

        await self.websocket.send(json.dumps(subscription))
        self.subscriptions[f"tickers.{symbol}"] = True
        logger.info(f"Subscribed to ticker for {symbol}")

    async def _subscribe_kline(self, symbol: str, interval: str):
        """Subscribe to kline updates for a symbol"""
        # Convert interval to Bybit format
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

        subscription = {"op": "subscribe", "args": [f"kline.{bybit_interval}.{symbol}"]}

        await self.websocket.send(json.dumps(subscription))
        self.subscriptions[f"kline.{bybit_interval}.{symbol}"] = True
        logger.info(f"Subscribed to {interval} klines for {symbol}")

    async def _heartbeat_loop(self):
        """Send ping messages to keep connection alive"""
        while self.running:
            try:
                if self.websocket:
                    ping_message = {"op": "ping"}
                    await self.websocket.send(json.dumps(ping_message))

                if self.private_websocket:
                    ping_message = {"op": "ping"}
                    await self.private_websocket.send(json.dumps(ping_message))

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def _process_messages(self):
        """Process incoming WebSocket messages"""
        while self.running:
            try:
                if self.websocket:
                    message = await self.websocket.recv()
                    await self._handle_message(json.loads(message))

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Bybit WebSocket connection closed")
                if self.running:
                    await self._reconnect()
                break
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _handle_message(self, message: Dict):
        """Handle incoming WebSocket message"""
        try:
            if message.get("op") == "pong":
                # Heartbeat response
                return

            if message.get("op") == "subscribe":
                # Subscription confirmation
                logger.info(f"Subscription confirmed: {message}")
                return

            # Handle data updates
            topic = message.get("topic", "")
            data = message.get("data", {})

            if topic.startswith("tickers."):
                await self._handle_ticker_update(topic, data)
            elif topic.startswith("kline."):
                await self._handle_kline_update(topic, data)
            elif "order" in topic:
                await self._handle_order_update(topic, data)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_ticker_update(self, topic: str, data: Dict):
        """Handle ticker price updates"""
        try:
            symbol = topic.split(".")[-1]

            # Convert Bybit ticker format to standard format
            ticker_data = {
                "symbol": symbol,
                "price": float(data.get("lastPrice", 0)),
                "volume": float(data.get("volume24h", 0)),
                "change": float(data.get("price24hPcnt", 0)),
                "high": float(data.get("highPrice24h", 0)),
                "low": float(data.get("lowPrice24h", 0)),
                "timestamp": datetime.now().isoformat(),
            }

            # Trigger callbacks
            await self._trigger_callbacks(symbol, "ticker_update", ticker_data)

        except Exception as e:
            logger.error(f"Error processing ticker update: {e}")

    async def _handle_kline_update(self, topic: str, data: Dict):
        """Handle kline/candlestick updates"""
        try:
            parts = topic.split(".")
            interval = parts[1]
            symbol = parts[2]

            # Convert Bybit kline format to standard format
            for kline in data:
                kline_data = {
                    "symbol": symbol,
                    "interval": interval,
                    "open_time": int(kline.get("start", 0)),
                    "close_time": int(kline.get("end", 0)),
                    "open": float(kline.get("open", 0)),
                    "high": float(kline.get("high", 0)),
                    "low": float(kline.get("low", 0)),
                    "close": float(kline.get("close", 0)),
                    "volume": float(kline.get("volume", 0)),
                    "confirm": kline.get("confirm", False),
                    "timestamp": datetime.now().isoformat(),
                }

                # Trigger callbacks
                await self._trigger_callbacks(symbol, "kline_update", kline_data)

        except Exception as e:
            logger.error(f"Error processing kline update: {e}")

    async def _handle_order_update(self, topic: str, data: Dict):
        """Handle order status updates"""
        try:
            # Trigger callbacks for order updates
            await self._trigger_callbacks("orders", "order_update", data)

        except Exception as e:
            logger.error(f"Error processing order update: {e}")

    async def _trigger_callbacks(self, symbol: str, event_type: str, data: Dict):
        """Trigger registered callbacks for events"""
        callback_key = f"{symbol}_{event_type}"
        if callback_key in self.callbacks:
            try:
                callback = self.callbacks[callback_key]
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data)
                else:
                    callback(symbol, data)
            except Exception as e:
                logger.error(f"Error in callback {callback_key}: {e}")

    async def _reconnect(self):
        """Reconnect WebSocket connection"""
        logger.info("Attempting to reconnect Bybit WebSocket...")

        try:
            if self.websocket:
                await self.websocket.close()

            await asyncio.sleep(5)  # Wait before reconnecting
            await self._connect_public_ws()

            # Re-subscribe to all topics
            for topic in self.subscriptions:
                if topic.startswith("tickers."):
                    symbol = topic.split(".")[-1]
                    await self._subscribe_ticker(symbol)
                elif topic.startswith("kline."):
                    parts = topic.split(".")
                    interval = parts[1]
                    symbol = parts[2]
                    await self._subscribe_kline(symbol, interval)

            logger.info("Bybit WebSocket reconnected successfully")

        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")

    def subscribe(self, symbol: str, event_type: str, callback: Callable):
        """Subscribe to events for a symbol"""
        callback_key = f"{symbol}_{event_type}"
        self.callbacks[callback_key] = callback
        logger.info(f"Callback registered for {callback_key}")

    def unsubscribe(self, symbol: str, event_type: str):
        """Unsubscribe from events for a symbol"""
        callback_key = f"{symbol}_{event_type}"
        if callback_key in self.callbacks:
            del self.callbacks[callback_key]
            logger.info(f"Callback removed for {callback_key}")

    async def get_historical_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Get historical klines data"""
        return await self.client.get_klines(symbol, interval, limit)

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        ticker = await self.client.get_ticker(symbol)
        return float(ticker.get("lastPrice", 0))
