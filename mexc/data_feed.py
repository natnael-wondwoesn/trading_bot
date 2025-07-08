import asyncio
import pandas as pd
from typing import Dict, Callable, List
from datetime import datetime
import logging
from mexc.mexc_client import MEXCClient

logger = logging.getLogger(__name__)


class MEXCDataFeed:
    """Real-time data feed from MEXC"""

    def __init__(self, client: MEXCClient):
        self.client = client
        self.subscribers = {}
        self.kline_data = {}
        self.orderbook_data = {}
        self.running = False

    async def start(self, symbols: List[str], interval: str = "1m"):
        """Start real-time data feed"""
        self.running = True

        # Subscribe to WebSocket streams
        await self.client.start_websocket()

        for symbol in symbols:
            await self.client.subscribe_klines(symbol, interval)
            await self.client.subscribe_ticker(symbol)
            await self.client.subscribe_orderbook(symbol)

            # Initialize with historical data
            self.kline_data[symbol] = await self.client.get_klines(
                symbol, interval, 100
            )

        # Start processing messages
        asyncio.create_task(self._process_websocket_messages())
        logger.info(f"Data feed started for symbols: {symbols}")

    async def _process_websocket_messages(self):
        """Process incoming WebSocket messages"""
        try:
            async for message in self.client.read_websocket():
                if "data" in message:
                    await self._handle_message(message)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            if self.running:
                # Reconnect
                await asyncio.sleep(5)
                await self.start(list(self.kline_data.keys()))

    async def _handle_message(self, message: Dict):
        """Handle different types of WebSocket messages"""
        data = message.get("data", {})

        if "s" in data:  # Symbol field
            symbol = data["s"]

            # Kline update
            if "k" in data:
                await self._update_kline(symbol, data["k"])

            # Ticker update
            elif "c" in data:  # Current price
                await self._update_ticker(symbol, data)

            # Order book update
            elif "asks" in data or "bids" in data:
                await self._update_orderbook(symbol, data)

    async def _update_kline(self, symbol: str, kline_data: Dict):
        """Update kline data"""
        if symbol not in self.kline_data:
            return

        # Convert kline data to DataFrame row
        new_row = pd.DataFrame(
            [
                {
                    "open": float(kline_data["o"]),
                    "high": float(kline_data["h"]),
                    "low": float(kline_data["l"]),
                    "close": float(kline_data["c"]),
                    "volume": float(kline_data["v"]),
                }
            ],
            index=[pd.to_datetime(kline_data["t"], unit="ms")],
        )

        # Update or append
        if kline_data["x"]:  # Kline closed
            self.kline_data[symbol] = pd.concat([self.kline_data[symbol], new_row])
            self.kline_data[symbol] = self.kline_data[symbol].last(
                "100D"
            )  # Keep last 100 candles

            # Notify subscribers
            await self._notify_subscribers(symbol, "kline_closed")
        else:
            # Update current candle
            self.kline_data[symbol].iloc[-1] = new_row.iloc[0]

    async def _update_ticker(self, symbol: str, ticker_data: Dict):
        """Update ticker data"""
        # Notify subscribers of price update
        await self._notify_subscribers(symbol, "ticker_update", ticker_data)

    async def _update_orderbook(self, symbol: str, orderbook_data: Dict):
        """Update order book data"""
        self.orderbook_data[symbol] = {
            "asks": orderbook_data.get("asks", []),
            "bids": orderbook_data.get("bids", []),
            "timestamp": datetime.now(),
        }
        await self._notify_subscribers(symbol, "orderbook_update")

    def subscribe(self, symbol: str, event_type: str, callback: Callable):
        """Subscribe to data updates"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = {}
        if event_type not in self.subscribers[symbol]:
            self.subscribers[symbol][event_type] = []
        self.subscribers[symbol][event_type].append(callback)

    async def _notify_subscribers(
        self, symbol: str, event_type: str, data: Dict = None
    ):
        """Notify subscribers of updates"""
        if symbol in self.subscribers and event_type in self.subscribers[symbol]:
            for callback in self.subscribers[symbol][event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, data or self.get_latest_data(symbol))
                    else:
                        callback(symbol, data or self.get_latest_data(symbol))
                except Exception as e:
                    logger.error(f"Subscriber callback error: {str(e)}")

    def get_latest_data(self, symbol: str) -> pd.DataFrame:
        """Get latest kline data for symbol"""
        return self.kline_data.get(symbol, pd.DataFrame())

    def get_orderbook(self, symbol: str) -> Dict:
        """Get latest order book data"""
        return self.orderbook_data.get(symbol, {})

    async def stop(self):
        """Stop data feed"""
        self.running = False
        await self.client.close()
