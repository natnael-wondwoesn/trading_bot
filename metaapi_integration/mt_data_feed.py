import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Callable
import logging

logger = logging.getLogger(__name__)


class MTDataFeed:
    """Real-time data feed from MetaTrader via MetaAPI"""

    def __init__(self, client):
        self.client = client
        self.subscribers = {}
        self.price_data = {}
        self.candle_data = {}
        self.running = False

    async def start(self, symbols: List[str], timeframe: str = "1H"):
        """Start data feed for symbols"""
        self.running = True

        # Subscribe to real-time data
        await self._subscribe_to_streams(symbols)

        # Load initial historical data
        for symbol in symbols:
            candles = await self.client.get_candles(symbol, timeframe, 100)
            if candles:
                self.candle_data[symbol] = self._candles_to_dataframe(candles, symbol)

        # Start price update loop
        asyncio.create_task(self._update_loop())

        logger.info(f"Data feed started for {len(symbols)} symbols")

    def _candles_to_dataframe(self, candles: List[Dict], symbol: str) -> pd.DataFrame:
        """Convert candles to DataFrame"""
        df = pd.DataFrame(candles)

        # Rename columns to match expected format
        df = df.rename(
            columns={
                "time": "timestamp",
                "brokerTime": "broker_time",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tickVolume": "volume",
                "spread": "spread",
                "volume": "real_volume",
            }
        )

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.attrs["pair"] = symbol
        return df

    async def _subscribe_to_streams(self, symbols: List[str]):
        """Subscribe to real-time price updates"""
        if not self.client.streaming_connection:
            await self.client.streaming_connection.connect()
            await self.client.streaming_connection.wait_synchronized()

        # Add listener for price updates
        listener = {
            "on_symbol_price_updated": self._on_price_update,
            "on_candles": self._on_candle_update,
            "on_tick": self._on_tick_update,
        }

        self.client.streaming_connection.add_synchronization_listener(listener)

        # Subscribe to symbols
        await self.client.streaming_connection.subscribe_to_market_data(
            symbols, ["quotes", "candles", "ticks"]
        )

    async def _on_price_update(self, account_id: str, prices: List[Dict]):
        """Handle price updates"""
        for price in prices:
            symbol = price["symbol"]
            self.price_data[symbol] = {
                "bid": price["bid"],
                "ask": price["ask"],
                "time": datetime.now(),
            }

            # Notify subscribers
            await self._notify_subscribers(symbol, "price_update", price)

    async def _on_candle_update(self, account_id: str, candles: List[Dict]):
        """Handle candle updates"""
        for candle in candles:
            symbol = candle["symbol"]

            if symbol in self.candle_data:
                # Update or append candle
                new_candle = pd.DataFrame(
                    [
                        {
                            "open": candle["open"],
                            "high": candle["high"],
                            "low": candle["low"],
                            "close": candle["close"],
                            "volume": candle.get("tickVolume", 0),
                        }
                    ],
                    index=[pd.to_datetime(candle["time"])],
                )

                # Check if candle exists
                if candle["time"] in self.candle_data[symbol].index:
                    # Update existing candle
                    self.candle_data[symbol].loc[candle["time"]] = new_candle.iloc[0]
                else:
                    # Append new candle
                    self.candle_data[symbol] = pd.concat(
                        [self.candle_data[symbol], new_candle]
                    )

                    # Keep only last 200 candles
                    self.candle_data[symbol] = self.candle_data[symbol].tail(200)

                    # Notify on closed candle
                    await self._notify_subscribers(symbol, "candle_closed", candle)

    async def _on_tick_update(self, account_id: str, tick: Dict):
        """Handle tick updates"""
        symbol = tick["symbol"]
        await self._notify_subscribers(symbol, "tick", tick)

    async def _update_loop(self):
        """Periodic update loop"""
        while self.running:
            try:
                # Update prices for all symbols
                for symbol in self.price_data.keys():
                    price = await self.client.get_price(symbol)
                    if price:
                        self.price_data[symbol] = {
                            "bid": price.get("bid"),
                            "ask": price.get("ask"),
                            "time": datetime.now(),
                        }

                await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                logger.error(f"Update loop error: {str(e)}")
                await asyncio.sleep(10)

    def subscribe(self, symbol: str, event_type: str, callback: Callable):
        """Subscribe to data updates"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = {}
        if event_type not in self.subscribers[symbol]:
            self.subscribers[symbol][event_type] = []
        self.subscribers[symbol][event_type].append(callback)

    async def _notify_subscribers(self, symbol: str, event_type: str, data: Dict):
        """Notify subscribers of updates"""
        if symbol in self.subscribers and event_type in self.subscribers[symbol]:
            for callback in self.subscribers[symbol][event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, data)
                    else:
                        callback(symbol, data)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {str(e)}")

    def get_candles(self, symbol: str) -> pd.DataFrame:
        """Get latest candle data"""
        return self.candle_data.get(symbol, pd.DataFrame())

    def get_price(self, symbol: str) -> Dict:
        """Get latest price data"""
        return self.price_data.get(symbol, {})

    def calculate_spread(self, symbol: str) -> float:
        """Calculate current spread in pips"""
        price = self.get_price(symbol)
        if price and "bid" in price and "ask" in price:
            spread = price["ask"] - price["bid"]

            # Convert to pips based on symbol
            if "JPY" in symbol:
                return spread * 100
            else:
                return spread * 10000
        return 0

    async def stop(self):
        """Stop data feed"""
        self.running = False
        if self.client.streaming_connection:
            await self.client.streaming_connection.close()
