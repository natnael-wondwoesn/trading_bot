#!/usr/bin/env python3
"""
Real Market Data Service
Integrates MEXC real-time data with the trading orchestrator for production use
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from config.config import Config
from mexc.mexc_client import MEXCClient
from mexc.data_feed import MEXCDataFeed
from services.trading_orchestrator import trading_orchestrator

logger = logging.getLogger(__name__)


class RealMarketDataService:
    """Production market data service using real MEXC data"""

    def __init__(self):
        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
        self.data_feed = MEXCDataFeed(self.mexc_client)

        # Trading pairs to monitor
        self.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "BNBUSDT"]

        # Data storage
        self.latest_prices = {}
        self.kline_data = {}

        # Control flags
        self.running = False
        self.initialized = False

        # Signal rate limiting - prevent overwhelming users
        self.last_signal_time = {}  # Track last signal time per symbol
        self.signal_cooldown_seconds = getattr(
            Config, "SIGNAL_CHECK_INTERVAL", 300
        )  # 5 minutes default

        # Performance tracking
        self.stats = {
            "data_updates": 0,
            "signals_generated": 0,
            "signals_throttled": 0,
            "errors": 0,
            "start_time": None,
        }

    async def initialize(self):
        """Initialize the real market data service"""
        if self.initialized:
            return

        logger.info("Initializing Real Market Data Service...")

        try:
            # Test API connectivity
            exchange_info = await self.mexc_client.get_exchange_info()
            logger.info(
                f"[OK] Connected to MEXC API (Server time: {datetime.fromtimestamp(exchange_info['serverTime']/1000)})"
            )

            # Verify trading pairs are available
            available_symbols = {s["symbol"] for s in exchange_info["symbols"]}
            valid_pairs = []

            for pair in self.trading_pairs:
                if pair in available_symbols:
                    valid_pairs.append(pair)
                    logger.info(f"[OK] {pair}: Available for trading")
                else:
                    logger.warning(f"[WARN] {pair}: Not available on MEXC")

            self.trading_pairs = valid_pairs

            if not self.trading_pairs:
                raise Exception("No valid trading pairs found!")

            # Initialize historical data for each pair
            logger.info("Loading historical data...")
            for pair in self.trading_pairs:
                try:
                    # Get initial historical data (last 100 1-hour candles)
                    klines = await self.mexc_client.get_klines(pair, "1h", 100)
                    self.kline_data[pair] = klines

                    # Get current price
                    ticker = await self.mexc_client.get_ticker(pair)
                    self.latest_prices[pair] = float(ticker["lastPrice"])

                    logger.info(
                        f"[OK] {pair}: Loaded {len(klines)} candles, current price: ${self.latest_prices[pair]:,.4f}"
                    )

                except Exception as e:
                    logger.error(f"Failed to load data for {pair}: {e}")
                    self.trading_pairs.remove(pair)

            self.initialized = True
            logger.info(
                f"Real Market Data Service initialized with {len(self.trading_pairs)} pairs"
            )

        except Exception as e:
            logger.error(f"Failed to initialize market data service: {e}")
            raise

    async def start(self):
        """Start the real-time data feed"""
        if not self.initialized:
            await self.initialize()

        if self.running:
            return

        logger.info("Starting real-time market data feed...")

        try:
            self.running = True
            self.stats["start_time"] = datetime.now()

            # Start WebSocket data feed
            await self.data_feed.start(self.trading_pairs, "1m")

            # Subscribe to market data updates
            for pair in self.trading_pairs:
                # Subscribe to kline updates (for strategy analysis)
                self.data_feed.subscribe(pair, "kline_closed", self.on_kline_update)

                # Subscribe to ticker updates (for price monitoring)
                self.data_feed.subscribe(pair, "ticker_update", self.on_ticker_update)

            # Start periodic data refresh task
            asyncio.create_task(self._periodic_data_refresh())

            logger.info(
                f"[OK] Real-time data feed started for {len(self.trading_pairs)} pairs"
            )

        except Exception as e:
            logger.error(f"Failed to start market data service: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop the market data service"""
        logger.info("Stopping real market data service...")

        self.running = False

        try:
            if self.data_feed:
                await self.data_feed.stop()

            if self.mexc_client and self.mexc_client.session:
                await self.mexc_client.session.close()

            logger.info("[OK] Market data service stopped")

        except Exception as e:
            logger.error(f"Error stopping market data service: {e}")

    async def on_kline_update(self, symbol: str, kline_data):
        """Handle new kline (candlestick) data"""
        try:
            if not self.running:
                return

            # Update stored kline data
            if hasattr(kline_data, "iloc") and len(kline_data) > 0:
                self.kline_data[symbol] = kline_data
                latest_candle = kline_data.iloc[-1]

                # Update latest price
                self.latest_prices[symbol] = float(latest_candle["close"])

                # Create market data package for orchestrator
                market_data = {
                    "symbol": symbol,
                    "price": float(latest_candle["close"]),
                    "volume": float(latest_candle["volume"]),
                    "high": float(latest_candle["high"]),
                    "low": float(latest_candle["low"]),
                    "open": float(latest_candle["open"]),
                    "timestamp": datetime.now().isoformat(),
                    "kline_data": kline_data,  # Full historical data for strategy analysis
                    "data_source": "mexc_realtime",
                }

                # Rate limiting: Only process signals if enough time has passed
                current_time = datetime.now()
                last_signal_time = self.last_signal_time.get(symbol)

                if (
                    last_signal_time is None
                    or (current_time - last_signal_time).total_seconds()
                    >= self.signal_cooldown_seconds
                ):

                    # Send to trading orchestrator
                    signals = await trading_orchestrator.process_market_signal(
                        symbol, market_data
                    )

                    # Update statistics
                    if signals:
                        self.stats["signals_generated"] += len(signals)
                        self.last_signal_time[symbol] = current_time
                        logger.info(
                            f"📊 {symbol}: Generated {len(signals)} signals @ ${self.latest_prices[symbol]:,.4f}"
                        )
                else:
                    # Signal throttled
                    self.stats["signals_throttled"] += 1
                    time_remaining = (
                        self.signal_cooldown_seconds
                        - (current_time - last_signal_time).total_seconds()
                    )
                    logger.debug(
                        f"📊 {symbol}: Signal throttled, {time_remaining:.0f}s remaining"
                    )

                self.stats["data_updates"] += 1

        except Exception as e:
            logger.error(f"Error processing kline update for {symbol}: {e}")
            self.stats["errors"] += 1

    async def on_ticker_update(self, symbol: str, ticker_data):
        """Handle ticker price updates"""
        try:
            if not self.running:
                return

            if isinstance(ticker_data, dict) and "c" in ticker_data:
                new_price = float(ticker_data["c"])
                old_price = self.latest_prices.get(symbol, new_price)

                # Update price
                self.latest_prices[symbol] = new_price

                # Log significant price changes (> 0.5%)
                if old_price > 0:
                    change_percent = ((new_price - old_price) / old_price) * 100
                    if abs(change_percent) > 0.5:
                        direction = "📈" if change_percent > 0 else "📉"
                        logger.info(
                            f"{direction} {symbol}: ${new_price:,.4f} ({change_percent:+.2f}%)"
                        )

        except Exception as e:
            logger.error(f"Error processing ticker update for {symbol}: {e}")

    async def _periodic_data_refresh(self):
        """Periodically refresh historical data to ensure accuracy"""
        while self.running:
            try:
                # Refresh data every 5 minutes
                await asyncio.sleep(300)

                if not self.running:
                    break

                # Refresh kline data for each pair
                for pair in self.trading_pairs:
                    try:
                        # Get fresh historical data
                        fresh_klines = await self.mexc_client.get_klines(
                            pair, "1h", 100
                        )

                        # Merge with existing data to avoid gaps
                        if pair in self.kline_data and len(fresh_klines) > 0:
                            # Update with fresh data
                            self.kline_data[pair] = fresh_klines

                    except Exception as e:
                        logger.error(f"Error refreshing data for {pair}: {e}")

                logger.debug("Periodic data refresh completed")

            except Exception as e:
                logger.error(f"Error in periodic data refresh: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all monitored pairs"""
        return self.latest_prices.copy()

    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get latest kline data for a specific symbol"""
        return self.kline_data.get(symbol)

    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        uptime = 0
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        return {
            "running": self.running,
            "initialized": self.initialized,
            "trading_pairs": len(self.trading_pairs),
            "data_updates": self.stats["data_updates"],
            "signals_generated": self.stats["signals_generated"],
            "errors": self.stats["errors"],
            "uptime_seconds": uptime,
            "latest_prices": self.latest_prices,
        }

    async def force_signal_generation(self):
        """Force signal generation for all pairs (for testing)"""
        logger.info("Forcing signal generation for all pairs...")

        for symbol in self.trading_pairs:
            if symbol in self.kline_data:
                await self.on_kline_update(symbol, self.kline_data[symbol])

        logger.info("Signal generation completed")


# Global instance
real_market_data_service = RealMarketDataService()
