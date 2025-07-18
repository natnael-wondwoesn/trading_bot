#!/usr/bin/env python3
"""
Multi-Exchange Data Service
Initializes and manages data feeds from multiple exchanges (MEXC, Bybit)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd

from config.config import Config
from mexc.mexc_client import MEXCClient
from mexc.data_feed import MEXCDataFeed
from bybit.bybit_client import BybitClient
from services.trading_orchestrator import trading_orchestrator

logger = logging.getLogger(__name__)


class MultiExchangeDataService:
    """Production market data service using multiple exchanges"""

    def __init__(self):
        # Exchange clients
        self.mexc_client = None
        self.bybit_client = None

        # Data feeds
        self.mexc_data_feed = None
        self.bybit_data_feed = None

        # Trading pairs to monitor
        self.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "BNBUSDT"]

        # Data storage per exchange
        self.latest_prices = {"MEXC": {}, "Bybit": {}}
        self.kline_data = {"MEXC": {}, "Bybit": {}}

        # Exchange availability
        self.exchange_status = {
            "MEXC": {"available": False, "initialized": False, "error": None},
            "Bybit": {"available": False, "initialized": False, "error": None},
        }

        # Control flags
        self.running = False
        self.initialized = False

        # Signal rate limiting - prevent overwhelming users
        self.last_signal_time = {}  # Track last signal time per symbol per exchange
        self.signal_cooldown_seconds = getattr(
            Config, "SIGNAL_CHECK_INTERVAL", 300
        )  # 5 minutes default

        # Performance tracking
        self.stats = {
            "mexc_updates": 0,
            "bybit_updates": 0,
            "signals_generated": 0,
            "signals_throttled": 0,
            "errors": 0,
            "start_time": None,
        }

    async def initialize(self):
        """Initialize the multi-exchange data service"""
        if self.initialized:
            return

        logger.info("Initializing Multi-Exchange Data Service...")

        # Initialize each exchange
        await self._initialize_mexc()
        await self._initialize_bybit()

        # Check if at least one exchange is available
        available_exchanges = [
            name for name, status in self.exchange_status.items() if status["available"]
        ]

        if not available_exchanges:
            raise Exception("No exchanges are available! Check API configurations.")

        logger.info(
            f"Multi-Exchange Data Service initialized with: {', '.join(available_exchanges)}"
        )
        self.initialized = True

    async def _initialize_mexc(self):
        """Initialize MEXC exchange"""
        try:
            logger.info("Initializing MEXC exchange...")

            # Check if API keys are configured
            if not Config.MEXC_API_KEY or not Config.MEXC_API_SECRET:
                logger.warning(
                    "MEXC API keys not configured, skipping MEXC initialization"
                )
                self.exchange_status["MEXC"]["error"] = "API keys not configured"
                return

            # Create MEXC client
            self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)

            # Test connectivity
            exchange_info = await self.mexc_client.get_exchange_info()
            logger.info(
                f"[OK] Connected to MEXC API (Server time: {datetime.fromtimestamp(exchange_info['serverTime']/1000)})"
            )

            # Verify trading pairs
            available_symbols = {s["symbol"] for s in exchange_info["symbols"]}
            valid_pairs = []

            for pair in self.trading_pairs:
                if pair in available_symbols:
                    valid_pairs.append(pair)
                    logger.info(f"[OK] MEXC {pair}: Available for trading")
                else:
                    logger.warning(f"[WARN] MEXC {pair}: Not available")

            if valid_pairs:
                # Load historical data
                for pair in valid_pairs:
                    try:
                        klines = await self.mexc_client.get_klines(pair, "1h", 100)
                        self.kline_data["MEXC"][pair] = klines

                        ticker = await self.mexc_client.get_ticker(pair)
                        self.latest_prices["MEXC"][pair] = float(ticker["lastPrice"])

                        logger.info(
                            f"[OK] MEXC {pair}: Loaded {len(klines)} candles, current price: ${self.latest_prices['MEXC'][pair]:,.4f}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to load MEXC data for {pair}: {e}")

                # Create data feed
                self.mexc_data_feed = MEXCDataFeed(self.mexc_client)

                self.exchange_status["MEXC"]["available"] = True
                self.exchange_status["MEXC"]["initialized"] = True
                logger.info(
                    f"✅ MEXC initialized successfully with {len(valid_pairs)} pairs"
                )

        except Exception as e:
            logger.error(f"Failed to initialize MEXC: {e}")
            self.exchange_status["MEXC"]["error"] = str(e)

    async def _initialize_bybit(self):
        """Initialize Bybit exchange"""
        try:
            logger.info("Initializing Bybit exchange...")

            # Check if API keys are configured
            if not Config.BYBIT_API_KEY or not Config.BYBIT_API_SECRET:
                logger.warning(
                    "Bybit API keys not configured, skipping Bybit initialization"
                )
                self.exchange_status["Bybit"]["error"] = "API keys not configured"
                return

            # Create Bybit client
            self.bybit_client = BybitClient(
                Config.BYBIT_API_KEY,
                Config.BYBIT_API_SECRET,
                testnet=Config.BYBIT_TESTNET,
            )

            # Test connectivity
            server_time = await self.bybit_client.get_server_time()
            time_second = int(
                server_time["result"]["timeSecond"]
            )  # Convert string to int
            logger.info(
                f"[OK] Connected to Bybit API (Server time: {datetime.fromtimestamp(time_second)})"
            )

            # Get instruments info to verify pairs
            instruments_response = await self.bybit_client.get_instruments_info(
                category="spot"
            )
            available_symbols = {
                item["symbol"]
                for item in instruments_response.get("result", {}).get("list", [])
            }

            valid_pairs = []

            for pair in self.trading_pairs:
                if pair in available_symbols:
                    valid_pairs.append(pair)
                    logger.info(f"[OK] Bybit {pair}: Available for trading")
                else:
                    logger.warning(f"[WARN] Bybit {pair}: Not available")

            if valid_pairs:
                # Load historical data
                for pair in valid_pairs:
                    try:
                        klines = await self.bybit_client.get_klines(
                            pair, "1h", 100, category="spot"
                        )
                        if len(klines) > 0:
                            self.kline_data["Bybit"][pair] = klines

                            # Get current price
                            ticker = await self.bybit_client.get_ticker(
                                pair, category="spot"
                            )
                            if ticker:
                                self.latest_prices["Bybit"][pair] = float(
                                    ticker["lastPrice"]
                                )
                                logger.info(
                                    f"[OK] Bybit {pair}: Loaded {len(klines)} candles, current price: ${self.latest_prices['Bybit'][pair]:,.4f}"
                                )

                    except Exception as e:
                        logger.error(f"Failed to load Bybit data for {pair}: {e}")

                self.exchange_status["Bybit"]["available"] = True
                self.exchange_status["Bybit"]["initialized"] = True
                logger.info(
                    f"✅ Bybit initialized successfully with {len(valid_pairs)} pairs"
                )

        except Exception as e:
            logger.error(f"Failed to initialize Bybit: {e}")
            self.exchange_status["Bybit"]["error"] = str(e)

            # Close client if it was created
            if self.bybit_client:
                try:
                    await self.bybit_client.close()
                except:
                    pass

    async def start(self):
        """Start the real-time data feeds"""
        if not self.initialized:
            await self.initialize()

        if self.running:
            return

        logger.info("Starting multi-exchange real-time data feeds...")

        try:
            self.running = True
            self.stats["start_time"] = datetime.now()

            # Start MEXC data feed if available
            if self.exchange_status["MEXC"]["available"] and self.mexc_data_feed:
                try:
                    mexc_pairs = list(self.kline_data["MEXC"].keys())
                    if mexc_pairs:
                        await self.mexc_data_feed.start(mexc_pairs, "1m")

                        for pair in mexc_pairs:
                            self.mexc_data_feed.subscribe(
                                pair,
                                "kline_closed",
                                lambda symbol, data: self.on_mexc_kline_update(
                                    symbol, data
                                ),
                            )
                            self.mexc_data_feed.subscribe(
                                pair,
                                "ticker_update",
                                lambda symbol, data: self.on_mexc_ticker_update(
                                    symbol, data
                                ),
                            )

                        logger.info(
                            f"[OK] MEXC real-time feed started for {len(mexc_pairs)} pairs"
                        )
                except Exception as e:
                    logger.error(f"Failed to start MEXC data feed: {e}")

            # Start periodic tasks
            asyncio.create_task(self._periodic_data_refresh())
            asyncio.create_task(self._exchange_health_monitor())

            available_exchanges = [
                name
                for name, status in self.exchange_status.items()
                if status["available"]
            ]
            logger.info(
                f"[OK] Multi-exchange data service started with: {', '.join(available_exchanges)}"
            )

        except Exception as e:
            logger.error(f"Failed to start multi-exchange data service: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop all data feeds"""
        logger.info("Stopping multi-exchange data service...")

        self.running = False

        try:
            # Stop MEXC data feed
            if self.mexc_data_feed:
                try:
                    await self.mexc_data_feed.stop()
                except Exception as e:
                    logger.error(f"Error stopping MEXC data feed: {e}")

            # Close MEXC client
            if (
                self.mexc_client
                and hasattr(self.mexc_client, "session")
                and self.mexc_client.session
            ):
                try:
                    await self.mexc_client.session.close()
                except Exception as e:
                    logger.error(f"Error closing MEXC client: {e}")

            # Close Bybit client
            if self.bybit_client:
                try:
                    await self.bybit_client.close()
                except Exception as e:
                    logger.error(f"Error closing Bybit client: {e}")

            logger.info("[OK] Multi-exchange data service stopped")

        except Exception as e:
            logger.error(f"Error stopping multi-exchange data service: {e}")

    async def on_mexc_kline_update(self, symbol: str, kline_data):
        """Handle MEXC kline updates"""
        try:
            if not self.running:
                return

            # Update stored data
            if hasattr(kline_data, "iloc") and len(kline_data) > 0:
                self.kline_data["MEXC"][symbol] = kline_data
                latest_candle = kline_data.iloc[-1]
                self.latest_prices["MEXC"][symbol] = float(latest_candle["close"])

                # Create market data package
                market_data = {
                    "symbol": symbol,
                    "exchange": "MEXC",
                    "price": float(latest_candle["close"]),
                    "volume": float(latest_candle["volume"]),
                    "high": float(latest_candle["high"]),
                    "low": float(latest_candle["low"]),
                    "open": float(latest_candle["open"]),
                    "timestamp": datetime.now().isoformat(),
                    "kline_data": kline_data,
                    "data_source": "mexc_realtime",
                }

                # Rate limiting: Only process signals if enough time has passed
                current_time = datetime.now()
                signal_key = f"MEXC_{symbol}"
                last_signal_time = self.last_signal_time.get(signal_key)

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
                        self.last_signal_time[signal_key] = current_time
                        logger.debug(
                            f"📊 MEXC {symbol}: Generated {len(signals)} signals"
                        )
                else:
                    # Signal throttled
                    self.stats["signals_throttled"] += 1
                    time_remaining = (
                        self.signal_cooldown_seconds
                        - (current_time - last_signal_time).total_seconds()
                    )
                    logger.debug(
                        f"📊 MEXC {symbol}: Signal throttled, {time_remaining:.0f}s remaining"
                    )

                self.stats["mexc_updates"] += 1

        except Exception as e:
            logger.error(f"Error processing MEXC kline update for {symbol}: {e}")
            self.stats["errors"] += 1

    async def on_mexc_ticker_update(self, symbol: str, ticker_data):
        """Handle MEXC ticker updates"""
        try:
            if not self.running:
                return

            if isinstance(ticker_data, dict) and "lastPrice" in ticker_data:
                self.latest_prices["MEXC"][symbol] = float(ticker_data["lastPrice"])

        except Exception as e:
            logger.error(f"Error processing MEXC ticker update for {symbol}: {e}")

    async def _periodic_data_refresh(self):
        """Periodic data refresh for both exchanges"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Refresh MEXC data
                if self.exchange_status["MEXC"]["available"]:
                    await self._refresh_mexc_data()

                # Refresh Bybit data
                if self.exchange_status["Bybit"]["available"]:
                    await self._refresh_bybit_data()

            except Exception as e:
                logger.error(f"Error in periodic data refresh: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _refresh_mexc_data(self):
        """Refresh MEXC market data"""
        try:
            for symbol in self.kline_data["MEXC"].keys():
                ticker = await self.mexc_client.get_ticker(symbol)
                self.latest_prices["MEXC"][symbol] = float(ticker["lastPrice"])
        except Exception as e:
            logger.error(f"Error refreshing MEXC data: {e}")

    async def _refresh_bybit_data(self):
        """Refresh Bybit market data"""
        try:
            for symbol in self.kline_data["Bybit"].keys():
                ticker = await self.bybit_client.get_ticker(symbol, category="spot")
                if ticker:
                    self.latest_prices["Bybit"][symbol] = float(ticker["lastPrice"])
        except Exception as e:
            logger.error(f"Error refreshing Bybit data: {e}")

    async def _exchange_health_monitor(self):
        """Monitor exchange health and connectivity"""
        while self.running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes

                # Check MEXC health
                if self.exchange_status["MEXC"]["available"]:
                    try:
                        await self.mexc_client.get_server_time()
                        logger.debug("MEXC health check: OK")
                    except Exception as e:
                        logger.warning(f"MEXC health check failed: {e}")

                # Check Bybit health
                if self.exchange_status["Bybit"]["available"]:
                    try:
                        await self.bybit_client.get_server_time()
                        logger.debug("Bybit health check: OK")
                    except Exception as e:
                        logger.warning(f"Bybit health check failed: {e}")

            except Exception as e:
                logger.error(f"Error in exchange health monitor: {e}")
                await asyncio.sleep(60)

    def get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """Get current prices from all exchanges"""
        return self.latest_prices.copy()

    def get_exchange_status(self) -> Dict[str, Dict]:
        """Get status of all exchanges"""
        return self.exchange_status.copy()

    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        uptime = (
            (datetime.now() - self.stats["start_time"]).total_seconds()
            if self.stats["start_time"]
            else 0
        )

        return {
            "running": self.running,
            "exchanges": self.exchange_status,
            "mexc_updates": self.stats["mexc_updates"],
            "bybit_updates": self.stats["bybit_updates"],
            "signals_generated": self.stats["signals_generated"],
            "errors": self.stats["errors"],
            "uptime_seconds": uptime,
            "trading_pairs": self.trading_pairs,
        }

    async def force_signal_generation(self):
        """Force signal generation for all exchanges and pairs"""
        logger.info("Forcing signal generation for all exchanges...")

        signals_generated = 0

        # Generate signals for MEXC pairs
        if self.exchange_status["MEXC"]["available"]:
            for symbol, kline_data in self.kline_data["MEXC"].items():
                try:
                    if len(kline_data) > 0:
                        latest_candle = kline_data.iloc[-1]
                        market_data = {
                            "symbol": symbol,
                            "exchange": "MEXC",
                            "price": float(latest_candle["close"]),
                            "kline_data": kline_data,
                            "data_source": "forced_generation",
                            "timestamp": datetime.now().isoformat(),
                        }

                        signals = await trading_orchestrator.process_market_signal(
                            symbol, market_data
                        )
                        signals_generated += len(signals) if signals else 0

                except Exception as e:
                    logger.error(f"Error forcing signals for MEXC {symbol}: {e}")

        # Generate signals for Bybit pairs
        if self.exchange_status["Bybit"]["available"]:
            for symbol, kline_data in self.kline_data["Bybit"].items():
                try:
                    if len(kline_data) > 0:
                        latest_candle = kline_data.iloc[-1]
                        market_data = {
                            "symbol": symbol,
                            "exchange": "Bybit",
                            "price": float(latest_candle["close"]),
                            "kline_data": kline_data,
                            "data_source": "forced_generation",
                            "timestamp": datetime.now().isoformat(),
                        }

                        signals = await trading_orchestrator.process_market_signal(
                            symbol, market_data
                        )
                        signals_generated += len(signals) if signals else 0

                except Exception as e:
                    logger.error(f"Error forcing signals for Bybit {symbol}: {e}")

        logger.info(
            f"Force signal generation completed: {signals_generated} signals generated"
        )
        return signals_generated


# Global instance
multi_exchange_data_service = MultiExchangeDataService()
