#!/usr/bin/env python3
"""
Live Signal Monitor - Enhanced Strategy
Continuously monitors all pairs for trading signals using the enhanced RSI EMA strategy
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LiveSignalMonitor:
    """Live signal monitoring system"""

    def __init__(self):
        self.running = False
        self.signals_found = []
        self.last_prices = {}

    async def initialize(self):
        """Initialize the monitoring system"""
        try:
            from strategy.strategies.enhanced_rsi_ema_strategy import (
                EnhancedRSIEMAStrategy,
            )
            from bybit.bybit_client import BybitClient
            from config.config import Config

            self.strategy = EnhancedRSIEMAStrategy()
            self.client = BybitClient(
                Config.BYBIT_API_KEY,
                Config.BYBIT_API_SECRET,
                testnet=Config.BYBIT_TESTNET,
            )
            self.pairs = Config.TRADING_PAIRS

            logger.info(f"🚀 Live Signal Monitor initialized")
            logger.info(f"📊 Strategy: {self.strategy.name}")
            logger.info(
                f"💰 Monitoring {len(self.pairs)} pairs: {', '.join(self.pairs)}"
            )
            logger.info(
                f"🎯 RSI Thresholds: Buy<{self.strategy.rsi_oversold}, Sell>{self.strategy.rsi_overbought}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    async def get_market_data(self, symbol: str):
        """Get market data for a symbol with proper error handling"""
        try:
            response = await self.client._request(
                "GET",
                "/v5/market/kline",
                {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": "60",  # 1h
                    "limit": 100,
                },
            )

            klines_data = response.get("result", {}).get("list", [])
            if len(klines_data) < 50:
                logger.warning(
                    f"⚠️ {symbol}: Insufficient data ({len(klines_data)} candles)"
                )
                return None

            # Create DataFrame with proper column handling
            df = pd.DataFrame(
                klines_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                ],
            )

            # Convert data types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.attrs["pair"] = symbol
            return df

        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to get market data - {e}")
            return None

    async def check_single_pair(self, symbol: str):
        """Check a single pair for signals"""
        try:
            # Get market data
            df = await self.get_market_data(symbol)
            if df is None:
                return None

            # Calculate indicators
            indicators = self.strategy.calculate_indicators(df)
            current_rsi = indicators["rsi"]
            current_price = indicators["current_price"]

            # Generate signal
            signal = self.strategy.generate_signal(df)

            # Store current price for tracking
            self.last_prices[symbol] = current_price

            # Log current status
            logger.info(
                f"📊 {symbol}: ${current_price:,.4f} | RSI: {current_rsi:.1f} | Signal: {signal.action}"
            )

            # Check for new signals
            if signal.action != "HOLD":
                signal_info = {
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "action": signal.action,
                    "price": current_price,
                    "confidence": signal.confidence,
                    "rsi": current_rsi,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "indicators": indicators,
                }

                # Check if this is a new signal (not duplicate)
                if not self.is_duplicate_signal(signal_info):
                    self.signals_found.append(signal_info)
                    await self.alert_new_signal(signal_info)

                return signal_info

            return None

        except Exception as e:
            logger.error(f"❌ {symbol}: Signal check failed - {e}")
            return None

    def is_duplicate_signal(self, new_signal):
        """Check if this signal was already found recently"""
        if not self.signals_found:
            return False

        # Check last 10 signals for duplicates within 1 hour
        recent_signals = [
            s
            for s in self.signals_found[-10:]
            if (datetime.now() - s["timestamp"]).total_seconds() < 3600
        ]

        for recent in recent_signals:
            if (
                recent["symbol"] == new_signal["symbol"]
                and recent["action"] == new_signal["action"]
                and abs(recent["price"] - new_signal["price"]) / recent["price"] < 0.02
            ):  # Less than 2% price difference
                return True

        return False

    async def alert_new_signal(self, signal_info):
        """Alert about new signal"""
        symbol = signal_info["symbol"]
        action = signal_info["action"]
        price = signal_info["price"]
        confidence = signal_info["confidence"]
        rsi = signal_info["rsi"]

        logger.info("=" * 60)
        logger.info(f"🎯 NEW SIGNAL DETECTED!")
        logger.info(f"📈 Pair: {symbol}")
        logger.info(f"⚡ Action: {action}")
        logger.info(f"💰 Price: ${price:,.4f}")
        logger.info(f"🎲 Confidence: {confidence:.1%}")
        logger.info(f"📊 RSI: {rsi:.1f}")
        logger.info(f"🛑 Stop Loss: ${signal_info['stop_loss']:,.4f}")
        logger.info(f"🎯 Take Profit: ${signal_info['take_profit']:,.4f}")
        logger.info(
            f"⏰ Time: {signal_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("=" * 60)

        # Try to send Telegram notification if available
        try:
            await self.send_telegram_alert(signal_info)
        except:
            pass  # Don't let Telegram failures stop the monitor

    async def send_telegram_alert(self, signal_info):
        """Send Telegram alert about the signal"""
        try:
            from bot import TradingBot
            from config.config import Config

            bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

            message = f"""📈 **{signal_info['symbol']}** - {signal_info['action']}
💰 Price: ${signal_info['price']:,.4f}
🎲 Confidence: {signal_info['confidence']:.1%}
📊 RSI: {signal_info['rsi']:.1f}

🛑 Stop Loss: ${signal_info['stop_loss']:,.4f}
🎯 Take Profit: ${signal_info['take_profit']:,.4f}

⏰ {signal_info['timestamp'].strftime('%H:%M:%S')}"""

            await bot.send_alert("ENHANCED STRATEGY SIGNAL", message, "money")
            logger.info("📱 Telegram alert sent successfully")

        except Exception as e:
            logger.warning(f"⚠️ Telegram alert failed: {e}")
            # Log more details for debugging
            import traceback

            logger.debug(f"Full error: {traceback.format_exc()}")

    async def run_scan(self):
        """Run a complete scan of all pairs"""
        logger.info(f"🔍 Starting market scan at {datetime.now().strftime('%H:%M:%S')}")

        signals_this_scan = []

        for symbol in self.pairs:
            signal = await self.check_single_pair(symbol)
            if signal:
                signals_this_scan.append(signal)

            # Small delay between pairs to avoid rate limiting
            await asyncio.sleep(0.5)

        if signals_this_scan:
            logger.info(f"✅ Scan complete: {len(signals_this_scan)} signals found")
        else:
            logger.info(f"⚪ Scan complete: No new signals")

        return signals_this_scan

    async def start_monitoring(self, scan_interval=300):  # 5 minutes
        """Start continuous monitoring"""
        if not await self.initialize():
            logger.error("❌ Failed to initialize. Cannot start monitoring.")
            return

        self.running = True
        logger.info(f"🚀 Live monitoring started (scan every {scan_interval}s)")

        try:
            while self.running:
                await self.run_scan()

                if self.running:  # Check if still running after scan
                    logger.info(f"😴 Waiting {scan_interval}s until next scan...")
                    await asyncio.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("⏹️ Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.running = False
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, "client"):
                await self.client.close()
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("🛑 Stop signal sent")


async def main():
    """Main function"""
    monitor = LiveSignalMonitor()

    try:
        # Run one initial scan
        logger.info("🔍 Running initial scan...")
        await monitor.initialize()
        initial_signals = await monitor.run_scan()

        if initial_signals:
            logger.info(f"🎉 Found {len(initial_signals)} signals in initial scan!")

        # Ask user if they want continuous monitoring
        print("\n" + "=" * 60)
        print("🎯 Initial scan complete!")
        print("🔄 Start continuous monitoring? (y/n): ", end="")

        # For automated runs, start monitoring automatically
        # For manual runs, prompt user
        try:
            import sys

            if sys.stdin.isatty():
                response = input().lower().strip()
                if response in ["y", "yes"]:
                    await monitor.start_monitoring(scan_interval=300)  # 5 minutes
            else:
                # Running non-interactively, start monitoring
                await monitor.start_monitoring(scan_interval=300)
        except:
            # Default to monitoring
            await monitor.start_monitoring(scan_interval=300)

    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
