#!/usr/bin/env python3
"""
Trading Orchestrator Integration Test
Tests the integration between MEXC data feed and the trading orchestrator
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from config.config import Config
from mexc.mexc_client import MEXCClient
from mexc.data_feed import MEXCDataFeed
from services.trading_orchestrator import TradingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class OrchestratorIntegrationTest:
    """Tests the full integration between market data and trading orchestrator"""

    def __init__(self):
        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
        self.data_feed = MEXCDataFeed(self.mexc_client)
        self.orchestrator = TradingOrchestrator()

        self.test_symbols = ["BTCUSDT", "ETHUSDT"]
        self.signals_received = 0

    async def run_integration_test(self):
        """Run full integration test"""
        print("=" * 80)
        print("TRADING ORCHESTRATOR INTEGRATION TEST")
        print("=" * 80)

        try:
            # Initialize orchestrator
            print("1. Initializing Trading Orchestrator...")
            await self.orchestrator.initialize()
            print("✓ Orchestrator initialized")

            # Test user session creation
            print("\n2. Creating test user session...")
            session = await self.orchestrator.get_or_create_session(
                user_id=1, telegram_id=123456789
            )
            print(f"✓ User session created: {session.user_id}")
            print(f"  Strategy: {type(session.strategy).__name__}")
            print(f"  Active: {session.is_active}")

            # Start data feed
            print("\n3. Starting MEXC data feed...")
            await self.data_feed.start(self.test_symbols, "1m")

            # Set up data feed callback to orchestrator
            for symbol in self.test_symbols:
                self.data_feed.subscribe(symbol, "kline_closed", self.on_market_data)
                self.data_feed.subscribe(symbol, "ticker_update", self.on_ticker_update)

            print("✓ Data feed started and connected to orchestrator")

            # Simulate some market data processing
            print("\n4. Testing market data processing...")
            await self.test_market_data_processing()

            # Monitor for signals
            print("\n5. Monitoring for trading signals...")
            await self.monitor_signals(duration=30)

            # Test orchestrator statistics
            print("\n6. Checking orchestrator statistics...")
            stats = self.orchestrator.get_system_stats()
            print(f"✓ System Stats:")
            print(f"  Active Users: {stats['total_users_active']}")
            print(f"  Signals Processed: {stats['total_signals_processed']}")
            print(f"  Trades Executed: {stats['total_trades_executed']}")
            print(f"  Error Count: {stats['error_count']}")
            print(f"  Avg Processing Time: {stats['avg_processing_time']:.3f}s")

            print("\n✓ Integration test completed successfully!")

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise
        finally:
            await self.cleanup()

    async def on_market_data(self, symbol: str, kline_data):
        """Handle kline updates from data feed"""
        try:
            # Convert kline data to market data format expected by orchestrator
            if hasattr(kline_data, "iloc"):
                latest_candle = kline_data.iloc[-1]
                market_data = {
                    "symbol": symbol,
                    "price": float(latest_candle["close"]),
                    "volume": float(latest_candle["volume"]),
                    "high": float(latest_candle["high"]),
                    "low": float(latest_candle["low"]),
                    "open": float(latest_candle["open"]),
                    "timestamp": datetime.now().isoformat(),
                    "kline_data": kline_data,
                }

                # Send to orchestrator for processing
                signals = await self.orchestrator.process_market_signal(
                    symbol, market_data
                )

                if signals:
                    self.signals_received += len(signals)
                    print(f"📊 Generated {len(signals)} signals for {symbol}")
                    for signal in signals:
                        print(
                            f"   {signal.action} @ ${signal.price:.4f} (confidence: {signal.confidence:.2f})"
                        )

        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")

    async def on_ticker_update(self, symbol: str, ticker_data):
        """Handle ticker updates"""
        try:
            if isinstance(ticker_data, dict) and "c" in ticker_data:
                price = float(ticker_data["c"])
                print(f"💰 {symbol}: ${price:,.4f}")
        except Exception as e:
            logger.error(f"Error processing ticker for {symbol}: {e}")

    async def test_market_data_processing(self):
        """Test direct market data processing"""
        try:
            # Get some historical data
            klines = await self.mexc_client.get_klines("BTCUSDT", "1h", 50)

            # Create market data
            latest_candle = klines.iloc[-1]
            market_data = {
                "symbol": "BTCUSDT",
                "price": float(latest_candle["close"]),
                "volume": float(latest_candle["volume"]),
                "high": float(latest_candle["high"]),
                "low": float(latest_candle["low"]),
                "open": float(latest_candle["open"]),
                "timestamp": datetime.now().isoformat(),
                "kline_data": klines,
            }

            # Process through orchestrator
            signals = await self.orchestrator.process_market_signal(
                "BTCUSDT", market_data
            )

            print(f"✓ Processed market data for BTCUSDT")
            print(f"  Generated {len(signals)} signals")

            if signals:
                for signal in signals:
                    print(f"  Signal: {signal.action} @ ${signal.price:.4f}")

        except Exception as e:
            logger.error(f"Market data processing test failed: {e}")

    async def monitor_signals(self, duration: int = 30):
        """Monitor for trading signals for specified duration"""
        print(f"Monitoring for {duration} seconds...")

        start_time = datetime.now()
        initial_signals = self.signals_received

        while (datetime.now() - start_time).seconds < duration:
            await asyncio.sleep(1)

            # Show progress every 5 seconds
            elapsed = (datetime.now() - start_time).seconds
            if elapsed % 5 == 0 and elapsed > 0:
                new_signals = self.signals_received - initial_signals
                print(f"  {elapsed}/{duration}s - Signals received: {new_signals}")

        total_new_signals = self.signals_received - initial_signals
        print(f"✓ Monitoring complete: {total_new_signals} new signals in {duration}s")

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.orchestrator:
                await self.orchestrator.shutdown()
            if self.data_feed:
                await self.data_feed.stop()
            if self.mexc_client and self.mexc_client.session:
                await self.mexc_client.session.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main test function"""
    test = OrchestratorIntegrationTest()

    try:
        await test.run_integration_test()
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            print("\n🎉 Integration test completed successfully!")
            print(
                "The trading system is properly connected and processing market data."
            )
        else:
            print("\n❌ Integration test failed!")
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
