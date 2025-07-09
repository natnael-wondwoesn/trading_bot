#!/usr/bin/env python3
"""
Bybit Integration Test
Tests the Bybit integration and multi-exchange functionality
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from config.config import Config
from exchange_factory import ExchangeFactory
from exchange_interface import BaseExchangeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BybitIntegrationTest:
    """Test Bybit integration and multi-exchange functionality"""

    def __init__(self):
        self.test_symbols = ["BTCUSDT", "ETHUSDT"]

    async def run_comprehensive_test(self):
        """Run comprehensive test of Bybit integration"""
        logger.info("🚀 Starting Bybit Integration Test")
        print("=" * 60)

        # Test 1: Check available exchanges
        await self._test_available_exchanges()

        # Test 2: Test Bybit configuration
        await self._test_bybit_configuration()

        # Test 3: Test Bybit client creation
        await self._test_bybit_client_creation()

        # Test 4: Test Bybit market data
        await self._test_bybit_market_data()

        # Test 5: Test exchange factory
        await self._test_exchange_factory()

        # Test 6: Test multi-exchange support
        await self._test_multi_exchange_support()

        print("=" * 60)
        logger.info("✅ Bybit Integration Test Complete!")

    async def _test_available_exchanges(self):
        """Test available exchanges detection"""
        logger.info("📊 Testing available exchanges...")

        try:
            exchanges = ExchangeFactory.get_available_exchanges()

            print("\n🏦 Available Exchanges:")
            for name, capabilities in exchanges.items():
                status = (
                    "✅ Available" if capabilities.available else "❌ Not Available"
                )
                print(f"  • {capabilities.name}: {status}")
                if not capabilities.available:
                    print(f"    Error: {capabilities.error_message}")

            logger.info(f"✅ Found {len(exchanges)} exchanges")

        except Exception as e:
            logger.error(f"❌ Failed to get available exchanges: {e}")

    async def _test_bybit_configuration(self):
        """Test Bybit configuration validation"""
        logger.info("⚙️  Testing Bybit configuration...")

        try:
            # Test MEXC configuration
            mexc_valid, mexc_msg = ExchangeFactory.validate_exchange_config("MEXC")
            print(f"\n📋 MEXC Config: {'✅' if mexc_valid else '❌'} {mexc_msg}")

            # Test Bybit configuration
            bybit_valid, bybit_msg = ExchangeFactory.validate_exchange_config("BYBIT")
            print(f"📋 Bybit Config: {'✅' if bybit_valid else '❌'} {bybit_msg}")

            if bybit_valid:
                logger.info("✅ Bybit configuration is valid")
            else:
                logger.warning(f"⚠️  Bybit configuration issue: {bybit_msg}")

        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")

    async def _test_bybit_client_creation(self):
        """Test Bybit client creation"""
        logger.info("🔧 Testing Bybit client creation...")

        try:
            # Test default exchange
            default_exchange = ExchangeFactory.get_default_exchange()
            print(f"\n🎯 Default Exchange: {default_exchange}")

            # Try to create Bybit client
            available_exchanges = ExchangeFactory.get_available_exchanges()
            if (
                "BYBIT" in available_exchanges
                and available_exchanges["BYBIT"].available
            ):
                client = ExchangeFactory.create_client("BYBIT", testnet=True)
                print("✅ Bybit client created successfully")

                # Test basic connectivity
                server_time = await client.get_server_time()
                print(f"✅ Bybit server time: {server_time}")

                await client.close()
                logger.info("✅ Bybit client test passed")
            else:
                logger.warning("⚠️  Bybit not available, skipping client test")

        except Exception as e:
            logger.error(f"❌ Bybit client creation failed: {e}")

    async def _test_bybit_market_data(self):
        """Test Bybit market data functionality"""
        logger.info("📈 Testing Bybit market data...")

        try:
            available_exchanges = ExchangeFactory.get_available_exchanges()
            if (
                "BYBIT" not in available_exchanges
                or not available_exchanges["BYBIT"].available
            ):
                logger.warning("⚠️  Bybit not available, skipping market data test")
                return

            client = ExchangeFactory.create_client("BYBIT", testnet=True)

            print(f"\n💰 Testing market data for {self.test_symbols}:")

            for symbol in self.test_symbols:
                try:
                    # Get ticker data
                    ticker = await client.get_ticker(symbol)
                    price = float(ticker.get("lastPrice", 0))
                    print(f"  • {symbol}: ${price:,.2f}")

                    # Get klines data
                    klines = await client.get_klines(symbol, "5m", 10)
                    print(f"    📊 Klines: {len(klines)} candles")

                except Exception as e:
                    print(f"  ❌ {symbol}: Error - {str(e)[:50]}...")

            await client.close()
            logger.info("✅ Market data test completed")

        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")

    async def _test_exchange_factory(self):
        """Test exchange factory functionality"""
        logger.info("🏭 Testing exchange factory...")

        try:
            print(f"\n🔧 Testing Exchange Factory:")

            # Test supported pairs
            mexc_pairs = ExchangeFactory.get_supported_pairs("MEXC")
            bybit_pairs = ExchangeFactory.get_supported_pairs("BYBIT")

            print(f"  • MEXC pairs: {len(mexc_pairs)} symbols")
            print(f"  • Bybit pairs: {len(bybit_pairs)} symbols")

            # Test symbol normalization
            test_symbol = "btcusdt"
            mexc_normalized = ExchangeFactory.normalize_symbol_for_exchange(
                test_symbol, "MEXC"
            )
            bybit_normalized = ExchangeFactory.normalize_symbol_for_exchange(
                test_symbol, "BYBIT"
            )

            print(f"  • Symbol normalization:")
            print(f"    MEXC: {test_symbol} → {mexc_normalized}")
            print(f"    Bybit: {test_symbol} → {bybit_normalized}")

            # Test display names
            mexc_display = ExchangeFactory.get_exchange_display_name("MEXC")
            bybit_display = ExchangeFactory.get_exchange_display_name("BYBIT")

            print(f"  • Display names:")
            print(f"    MEXC: {mexc_display}")
            print(f"    Bybit: {bybit_display}")

            logger.info("✅ Exchange factory test passed")

        except Exception as e:
            logger.error(f"❌ Exchange factory test failed: {e}")

    async def _test_multi_exchange_support(self):
        """Test multi-exchange support"""
        logger.info("🔄 Testing multi-exchange support...")

        try:
            available_exchanges = ExchangeFactory.get_available_exchanges()
            available_names = [
                name for name, cap in available_exchanges.items() if cap.available
            ]

            print(f"\n🔀 Multi-Exchange Test:")
            print(f"Available exchanges: {', '.join(available_names)}")

            # Test creating clients for all available exchanges
            for exchange_name in available_names:
                try:
                    client, data_feed, executor = ExchangeFactory.create_exchange_suite(
                        exchange_name
                    )

                    # Test basic functionality
                    server_time = await client.get_server_time()

                    print(f"  ✅ {exchange_name}: Suite created successfully")

                    await client.close()

                except Exception as e:
                    print(f"  ❌ {exchange_name}: {str(e)[:50]}...")

            logger.info("✅ Multi-exchange test completed")

        except Exception as e:
            logger.error(f"❌ Multi-exchange test failed: {e}")

    async def test_user_simulation(self):
        """Simulate user switching between exchanges"""
        logger.info("👤 Testing user exchange switching simulation...")

        try:
            # Simulate user settings
            user_settings_mexc = {"exchange": "MEXC", "trading_strategy": "RSI_EMA"}
            user_settings_bybit = {"exchange": "BYBIT", "trading_strategy": "MACD"}

            print(f"\n👤 User Exchange Switch Simulation:")

            # Test MEXC setup
            if ExchangeFactory.get_available_exchanges().get("MEXC", {}).available:
                mexc_client, mexc_feed, mexc_executor = (
                    ExchangeFactory.create_exchange_suite("MEXC")
                )
                print("  ✅ User configured for MEXC")
                await mexc_client.close()

            # Test Bybit setup
            if ExchangeFactory.get_available_exchanges().get("BYBIT", {}).available:
                bybit_client, bybit_feed, bybit_executor = (
                    ExchangeFactory.create_exchange_suite("BYBIT")
                )
                print("  ✅ User switched to Bybit")
                await bybit_client.close()

            logger.info("✅ User simulation test passed")

        except Exception as e:
            logger.error(f"❌ User simulation test failed: {e}")


async def main():
    """Run the Bybit integration test"""
    test = BybitIntegrationTest()

    await test.run_comprehensive_test()
    await test.test_user_simulation()


if __name__ == "__main__":
    asyncio.run(main())
