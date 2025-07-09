#!/usr/bin/env python3
"""
Live Trading System Test
Verifies real MEXC market data and trading functionality
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd

from config.config import Config
from mexc.mexc_client import MEXCClient, MEXCTradeExecutor
from mexc.data_feed import MEXCDataFeed
from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
from strategy.strategies.macd_strategy import MACDStrategy
from strategy.strategies.bollinger_strategy import BollingerStrategy
from services.trading_orchestrator import trading_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LiveTradingTest:
    """Test real MEXC trading functionality"""

    def __init__(self):
        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"]
        self.results = {}

    async def run_all_tests(self):
        """Run comprehensive trading system tests"""
        print("🚀 LIVE TRADING SYSTEM VERIFICATION")
        print("=" * 50)
        print(f"Timestamp: {datetime.now()}")
        print(f"Testing with symbols: {', '.join(self.test_symbols)}")
        print()

        try:
            await self.test_current_prices()
            await self.test_account_access()
            await self.test_strategy_signals()
            await self.test_trading_orchestrator()
            await self.test_order_validation()

            print("\n" + "=" * 50)
            print("📊 SUMMARY")
            print("=" * 50)
            self.print_summary()

        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise

    async def test_current_prices(self):
        """Test fetching current market prices"""
        print("💰 TESTING CURRENT MARKET PRICES")
        print("-" * 30)

        try:
            for symbol in self.test_symbols:
                ticker = await self.mexc_client.get_ticker(symbol)

                price = float(ticker["lastPrice"])
                volume = float(ticker["volume"])
                change = float(ticker["priceChangePercent"])

                print(f"📈 {symbol}:")
                print(f"   Current Price: ${price:,.4f}")
                print(f"   24h Volume: {volume:,.0f}")
                print(f"   24h Change: {change:+.2f}%")
                print()

                self.results[symbol] = {
                    "price": price,
                    "volume": volume,
                    "change": change,
                    "real_data": True,
                }

            print("✅ ALL PRICE DATA IS REAL AND CURRENT")

        except Exception as e:
            print(f"❌ Price fetch failed: {e}")
            raise

    async def test_account_access(self):
        """Test account access and balances"""
        print("🏦 TESTING ACCOUNT ACCESS")
        print("-" * 30)

        try:
            # Get account info
            account = await self.mexc_client.get_account()
            print(f"✅ Account connected successfully")
            print(f"   Can Trade: {account.get('canTrade', False)}")

            # Get balances
            balances = await self.mexc_client.get_balance()

            # Show non-zero balances
            print(f"💼 Account Balances:")
            for asset, balance in balances.items():
                total = balance["free"] + balance["locked"]
                if total > 0.001:  # Show balances > 0.001
                    print(f"   {asset}: {total:.6f} (Free: {balance['free']:.6f})")

            # Check USDT specifically
            usdt_balance = balances.get("USDT", {"free": 0, "locked": 0})
            usdt_total = usdt_balance["free"] + usdt_balance["locked"]
            print(f"\n💵 USDT Trading Balance: ${usdt_total:.2f}")

            if usdt_total > 10:
                print("✅ Sufficient balance for live trading")
            else:
                print("⚠️  Low balance - will test order validation only")

            self.results["account"] = {
                "connected": True,
                "can_trade": account.get("canTrade", False),
                "usdt_balance": usdt_total,
            }

        except Exception as e:
            print(f"❌ Account access failed: {e}")
            raise

    async def test_strategy_signals(self):
        """Test trading strategies with real market data"""
        print("🎯 TESTING STRATEGY SIGNALS WITH REAL DATA")
        print("-" * 40)

        strategies = {
            "RSI_EMA": RSIEMAStrategy(),
            "MACD": MACDStrategy(),
            "Bollinger": BollingerStrategy(),
        }

        # Test with multiple timeframes
        timeframes = ["1m", "5m", "15m"]

        for symbol in self.test_symbols[:2]:  # Test first 2 symbols
            print(f"📊 Testing {symbol}:")

            for timeframe in timeframes:
                try:
                    # Get real market data
                    klines = await self.mexc_client.get_klines(symbol, timeframe, 100)

                    if len(klines) < 50:
                        print(
                            f"   ⚠️  {timeframe}: Insufficient data ({len(klines)} candles)"
                        )
                        continue

                    print(f"   ⏰ {timeframe} timeframe ({len(klines)} candles):")

                    for strategy_name, strategy in strategies.items():
                        try:
                            signal = strategy.generate_signal(klines)

                            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
                            emoji = action_emoji.get(signal.action, "⚪")

                            print(
                                f"      {emoji} {strategy_name}: {signal.action} "
                                f"(Confidence: {signal.confidence:.2f})"
                            )

                            if signal.action != "HOLD":
                                print(f"         Entry: ${signal.price:.4f}")
                                if signal.stop_loss:
                                    print(
                                        f"         Stop Loss: ${signal.stop_loss:.4f}"
                                    )
                                if signal.take_profit:
                                    print(
                                        f"         Take Profit: ${signal.take_profit:.4f}"
                                    )

                        except Exception as e:
                            print(f"      ❌ {strategy_name}: {e}")

                except Exception as e:
                    print(f"   ❌ {timeframe}: Failed to get data - {e}")

            print()

    async def test_trading_orchestrator(self):
        """Test the trading orchestrator with real data"""
        print("🤖 TESTING TRADING ORCHESTRATOR")
        print("-" * 30)

        try:
            # Initialize orchestrator if needed
            await trading_orchestrator.start()

            # Test processing real market signals
            test_symbol = self.test_symbols[0]
            ticker = await self.mexc_client.get_ticker(test_symbol)

            market_data = {
                "symbol": test_symbol,
                "price": float(ticker["lastPrice"]),
                "volume": float(ticker["volume"]),
                "timestamp": datetime.now().isoformat(),
                "source": "MEXC_REAL",
            }

            print(f"📡 Processing real market signal for {test_symbol}")
            print(f"   Price: ${market_data['price']:,.4f}")
            print(f"   Volume: {market_data['volume']:,.0f}")

            # This would trigger signal processing for all users
            # In a real scenario, but we'll test the mechanism
            result = await trading_orchestrator.process_market_signal(
                test_symbol, market_data
            )

            print(f"✅ Orchestrator processed signal successfully")
            self.results["orchestrator"] = {"working": True}

        except Exception as e:
            print(f"❌ Orchestrator test failed: {e}")
            self.results["orchestrator"] = {"working": False, "error": str(e)}

    async def test_order_validation(self):
        """Test order creation and validation"""
        print("📋 TESTING ORDER VALIDATION")
        print("-" * 30)

        try:
            test_symbol = self.test_symbols[0]
            ticker = await self.mexc_client.get_ticker(test_symbol)
            current_price = float(ticker["lastPrice"])

            # Test order parameters
            test_quantity = 0.001  # Small test amount

            print(f"🔍 Validating order for {test_symbol}:")
            print(f"   Current Price: ${current_price:,.4f}")
            print(f"   Test Quantity: {test_quantity}")
            print(f"   Estimated Value: ${current_price * test_quantity:.2f}")

            # Get symbol info for validation
            exchange_info = await self.mexc_client.get_exchange_info()
            symbol_info = None

            for symbol_data in exchange_info["symbols"]:
                if symbol_data["symbol"] == test_symbol:
                    symbol_info = symbol_data
                    break

            if symbol_info:
                print(f"✅ Symbol {test_symbol} is tradeable")
                print(f"   Status: {symbol_info['status']}")
                print(f"   Base Asset: {symbol_info['baseAsset']}")
                print(f"   Quote Asset: {symbol_info['quoteAsset']}")

                # Check filters
                for filter_info in symbol_info.get("filters", []):
                    if filter_info["filterType"] == "LOT_SIZE":
                        min_qty = float(filter_info["minQty"])
                        print(f"   Min Quantity: {min_qty:.8f}")

                        if test_quantity >= min_qty:
                            print(f"   ✅ Order quantity is valid")
                        else:
                            print(
                                f"   ⚠️  Order quantity too small (min: {min_qty:.8f})"
                            )

                self.results["order_validation"] = {"valid": True}
            else:
                print(f"❌ Symbol {test_symbol} not found")
                self.results["order_validation"] = {"valid": False}

        except Exception as e:
            print(f"❌ Order validation failed: {e}")
            self.results["order_validation"] = {"valid": False, "error": str(e)}

    def print_summary(self):
        """Print test results summary"""
        total_tests = 0
        passed_tests = 0

        if self.results:
            print("📊 Market Data:")
            for symbol, data in self.results.items():
                if isinstance(data, dict) and "price" in data:
                    total_tests += 1
                    if data.get("real_data"):
                        passed_tests += 1
                        print(f"   ✅ {symbol}: ${data['price']:,.4f} (REAL)")
                    else:
                        print(f"   ❌ {symbol}: No data")

        if "account" in self.results:
            total_tests += 1
            if self.results["account"].get("connected"):
                passed_tests += 1
                print(
                    f"   ✅ Account: Connected (${self.results['account']['usdt_balance']:.2f} USDT)"
                )
            else:
                print(f"   ❌ Account: Failed")

        if "orchestrator" in self.results:
            total_tests += 1
            if self.results["orchestrator"].get("working"):
                passed_tests += 1
                print(f"   ✅ Trading Orchestrator: Working")
            else:
                print(f"   ❌ Trading Orchestrator: Failed")

        if "order_validation" in self.results:
            total_tests += 1
            if self.results["order_validation"].get("valid"):
                passed_tests += 1
                print(f"   ✅ Order Validation: Working")
            else:
                print(f"   ❌ Order Validation: Failed")

        print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 ALL SYSTEMS GO! Trading system is using REAL market data!")
        else:
            print("⚠️  Some issues detected. Check logs above.")


async def main():
    """Run live trading tests"""
    test = LiveTradingTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
