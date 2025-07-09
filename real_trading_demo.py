#!/usr/bin/env python3
"""
Real Trading Demo
Shows current MEXC prices and live trading signal generation
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict
import pandas as pd

from config.config import Config
from mexc.mexc_client import MEXCClient
from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
from strategy.strategies.macd_strategy import MACDStrategy
from strategy.strategies.bollinger_strategy import BollingerStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class RealTradingDemo:
    """Demonstrate real trading signals with current market data"""

    def __init__(self):
        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"]

        # Initialize strategies
        self.strategies = {
            "RSI_EMA": RSIEMAStrategy(),
            "MACD": MACDStrategy(),
            "Bollinger": BollingerStrategy(),
        }

    async def run_demo(self):
        """Run the trading demo"""
        print("🚀 REAL TRADING SYSTEM DEMO")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        await self.show_current_prices()
        await self.demonstrate_live_signals()
        await self.show_account_status()

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE - System is using REAL market data!")
        print("💰 All prices and signals are live from MEXC exchange")

    async def show_current_prices(self):
        """Display current real market prices"""
        print("💰 CURRENT MARKET PRICES (REAL-TIME)")
        print("-" * 40)

        for symbol in self.symbols:
            try:
                ticker = await self.mexc_client.get_ticker(symbol)

                price = float(ticker["lastPrice"])
                volume = float(ticker["volume"])
                change = float(ticker["priceChangePercent"])

                # Price direction indicator
                direction = "📈" if change >= 0 else "📉"

                print(f"{direction} {symbol}:")
                print(f"   Price: ${price:,.4f}")
                print(f"   24h Change: {change:+.2f}%")
                print(f"   Volume: {volume:,.0f}")
                print()

            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")

        print("✅ ALL PRICES ARE LIVE FROM MEXC EXCHANGE")
        print()

    async def demonstrate_live_signals(self):
        """Show live trading signals being generated"""
        print("🎯 LIVE TRADING SIGNALS")
        print("-" * 40)

        # Test with BTC and ETH (most liquid pairs)
        demo_symbols = ["BTCUSDT", "ETHUSDT"]

        for symbol in demo_symbols:
            print(f"📊 Analyzing {symbol} with real market data...")

            try:
                # Get multiple timeframes of real data
                timeframes = [
                    ("1m", "1-minute"),
                    ("5m", "5-minute"),
                    ("15m", "15-minute"),
                ]

                for interval, name in timeframes:
                    # Get real historical data for analysis
                    klines = await self.mexc_client.get_klines(symbol, interval, 100)

                    if len(klines) < 50:
                        print(f"   ⚠️  {name}: Insufficient data")
                        continue

                    print(f"   ⏰ {name} analysis:")
                    print(f"      Data points: {len(klines)} candles")
                    print(f"      Latest price: ${klines['close'].iloc[-1]:,.4f}")

                    # Test each strategy with real data
                    for strategy_name, strategy in self.strategies.items():
                        try:
                            signal = strategy.generate_signal(klines)

                            # Format output based on signal
                            if signal.action == "BUY":
                                action_str = "🟢 BUY"
                            elif signal.action == "SELL":
                                action_str = "🔴 SELL"
                            else:
                                action_str = "⚪ HOLD"

                            print(
                                f"      {action_str} {strategy_name}: "
                                f"Confidence {signal.confidence:.2f}"
                            )

                            # Show trade details for non-HOLD signals
                            if signal.action != "HOLD":
                                # Get current price for comparison
                                current_ticker = await self.mexc_client.get_ticker(
                                    symbol
                                )
                                current_price = float(current_ticker["lastPrice"])

                                print(f"         Signal Price: ${current_price:.4f}")
                                if hasattr(signal, "stop_loss") and signal.stop_loss:
                                    print(
                                        f"         Stop Loss: ${signal.stop_loss:.4f}"
                                    )
                                if (
                                    hasattr(signal, "take_profit")
                                    and signal.take_profit
                                ):
                                    print(
                                        f"         Take Profit: ${signal.take_profit:.4f}"
                                    )

                        except Exception as e:
                            print(f"      ❌ {strategy_name}: {str(e)[:50]}...")

                    print()

            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {e}")

            print()

    async def show_account_status(self):
        """Show account trading status"""
        print("🏦 ACCOUNT STATUS")
        print("-" * 20)

        try:
            # Get account info
            account = await self.mexc_client.get_account()
            print(f"✅ Account connected: {account.get('canTrade', False)}")

            # Get balances
            balances = await self.mexc_client.get_balance()

            # Show significant balances
            print("💼 Balances:")
            has_balance = False
            for asset, balance in balances.items():
                total = balance["free"] + balance["locked"]
                if total > 0.001:
                    print(f"   {asset}: {total:.6f}")
                    has_balance = True

            if not has_balance:
                print("   No significant balances")

            # USDT balance
            usdt_balance = balances.get("USDT", {"free": 0, "locked": 0})
            usdt_total = usdt_balance["free"] + usdt_balance["locked"]

            print(f"\n💵 Trading Balance: ${usdt_total:.2f} USDT")

            if usdt_total > 10:
                print("✅ Ready for live trading")
            else:
                print("⚠️  Low balance - demo mode only")

        except Exception as e:
            print(f"❌ Account error: {e}")


async def main():
    """Run the real trading demo"""
    demo = RealTradingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
