#!/usr/bin/env python3
"""
Strategy Signal Debugging Script
Analyzes why current strategy isn't generating signals and tests the new enhanced strategy
Uses Bybit API for testing
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

from config.config import Config
from bybit.bybit_client import BybitClient
from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
from indicators import calculate_rsi, calculate_ema, calculate_atr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class StrategyDebugger:
    """Debug strategy signal generation issues"""

    def __init__(self):
        self.bybit_client = BybitClient(
            Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET, Config.BYBIT_TESTNET
        )
        self.original_strategy = RSIEMAStrategy()
        self.enhanced_strategy = EnhancedRSIEMAStrategy()
        self.test_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

    async def run_debug_analysis(self):
        """Run comprehensive strategy debugging"""
        print("🔍 TRADING STRATEGY DEBUGGING ANALYSIS")
        print("=" * 60)
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        await self.analyze_market_conditions()
        await self.debug_original_strategy()
        await self.test_enhanced_strategy()
        await self.provide_recommendations()

    async def analyze_market_conditions(self):
        """Analyze current market conditions"""
        print("📊 CURRENT MARKET CONDITIONS")
        print("-" * 40)

        for pair in self.test_pairs:
            try:
                # Get recent kline data
                klines = await self.bybit_client.get_klines(pair, "1h", 100)

                if len(klines) < 50:
                    print(f"❌ {pair}: Insufficient data ({len(klines)} candles)")
                    continue

                # Calculate current indicators
                close_prices = klines["close"]
                rsi = calculate_rsi(close_prices, 14)
                ema_fast = calculate_ema(close_prices, 9)
                ema_slow = calculate_ema(close_prices, 21)
                volume_avg = klines["volume"].rolling(window=20).mean()

                current_rsi = rsi.iloc[-1]
                current_ema_fast = ema_fast.iloc[-1]
                current_ema_slow = ema_slow.iloc[-1]
                current_price = close_prices.iloc[-1]
                current_volume = klines["volume"].iloc[-1]
                avg_volume = volume_avg.iloc[-1]

                # Get 24h volume in USDT
                ticker = await self.bybit_client.get_ticker(pair)
                volume_usdt = float(ticker["turnover"])

                # Analyze conditions
                ema_trend = (
                    "Bullish" if current_ema_fast > current_ema_slow else "Bearish"
                )
                price_vs_ema = "Above" if current_price > current_ema_fast else "Below"
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                print(f"📈 {pair}:")
                print(f"   Price: ${current_price:.4f}")
                print(
                    f"   RSI: {current_rsi:.1f} ({'Oversold' if current_rsi < 35 else 'Overbought' if current_rsi > 65 else 'Neutral'})"
                )
                print(f"   EMA Trend: {ema_trend}")
                print(f"   Price vs EMA: {price_vs_ema} fast EMA")
                print(f"   Volume Ratio: {volume_ratio:.2f}x average")
                print(f"   24h Volume: ${volume_usdt:,.0f} USDT")

                # Check original strategy conditions
                self.check_original_strategy_conditions(
                    pair,
                    current_rsi,
                    current_ema_fast,
                    current_ema_slow,
                    current_price,
                    volume_ratio,
                    volume_usdt,
                )
                print()

            except Exception as e:
                print(f"❌ {pair}: Error - {e}")
                print()

    def check_original_strategy_conditions(
        self,
        pair: str,
        rsi: float,
        ema_fast: float,
        ema_slow: float,
        price: float,
        volume_ratio: float,
        volume_usdt: float,
    ):
        """Check why original strategy conditions aren't met"""
        print("   🔍 Original Strategy Analysis:")

        # RSI conditions
        rsi_buy_ok = rsi < 35
        rsi_sell_ok = rsi > 65
        print(f"      RSI Buy (< 35): {'✅' if rsi_buy_ok else '❌'}")
        print(f"      RSI Sell (> 65): {'✅' if rsi_sell_ok else '❌'}")

        # EMA conditions
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow
        price_above_ema = price > ema_fast
        price_below_ema = price < ema_fast

        ema_buy_ok = ema_bullish and price_above_ema
        ema_sell_ok = ema_bearish and price_below_ema

        print(f"      EMA Buy (bullish + price above): {'✅' if ema_buy_ok else '❌'}")
        print(
            f"      EMA Sell (bearish + price below): {'✅' if ema_sell_ok else '❌'}"
        )

        # Volume conditions
        volume_ok = volume_ratio > 0.8 and volume_usdt > 50000
        print(f"      Volume (>0.8x avg + >50k USDT): {'✅' if volume_ok else '❌'}")

        # Combined signal potential
        buy_possible = rsi_buy_ok and ema_buy_ok and volume_ok
        sell_possible = rsi_sell_ok and ema_sell_ok and volume_ok

        if buy_possible:
            print("      🟢 BUY signal conditions MET!")
        elif sell_possible:
            print("      🔴 SELL signal conditions MET!")
        else:
            print("      ⚪ No signal - conditions not aligned")

    async def debug_original_strategy(self):
        """Debug the original strategy with detailed analysis"""
        print("🔧 ORIGINAL STRATEGY DEBUGGING")
        print("-" * 40)

        total_signals = 0
        for pair in self.test_pairs:
            try:
                # Get recent data
                klines = await self.bybit_client.get_klines(pair, "1h", 100)

                if len(klines) < 50:
                    continue

                # Set pair attribute for strategy
                klines.attrs = {"pair": pair}

                # Generate signal
                signal = self.original_strategy.generate_signal(klines)

                print(f"📊 {pair} - Original Strategy:")
                print(f"   Signal: {signal.action}")
                print(f"   Confidence: {signal.confidence:.3f}")

                if signal.action != "HOLD":
                    total_signals += 1
                    print(f"   🎯 Price: ${signal.current_price:.4f}")
                    print(
                        f"   📈 Indicators: RSI={signal.indicators['rsi']:.1f}, EMA={signal.indicators['ema_trend']}"
                    )
                    print(
                        f"   💼 Volume OK: {signal.indicators['volume_confirmation']}"
                    )
                else:
                    print(f"   📊 Current RSI: {signal.indicators['rsi']:.1f}")
                    print(f"   📈 EMA Trend: {signal.indicators['ema_trend']}")
                    print(
                        f"   💼 Volume OK: {signal.indicators['volume_confirmation']}"
                    )

                print()

            except Exception as e:
                print(f"❌ {pair}: Error - {e}")
                print()

        print(
            f"📊 Original Strategy Summary: {total_signals} signals from {len(self.test_pairs)} pairs"
        )
        print()

    async def test_enhanced_strategy(self):
        """Test the new enhanced strategy"""
        print("🚀 ENHANCED STRATEGY TESTING")
        print("-" * 40)

        total_signals = 0
        for pair in self.test_pairs:
            try:
                # Get recent data
                klines = await self.bybit_client.get_klines(pair, "1h", 100)

                if len(klines) < 50:
                    continue

                # Set pair attribute for strategy
                klines.attrs = {"pair": pair}

                # Generate signal
                signal = self.enhanced_strategy.generate_signal(klines)

                print(f"📊 {pair} - Enhanced Strategy:")
                print(f"   Signal: {signal.action}")
                print(f"   Confidence: {signal.confidence:.3f}")

                if signal.action != "HOLD":
                    total_signals += 1
                    print(f"   🎯 Price: ${signal.current_price:.4f}")
                    print(
                        f"   📋 Reasons: {', '.join(signal.indicators['signal_reasons'])}"
                    )
                    print(
                        f"   💪 Buy Strength: {signal.indicators['buy_strength']:.3f}"
                    )
                    print(
                        f"   💪 Sell Strength: {signal.indicators['sell_strength']:.3f}"
                    )

                    if signal.stop_loss:
                        print(f"   🛡️ Stop Loss: ${signal.stop_loss:.4f}")
                    if signal.take_profit:
                        print(f"   🎯 Take Profit: ${signal.take_profit:.4f}")
                else:
                    print(f"   📊 Current RSI: {signal.indicators['rsi']:.1f}")
                    print(f"   📈 EMA Trend: {signal.indicators['ema_trend']}")
                    print(
                        f"   💪 Buy Strength: {signal.indicators['buy_strength']:.3f}"
                    )
                    print(
                        f"   💪 Sell Strength: {signal.indicators['sell_strength']:.3f}"
                    )

                print()

            except Exception as e:
                print(f"❌ {pair}: Error - {e}")
                print()

        print(
            f"📊 Enhanced Strategy Summary: {total_signals} signals from {len(self.test_pairs)} pairs"
        )
        print()

    async def provide_recommendations(self):
        """Provide recommendations for improving signal generation"""
        print("💡 RECOMMENDATIONS")
        print("-" * 40)

        print("Based on the analysis, here are the issues with your current strategy:")
        print()
        print("🔴 PROBLEMS WITH ORIGINAL STRATEGY:")
        print("   1. RSI thresholds (35/65) are too extreme - rarely reached")
        print("   2. Requires ALL conditions to align simultaneously")
        print("   3. High confidence threshold (0.7) filters out most signals")
        print("   4. Strict volume requirement (80% of average)")
        print("   5. Price must be above/below EMA for signal activation")
        print()

        print("✅ ENHANCED STRATEGY IMPROVEMENTS:")
        print("   1. More practical RSI levels (40/60 with 30/70 for strong signals)")
        print("   2. Weighted scoring system instead of ALL-or-NOTHING")
        print("   3. Lower confidence threshold (0.4) for more signals")
        print("   4. Less strict volume requirement (60% of average)")
        print("   5. Multiple signal types and crossover detection")
        print("   6. RSI momentum and trend continuation signals")
        print()

        print("🛠️ IMMEDIATE ACTIONS:")
        print("   1. Switch to the Enhanced RSI EMA Strategy")
        print("   2. Monitor signals for a few days")
        print("   3. Adjust confidence threshold if needed")
        print("   4. Consider paper trading first")
        print()

    async def close(self):
        """Clean up resources"""
        if hasattr(self.bybit_client, "session") and self.bybit_client.session:
            await self.bybit_client.close()


async def main():
    """Main execution"""
    debugger = StrategyDebugger()

    try:
        await debugger.run_debug_analysis()
    except Exception as e:
        logger.error(f"Debug analysis failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await debugger.close()


if __name__ == "__main__":
    asyncio.run(main())
