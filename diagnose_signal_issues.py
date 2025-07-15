#!/usr/bin/env python3
"""
Signal Generation Diagnostic Tool
Identifies why the enhanced strategy isn't generating signals
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_enhanced_strategy_signals():
    """Test enhanced strategy signal generation with current market data"""
    print("🔍 ENHANCED STRATEGY SIGNAL DIAGNOSTIC")
    print("=" * 60)

    try:
        from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
        from config.config import Config

        print("✅ Enhanced strategy imported successfully")

        # Initialize strategy
        strategy = EnhancedRSIEMAStrategy()
        print(f"✅ Strategy initialized: {strategy.name}")
        print(
            f"   RSI Thresholds: Buy<{strategy.rsi_oversold}, Sell>{strategy.rsi_overbought}"
        )
        print(
            f"   Strong Thresholds: Buy<{strategy.rsi_strong_oversold}, Sell>{strategy.rsi_strong_overbought}"
        )

    except Exception as e:
        print(f"❌ Failed to initialize enhanced strategy: {e}")
        return False

    # Test each trading pair
    print(f"\n📊 Testing signal generation on {len(Config.TRADING_PAIRS)} pairs...")

    try:
        # Try to get real market data
        if hasattr(Config, "BYBIT_API_KEY") and Config.BYBIT_API_KEY:
            await test_with_bybit_data(strategy)
        else:
            print("⚠️ No Bybit API keys found, testing with MEXC...")
            await test_with_mexc_data(strategy)

    except Exception as e:
        print(f"❌ Error testing with live data: {e}")
        print("🔄 Falling back to synthetic data test...")
        test_with_synthetic_data(strategy)


async def test_with_bybit_data(strategy):
    """Test with real Bybit market data"""
    print("\n🏦 Testing with real Bybit market data...")

    try:
        from bybit.bybit_client import BybitClient
        from config.config import Config

        client = BybitClient(
            Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET, testnet=Config.BYBIT_TESTNET
        )

        for pair in Config.TRADING_PAIRS:
            try:
                print(f"\n📈 Testing {pair}...")

                # Get historical klines
                klines = await client.get_klines(symbol=pair, interval="1h", limit=100)

                if not klines or len(klines) < 50:
                    print(
                        f"❌ {pair}: Insufficient data ({len(klines) if klines else 0} candles)"
                    )
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(
                    klines,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col])

                df.attrs["pair"] = pair

                # Test signal generation
                signal = strategy.generate_signal(df)

                # Get current price and RSI for analysis
                indicators = strategy.calculate_indicators(df)
                current_rsi = indicators["rsi"]
                current_price = indicators["current_price"]

                print(f"   Current Price: ${current_price:,.4f}")
                print(f"   Current RSI: {current_rsi:.2f}")
                print(
                    f"   Signal: {signal.action} (Confidence: {signal.confidence:.3f})"
                )

                if signal.action != "HOLD":
                    print(
                        f"🎯 {pair}: SIGNAL GENERATED! {signal.action} at ${current_price:,.4f}"
                    )
                    print(f"   Stop Loss: ${signal.stop_loss:,.4f}")
                    print(f"   Take Profit: ${signal.take_profit:,.4f}")
                else:
                    # Analyze why no signal
                    analyze_no_signal(current_rsi, indicators, strategy, pair)

            except Exception as e:
                print(f"❌ {pair}: Error - {e}")

    except ImportError:
        print("❌ Bybit client not available, falling back to MEXC...")
        await test_with_mexc_data(strategy)
    except Exception as e:
        print(f"❌ Bybit test failed: {e}")


async def test_with_mexc_data(strategy):
    """Test with real MEXC market data"""
    print("\n🏪 Testing with real MEXC market data...")

    try:
        from mexc.mexc_client import MEXCClient
        from config.config import Config

        client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)

        for pair in Config.TRADING_PAIRS:
            try:
                print(f"\n📈 Testing {pair}...")

                # Get historical klines
                klines = await client.get_klines(pair, "1h", 100)

                if not klines or len(klines) < 50:
                    print(
                        f"❌ {pair}: Insufficient data ({len(klines) if klines else 0} candles)"
                    )
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(klines)
                df.attrs["pair"] = pair

                # Test signal generation
                signal = strategy.generate_signal(df)

                # Get current price and RSI for analysis
                indicators = strategy.calculate_indicators(df)
                current_rsi = indicators["rsi"]
                current_price = indicators["current_price"]

                print(f"   Current Price: ${current_price:,.4f}")
                print(f"   Current RSI: {current_rsi:.2f}")
                print(
                    f"   Signal: {signal.action} (Confidence: {signal.confidence:.3f})"
                )

                if signal.action != "HOLD":
                    print(
                        f"🎯 {pair}: SIGNAL GENERATED! {signal.action} at ${current_price:,.4f}"
                    )
                else:
                    analyze_no_signal(current_rsi, indicators, strategy, pair)

            except Exception as e:
                print(f"❌ {pair}: Error - {e}")

    except Exception as e:
        print(f"❌ MEXC test failed: {e}")


def analyze_no_signal(current_rsi, indicators, strategy, pair):
    """Analyze why no signal was generated"""
    print(f"   📊 Analysis for {pair}:")

    # RSI analysis
    if current_rsi > strategy.rsi_overbought:
        print(
            f"   🔴 RSI ({current_rsi:.1f}) > Overbought ({strategy.rsi_overbought}) - Potential SELL zone"
        )
    elif current_rsi < strategy.rsi_oversold:
        print(
            f"   🟢 RSI ({current_rsi:.1f}) < Oversold ({strategy.rsi_oversold}) - Potential BUY zone"
        )
    else:
        print(
            f"   ⚪ RSI ({current_rsi:.1f}) in neutral zone ({strategy.rsi_oversold}-{strategy.rsi_overbought})"
        )

    # EMA analysis
    ema_trend = (
        "Bullish" if indicators["ema_fast"] > indicators["ema_slow"] else "Bearish"
    )
    print(f"   📈 EMA Trend: {ema_trend}")

    # Volume analysis
    if "volume" in indicators and "avg_volume" in indicators:
        volume_ratio = indicators["volume"] / indicators["avg_volume"]
        print(f"   📊 Volume: {volume_ratio:.2f}x average")
        if volume_ratio < 0.8:
            print(f"   ⚠️ Low volume may be preventing signals")


def test_with_synthetic_data(strategy):
    """Test with synthetic data to verify strategy logic"""
    print("\n🧪 Testing with synthetic data...")

    # Create synthetic oversold condition
    print("\n🟢 Testing BUY signal (Oversold + Bullish EMA)...")
    synthetic_data = create_synthetic_data(
        rsi_value=35,  # Oversold
        ema_fast=100,
        ema_slow=98,  # Bullish trend
        price=100,
        volume_ratio=1.2,
    )

    signal = strategy.generate_signal(synthetic_data)
    print(f"   Signal: {signal.action} (Confidence: {signal.confidence:.3f})")

    # Create synthetic overbought condition
    print("\n🔴 Testing SELL signal (Overbought + Bearish EMA)...")
    synthetic_data = create_synthetic_data(
        rsi_value=65,  # Overbought
        ema_fast=98,
        ema_slow=100,  # Bearish trend
        price=100,
        volume_ratio=1.2,
    )

    signal = strategy.generate_signal(synthetic_data)
    print(f"   Signal: {signal.action} (Confidence: {signal.confidence:.3f})")


def create_synthetic_data(rsi_value, ema_fast, ema_slow, price, volume_ratio):
    """Create synthetic market data for testing"""
    import numpy as np

    # Create 100 candles of synthetic data
    length = 100
    dates = pd.date_range(start="2024-01-01", periods=length, freq="1H")

    # Create price data that will result in desired RSI
    prices = np.random.normal(price, price * 0.01, length)

    # Manipulate last few prices to get desired RSI
    if rsi_value < 40:  # Want oversold
        prices[-10:] = prices[-10] * np.linspace(1.0, 0.95, 10)  # Declining prices
    elif rsi_value > 60:  # Want overbought
        prices[-10:] = prices[-10] * np.linspace(1.0, 1.05, 10)  # Rising prices

    volumes = np.random.normal(1000000, 100000, length)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": volumes,
        }
    )

    df.attrs["pair"] = "TESTUSDT"
    return df


async def check_system_configuration():
    """Check if system is properly configured"""
    print("\n⚙️ SYSTEM CONFIGURATION CHECK")
    print("=" * 40)

    from config.config import Config
    from user_settings import user_settings

    print(f"✅ Active Strategy: {Config.ACTIVE_STRATEGY}")
    print(f"✅ User Strategy: {user_settings.get_strategy()}")
    print(f"✅ Default Exchange: {Config.DEFAULT_EXCHANGE}")
    print(f"✅ Trading Pairs: {len(Config.TRADING_PAIRS)}")

    for i, pair in enumerate(Config.TRADING_PAIRS, 1):
        print(f"   {i}. {pair}")

    # Check risk management
    risk_settings = user_settings.get_risk_settings()
    print(f"\n🛡️ Risk Management:")
    print(f"   Trading Enabled: {risk_settings.get('trading_enabled', False)}")
    print(f"   Max Risk per Trade: {risk_settings.get('max_risk_per_trade', 0) * 100}%")
    print(f"   Max Open Positions: {risk_settings.get('max_open_positions', 0)}")


async def main():
    """Run full diagnostic"""
    await check_system_configuration()
    await test_enhanced_strategy_signals()

    print("\n" + "=" * 60)
    print("🔍 DIAGNOSTIC COMPLETE")
    print("\nIf no signals were found:")
    print("1. ⚠️ Current market conditions may not meet enhanced strategy criteria")
    print("2. 🔄 Try running debug_strategy_signals.py for live monitoring")
    print("3. 📊 Check if pairs are moving enough to trigger RSI thresholds")
    print("4. 🎛️ Consider temporarily lowering RSI thresholds for testing")


if __name__ == "__main__":
    asyncio.run(main())
