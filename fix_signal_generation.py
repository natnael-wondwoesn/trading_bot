#!/usr/bin/env python3
"""
Fix Signal Generation Issues
Identifies and fixes the DataFrame validation problem
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime


async def test_strategy_with_bybit():
    """Test strategy with real Bybit data"""
    print("🔧 FIXING SIGNAL GENERATION ISSUES")
    print("=" * 50)

    try:
        from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
        from bybit.bybit_client import BybitClient
        from config.config import Config

        print("✅ Imports successful")

        # Initialize strategy
        strategy = EnhancedRSIEMAStrategy()
        print(f"✅ Strategy initialized: {strategy.name}")

        # Get real Bybit data
        client = BybitClient(
            Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET, testnet=Config.BYBIT_TESTNET
        )

        print("\n📊 Testing with BTCUSDT...")

        # Get the raw response first
        response = await client._request(
            "GET",
            "/v5/market/kline",
            {
                "category": "spot",
                "symbol": "BTCUSDT",
                "interval": "60",  # 1h
                "limit": 100,
            },
        )

        print(f"✅ Raw API response received")
        print(f"   Result count: {len(response.get('result', {}).get('list', []))}")

        # Convert manually to debug
        klines_data = response.get("result", {}).get("list", [])
        if not klines_data:
            print("❌ No klines data received")
            return

        print(f"✅ Klines data: {len(klines_data)} candles")
        print(f"   First candle: {klines_data[0]}")

        # Create DataFrame manually
        df = pd.DataFrame(
            klines_data,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )

        print(f"✅ DataFrame created with shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        # Convert data types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print(f"✅ Data types converted")
        print(f"   Sample data:\n{df.head(2)}")

        # Test data validation
        print(f"\n🔍 Testing data validation...")
        required_columns = ["open", "high", "low", "close", "volume"]
        validation_result = all(col in df.columns for col in required_columns)
        print(f"   Validation result: {validation_result}")

        # Set pair attribute
        df.attrs["pair"] = "BTCUSDT"

        # Test each component step by step
        print(f"\n🧪 Testing strategy components...")

        # Test calculate_indicators
        try:
            print("   Testing calculate_indicators...")
            indicators = strategy.calculate_indicators(df)
            print(f"   ✅ Indicators calculated successfully")
            print(f"      RSI: {indicators['rsi']:.2f}")
            print(f"      EMA Fast: {indicators['ema_fast']:.2f}")
            print(f"      EMA Slow: {indicators['ema_slow']:.2f}")
            print(f"      Current Price: {indicators['current_price']:.2f}")
        except Exception as e:
            print(f"   ❌ calculate_indicators failed: {e}")
            return

        # Test generate_signal
        try:
            print("   Testing generate_signal...")
            signal = strategy.generate_signal(df)
            print(f"   ✅ Signal generated successfully!")
            print(f"      Action: {signal.action}")
            print(f"      Confidence: {signal.confidence:.3f}")
            print(f"      Current Price: {signal.current_price:.2f}")

            if signal.action != "HOLD":
                print(
                    f"🎯 SIGNAL FOUND: {signal.action} BTCUSDT at ${signal.current_price:.2f}"
                )
                print(f"   Stop Loss: ${signal.stop_loss:.2f}")
                print(f"   Take Profit: ${signal.take_profit:.2f}")
                print(f"   Reason: RSI={indicators['rsi']:.1f}")
            else:
                print(f"⚪ No signal - RSI: {indicators['rsi']:.1f} (need <40 or >60)")

        except Exception as e:
            print(f"   ❌ generate_signal failed: {e}")
            import traceback

            print(f"   Traceback: {traceback.format_exc()}")
            return

        await client.close()
        print(f"\n✅ Test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")


async def test_all_pairs():
    """Test signal generation on all configured pairs"""
    print("\n🌍 TESTING ALL PAIRS FOR SIGNALS")
    print("=" * 40)

    try:
        from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
        from bybit.bybit_client import BybitClient
        from config.config import Config

        strategy = EnhancedRSIEMAStrategy()
        client = BybitClient(
            Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET, testnet=Config.BYBIT_TESTNET
        )

        signals_found = 0

        for pair in Config.TRADING_PAIRS:
            try:
                print(f"\n📈 {pair}...")

                # Get data using the fixed method
                response = await client._request(
                    "GET",
                    "/v5/market/kline",
                    {
                        "category": "spot",
                        "symbol": pair,
                        "interval": "60",
                        "limit": 100,
                    },
                )

                klines_data = response.get("result", {}).get("list", [])
                if len(klines_data) < 50:
                    print(f"   ❌ Insufficient data ({len(klines_data)} candles)")
                    continue

                # Create and process DataFrame
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

                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df.attrs["pair"] = pair

                # Calculate indicators
                indicators = strategy.calculate_indicators(df)
                current_rsi = indicators["rsi"]
                current_price = indicators["current_price"]

                # Generate signal
                signal = strategy.generate_signal(df)

                print(
                    f"   Price: ${current_price:,.4f} | RSI: {current_rsi:.1f} | Signal: {signal.action}"
                )

                if signal.action != "HOLD":
                    signals_found += 1
                    print(
                        f"   🎯 SIGNAL: {signal.action} (Confidence: {signal.confidence:.3f})"
                    )
                    print(f"      Stop Loss: ${signal.stop_loss:,.4f}")
                    print(f"      Take Profit: ${signal.take_profit:,.4f}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        await client.close()

        print(f"\n📊 SUMMARY:")
        print(f"   Pairs tested: {len(Config.TRADING_PAIRS)}")
        print(f"   Signals found: {signals_found}")

        if signals_found == 0:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Current market may be in neutral RSI zones")
            print(f"   • Enhanced strategy uses practical thresholds (RSI 40/60)")
            print(f"   • Try running during more volatile market hours")
            print(f"   • Monitor debug_strategy_signals.py for live updates")

    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":

    async def main():
        await test_strategy_with_bybit()
        await test_all_pairs()

    asyncio.run(main())
