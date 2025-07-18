#!/usr/bin/env python3
"""
VishvaAlgo ML Strategy Testing Script
Test the ML strategy with sample data and validate performance
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.strategies.vishva_ml_strategy import VishvaMLStrategy
from config.ml_config import ML_CONFIG
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VishvaStrategyTester:
    """Test VishvaAlgo ML Strategy"""

    def __init__(self):
        self.test_symbols = Config.TRADING_PAIRS[:3]  # Test first 3 symbols

    def create_sample_data(self, periods: int = 1000) -> pd.DataFrame:
        """Create realistic sample OHLCV data for testing"""
        np.random.seed(42)

        # Generate realistic price movement
        base_price = 45000  # Starting price for BTC-like asset
        prices = [base_price]
        volumes = []

        for i in range(periods):
            # Add trend and noise
            trend = 0.0001 * np.sin(i * 0.01)  # Subtle long-term trend
            volatility = 0.02 + 0.01 * np.sin(i * 0.1)  # Variable volatility
            noise = np.random.normal(0, volatility)

            # Price movement
            price_change = trend + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1.0))  # Prevent negative prices

            # Volume (higher during volatile periods)
            base_volume = 1000000
            vol_multiplier = 1 + abs(noise) * 10
            volume = base_volume * vol_multiplier * np.random.uniform(0.5, 2.0)
            volumes.append(volume)

        # Create OHLCV DataFrame
        df = pd.DataFrame(
            {"close": prices[1:], "volume": volumes}  # Remove first price
        )

        # Generate OHLC from close prices
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

        # Add realistic high/low based on volatility
        volatility_series = df["close"].pct_change().rolling(20).std().fillna(0.02)
        high_factor = 1 + volatility_series * np.random.uniform(0.5, 1.5, len(df))
        low_factor = 1 - volatility_series * np.random.uniform(0.5, 1.5, len(df))

        df["high"] = df[["open", "close"]].max(axis=1) * high_factor
        df["low"] = df[["open", "close"]].min(axis=1) * low_factor

        # Reorder columns
        df = df[["open", "high", "low", "close", "volume"]]

        return df

    async def test_strategy_creation(self):
        """Test strategy creation and initialization"""
        print("\n🧪 Testing Strategy Creation")
        print("-" * 40)

        for symbol in self.test_symbols:
            try:
                strategy = VishvaMLStrategy(symbol=symbol)
                info = strategy.get_strategy_info()

                print(f"✅ {symbol}: Strategy created successfully")
                print(f"   Type: {info['type']} v{info['version']}")
                print(
                    f"   Risk params: SL={info['risk_management']['stop_loss']:.1%}, "
                    f"TP={info['risk_management']['take_profit']:.1%}"
                )

            except Exception as e:
                print(f"❌ {symbol}: Error creating strategy - {e}")

    async def test_feature_engineering(self):
        """Test feature engineering with sample data"""
        print("\n🔧 Testing Feature Engineering")
        print("-" * 40)

        data = self.create_sample_data(500)
        strategy = VishvaMLStrategy(symbol="BTCUSDT")

        try:
            indicators = strategy.calculate_indicators(data)

            print(f"✅ Feature calculation successful")
            print(
                f"   Features generated: {len([k for k in indicators.keys() if not k.startswith('ml_') and k != 'timestamp'])}"
            )
            print(f"   Sample features:")

            # Show sample of important features
            key_features = [
                "rsi_14",
                "ema_21",
                "macd_line",
                "bb_position",
                "volume_trend",
                "atr",
            ]
            for feature in key_features:
                if feature in indicators:
                    value = indicators[feature]
                    if isinstance(value, (int, float)):
                        print(f"      {feature}: {value:.4f}")
                    else:
                        print(f"      {feature}: {value}")

        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")

    async def test_model_training(self):
        """Test model training with sample data"""
        print("\n🤖 Testing Model Training")
        print("-" * 40)

        # Create larger dataset for training
        data = self.create_sample_data(2000)

        for symbol in ["BTCUSDT"]:  # Test with one symbol
            try:
                print(f"Training models for {symbol}...")
                strategy = VishvaMLStrategy(symbol=symbol)

                # Train models
                success = strategy.train_models(data, retrain=True)

                if success:
                    print(f"✅ {symbol}: Model training successful")

                    # Get model info
                    info = strategy.get_strategy_info()
                    ml_info = info["ml_models"]
                    print(f"   Ensemble available: {ml_info['ensemble_available']}")
                    print(f"   Neural network available: {ml_info['neural_available']}")
                    print(f"   Feature count: {ml_info['feature_count']}")
                    print(f"   Training samples: {ml_info['training_samples']}")

                else:
                    print(f"❌ {symbol}: Model training failed")

            except Exception as e:
                print(f"❌ {symbol}: Training error - {e}")

    async def test_signal_generation(self):
        """Test signal generation with trained models"""
        print("\n📊 Testing Signal Generation")
        print("-" * 40)

        # Create test data
        data = self.create_sample_data(300)

        for symbol in self.test_symbols:
            try:
                strategy = VishvaMLStrategy(symbol=symbol)

                # Generate signal
                signal = strategy.generate_signal(data)

                print(f"📈 {symbol} Signal:")
                print(f"   Action: {signal.action}")
                print(f"   Confidence: {signal.confidence:.2%}")
                print(f"   Current Price: ${signal.current_price:.2f}")

                if signal.action != "HOLD":
                    if signal.stop_loss:
                        print(f"   Stop Loss: ${signal.stop_loss:.2f}")
                    if signal.take_profit:
                        print(f"   Take Profit: ${signal.take_profit:.2f}")
                    if signal.risk_reward:
                        print(f"   Risk/Reward: 1:{signal.risk_reward:.2f}")

                # Show ML-specific indicators
                ml_indicators = signal.indicators
                if "ml_probabilities" in ml_indicators:
                    probs = ml_indicators["ml_probabilities"]
                    print(
                        f"   ML Probabilities: Neutral={probs['neutral']:.2%}, "
                        f"Long={probs['long']:.2%}, Short={probs['short']:.2%}"
                    )

                if "model_status" in ml_indicators:
                    print(f"   Model Status: {ml_indicators['model_status']}")

                print()

            except Exception as e:
                print(f"❌ {symbol}: Signal generation error - {e}")

    async def test_performance_tracking(self):
        """Test performance tracking functionality"""
        print("\n📈 Testing Performance Tracking")
        print("-" * 40)

        strategy = VishvaMLStrategy(symbol="BTCUSDT")

        # Simulate some predictions and outcomes
        for i in range(10):
            # Simulate random prediction outcome
            correct = np.random.choice([True, False], p=[0.8, 0.2])  # 80% success rate
            strategy.update_performance(correct)

        metrics = strategy.performance_metrics
        print(f"✅ Performance tracking test:")
        print(f"   Total predictions: {metrics['total_predictions']}")
        print(f"   Correct predictions: {metrics['correct_predictions']}")
        print(f"   Win rate: {metrics['win_rate']:.1%}")

    async def test_model_loading_saving(self):
        """Test model loading and saving functionality"""
        print("\n💾 Testing Model Loading/Saving")
        print("-" * 40)

        try:
            # Create strategy and train a simple model
            data = self.create_sample_data(1500)
            strategy1 = VishvaMLStrategy(symbol="ETHUSDT")

            # Train models
            success = strategy1.train_models(data, retrain=True)

            if success:
                print("✅ Model training successful")

                # Create new strategy instance to test loading
                strategy2 = VishvaMLStrategy(symbol="ETHUSDT")

                if strategy2.models["ensemble_model"] is not None:
                    print("✅ Model loading successful")

                    # Test that both strategies give same prediction
                    signal1 = strategy1.generate_signal(data.tail(200))
                    signal2 = strategy2.generate_signal(data.tail(200))

                    if signal1.action == signal2.action:
                        print("✅ Model consistency verified")
                    else:
                        print("⚠️ Model predictions differ (could be due to randomness)")

                else:
                    print("❌ Model loading failed")
            else:
                print("❌ Model training failed")

        except Exception as e:
            print(f"❌ Model loading/saving test failed: {e}")

    async def test_multi_symbol_performance(self):
        """Test performance across multiple symbols"""
        print("\n🔄 Testing Multi-Symbol Performance")
        print("-" * 40)

        results = {}

        for symbol in self.test_symbols:
            try:
                strategy = VishvaMLStrategy(symbol=symbol)
                data = self.create_sample_data(500)

                # Generate multiple signals
                signals = []
                for i in range(5):
                    subset = data.iloc[
                        i * 80 : (i + 1) * 100 + 100
                    ]  # Overlapping windows
                    if len(subset) >= 200:
                        signal = strategy.generate_signal(subset)
                        signals.append(signal)

                # Calculate signal distribution
                actions = [s.action for s in signals]
                action_counts = {
                    action: actions.count(action) for action in set(actions)
                }
                avg_confidence = np.mean([s.confidence for s in signals])

                results[symbol] = {
                    "signals": len(signals),
                    "actions": action_counts,
                    "avg_confidence": avg_confidence,
                }

                print(f"📊 {symbol} Results:")
                print(f"   Signals generated: {len(signals)}")
                print(f"   Action distribution: {action_counts}")
                print(f"   Average confidence: {avg_confidence:.2%}")

            except Exception as e:
                print(f"❌ {symbol}: Multi-symbol test failed - {e}")
                results[symbol] = {"error": str(e)}

        return results

    async def run_comprehensive_test(self):
        """Run all tests"""
        print("🧠 VISHVAALGO ML STRATEGY - COMPREHENSIVE TEST")
        print("=" * 60)

        # Run all test components
        await self.test_strategy_creation()
        await self.test_feature_engineering()
        await self.test_model_training()
        await self.test_signal_generation()
        await self.test_performance_tracking()
        await self.test_model_loading_saving()
        results = await self.test_multi_symbol_performance()

        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE TEST COMPLETED!")

        # Summary
        print("\n📋 Test Summary:")
        successful_symbols = sum(1 for r in results.values() if "error" not in r)
        total_symbols = len(results)

        print(f"   Symbols tested: {total_symbols}")
        print(f"   Successful: {successful_symbols}")
        print(
            f"   Success rate: {successful_symbols/total_symbols*100:.1f}%"
            if total_symbols > 0
            else "   Success rate: 0%"
        )

        print("\n📋 Next Steps:")
        print("   1. Run: python train_vishva_models.py")
        print("   2. Add VISHVA_ML to your user settings")
        print("   3. Test with paper trading first")
        print("   4. Monitor ML model performance")
        print("   5. Retrain models weekly for optimal performance")


async def test_specific_symbol(symbol: str):
    """Test specific symbol"""
    print(f"🧠 TESTING VISHVAALGO ML STRATEGY FOR {symbol}")
    print("=" * 50)

    try:
        strategy = VishvaMLStrategy(symbol=symbol)
        data = VishvaStrategyTester().create_sample_data(1000)

        print("🧪 Testing strategy components...")

        # Test feature calculation
        indicators = strategy.calculate_indicators(data)
        print(f"✅ Features calculated: {len(indicators)}")

        # Test signal generation
        signal = strategy.generate_signal(data)
        print(
            f"✅ Signal generated: {signal.action} (confidence: {signal.confidence:.2%})"
        )

        # Test model training
        print(f"🤖 Training models for {symbol}...")
        success = strategy.train_models(data, retrain=True)

        if success:
            print(f"✅ Model training successful")

            # Test signal after training
            signal_trained = strategy.generate_signal(data)
            print(
                f"✅ Post-training signal: {signal_trained.action} (confidence: {signal_trained.confidence:.2%})"
            )

            # Show model info
            info = strategy.get_strategy_info()
            ml_info = info["ml_models"]
            print(f"\n📊 Model Information:")
            print(f"   Training samples: {ml_info['training_samples']}")
            print(f"   Feature count: {ml_info['feature_count']}")
            print(f"   Ensemble model: {ml_info['ensemble_available']}")
            print(f"   Neural network: {ml_info['neural_available']}")

        else:
            print(f"❌ Model training failed")

        print(f"\n🎉 Testing completed for {symbol}!")

    except Exception as e:
        print(f"❌ Error testing {symbol}: {e}")


async def main():
    """Main test execution"""
    print("🧠 VISHVAALGO ML STRATEGY TESTING")
    print("=" * 50)

    if len(sys.argv) > 1:
        # Test specific symbol
        symbol = sys.argv[1].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        await test_specific_symbol(symbol)
    else:
        # Run comprehensive test
        tester = VishvaStrategyTester()
        await tester.run_comprehensive_test()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Testing stopped by user")
    except Exception as e:
        logger.error(f"❌ Testing script error: {e}")
        print(f"\n❌ Error: {e}")
        print("Check the logs above for details")
