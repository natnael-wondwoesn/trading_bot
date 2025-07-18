#!/usr/bin/env python3
"""
VishvaAlgo Model Training Script
Train ML models for all trading pairs
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.ml_config import ML_CONFIG
from config.config import Config
from ml_utils.model_trainer import train_vishva_models, save_models
from strategy.strategies.vishva_ml_strategy import VishvaMLStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def get_historical_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Get historical data for training (implement based on your exchange)"""
    try:
        # Try to get data from your exchange client
        # This is a placeholder implementation - adapt based on your actual exchange clients

        logger.info(f"Attempting to get historical data for {symbol}...")

        try:
            # Try Bybit first
            from bybit.bybit_client import BybitClient

            if Config.BYBIT_API_KEY and Config.BYBIT_API_SECRET:
                client = BybitClient(Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET)
                # Calculate required bars (assuming 1h timeframe)
                bars_needed = days * 24

                data = await client.get_klines(symbol, "1h", bars_needed)
                if data is not None and len(data) > 0:
                    logger.info(
                        f"✅ Retrieved {len(data)} bars from Bybit for {symbol}"
                    )
                    return data

        except Exception as e:
            logger.warning(f"⚠️ Bybit data fetch failed: {e}")

        try:
            # Try MEXC as fallback
            from mexc.mexc_client import MEXCClient

            if Config.MEXC_API_KEY and Config.MEXC_API_SECRET:
                client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
                bars_needed = days * 24

                data = await client.get_klines(symbol, "1h", bars_needed)
                if data is not None and len(data) > 0:
                    logger.info(f"✅ Retrieved {len(data)} bars from MEXC for {symbol}")
                    return data

        except Exception as e:
            logger.warning(f"⚠️ MEXC data fetch failed: {e}")

        # Generate sample data as fallback for testing
        logger.warning(
            f"⚠️ No exchange data available, generating sample data for {symbol}"
        )
        return generate_sample_data(symbol, days * 24)

    except Exception as e:
        logger.error(f"❌ Error getting historical data for {symbol}: {e}")
        # Return sample data as last resort
        return generate_sample_data(symbol, days * 24)


def generate_sample_data(symbol: str, periods: int = 1000) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing"""
    import numpy as np

    logger.info(f"Generating {periods} periods of sample data for {symbol}")

    np.random.seed(42)

    # Set base price based on symbol
    if "BTC" in symbol:
        base_price = 45000
    elif "ETH" in symbol:
        base_price = 2500
    elif "SOL" in symbol:
        base_price = 100
    elif "ADA" in symbol:
        base_price = 0.5
    else:
        base_price = 1.0

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
        prices.append(max(new_price, base_price * 0.1))  # Prevent too low prices

        # Volume (higher during volatile periods)
        base_volume = 1000000
        vol_multiplier = 1 + abs(noise) * 10
        volume = base_volume * vol_multiplier * np.random.uniform(0.5, 2.0)
        volumes.append(volume)

    # Create OHLCV DataFrame
    df = pd.DataFrame({"close": prices[1:], "volume": volumes})  # Remove first price

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


async def train_models_for_all_pairs():
    """Train VishvaAlgo models for all trading pairs"""

    # Get trading pairs from config
    trading_pairs = Config.TRADING_PAIRS

    logger.info(
        f"🧠 Starting VishvaAlgo ML model training for {len(trading_pairs)} pairs"
    )
    logger.info(f"Trading pairs: {trading_pairs}")

    results = {"successful": [], "failed": [], "total_processed": 0}

    for symbol in trading_pairs:
        logger.info(f"🔄 Processing {symbol}...")
        results["total_processed"] += 1

        try:
            # Get historical data (1 year)
            data = await get_historical_data(symbol, days=365)

            if len(data) < ML_CONFIG["min_training_samples"]:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data)} bars")
                results["failed"].append(f"{symbol} - insufficient data")
                continue

            # Initialize strategy (this will load or create models)
            strategy = VishvaMLStrategy(symbol=symbol)

            # Train models
            logger.info(f"🤖 Training models for {symbol}...")
            success = strategy.train_models(data, retrain=True)

            if success:
                logger.info(f"✅ Successfully trained models for {symbol}")

                # Get model info
                info = strategy.get_strategy_info()
                logger.info(f"   📊 Model info for {symbol}:")
                logger.info(
                    f"      Training samples: {info['ml_models']['training_samples']}"
                )
                logger.info(
                    f"      Feature count: {info['ml_models']['feature_count']}"
                )
                logger.info(
                    f"      Ensemble available: {info['ml_models']['ensemble_available']}"
                )
                logger.info(
                    f"      Neural available: {info['ml_models']['neural_available']}"
                )

                results["successful"].append(symbol)
            else:
                logger.error(f"❌ Failed to train models for {symbol}")
                results["failed"].append(f"{symbol} - training failed")

        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            results["failed"].append(f"{symbol} - {str(e)}")

        # Small delay between training sessions
        await asyncio.sleep(2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs processed: {results['total_processed']}")
    logger.info(f"Successful: {len(results['successful'])}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results["successful"]:
        logger.info(f"\n✅ Successfully trained models for:")
        for symbol in results["successful"]:
            logger.info(f"   • {symbol}")

    if results["failed"]:
        logger.info(f"\n❌ Failed to train models for:")
        for failure in results["failed"]:
            logger.info(f"   • {failure}")

    success_rate = (len(results["successful"]) / results["total_processed"]) * 100
    logger.info(f"\n📈 Success rate: {success_rate:.1f}%")

    if success_rate >= 80:
        logger.info("🎉 Excellent! Most models trained successfully.")
    elif success_rate >= 60:
        logger.info("👍 Good! Majority of models trained successfully.")
    else:
        logger.warning("⚠️ Low success rate. Please check logs for issues.")


async def train_single_pair(symbol: str):
    """Train model for a single trading pair"""
    logger.info(f"🧠 Training VishvaAlgo ML model for {symbol}")

    try:
        # Get historical data
        data = await get_historical_data(symbol, days=365)

        if len(data) < ML_CONFIG["min_training_samples"]:
            logger.error(
                f"❌ Insufficient data for {symbol}: {len(data)} bars (minimum: {ML_CONFIG['min_training_samples']})"
            )
            return False

        # Initialize strategy
        strategy = VishvaMLStrategy(symbol=symbol)

        # Train models
        logger.info(f"🤖 Training models for {symbol}...")
        success = strategy.train_models(data, retrain=True)

        if success:
            logger.info(f"✅ Successfully trained models for {symbol}")

            # Get model info
            info = strategy.get_strategy_info()
            logger.info(f"📊 Model info for {symbol}:")
            logger.info(f"   Training samples: {info['ml_models']['training_samples']}")
            logger.info(f"   Feature count: {info['ml_models']['feature_count']}")
            logger.info(
                f"   Ensemble available: {info['ml_models']['ensemble_available']}"
            )
            logger.info(f"   Neural available: {info['ml_models']['neural_available']}")

            return True
        else:
            logger.error(f"❌ Failed to train models for {symbol}")
            return False

    except Exception as e:
        logger.error(f"❌ Error training {symbol}: {e}")
        return False


async def main():
    """Main training execution"""
    print("🧠 VISHVAALGO ML MODEL TRAINING")
    print("=" * 50)

    if len(sys.argv) > 1:
        # Train specific symbol
        symbol = sys.argv[1].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"

        print(f"Training models for specific pair: {symbol}")
        success = await train_single_pair(symbol)

        if success:
            print(f"\n🎉 Training completed successfully for {symbol}!")
            print("\n📋 Next Steps:")
            print("   1. Test the strategy with test_vishva_strategy.py")
            print("   2. Add VISHVA_ML to your user settings")
            print("   3. Monitor performance in live trading")
        else:
            print(f"\n❌ Training failed for {symbol}")
            print("   Check the logs above for error details")
    else:
        # Train all pairs
        print("Training models for all trading pairs...")
        await train_models_for_all_pairs()

        print("\n📋 Next Steps:")
        print("   1. Run: python test_vishva_strategy.py")
        print("   2. Add VISHVA_ML to your user settings")
        print("   3. Test with paper trading first")
        print("   4. Monitor ML model performance")
        print("   5. Retrain models weekly for optimal performance")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
    except Exception as e:
        logger.error(f"❌ Training script error: {e}")
        print(f"\n❌ Error: {e}")
        print("Check the logs above for details")
