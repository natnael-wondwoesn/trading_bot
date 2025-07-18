#!/usr/bin/env python3
"""
Fixed Integration Script for Enhanced RSI EMA Strategy
Handles Unicode encoding issues and completes integration
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List


def safe_read_file(file_path):
    """Safely read file with proper encoding handling"""
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            print(f"   ✅ Successfully read file with {encoding} encoding")
            return content, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"   ❌ Error with {encoding}: {e}")
            continue

    print(f"   ❌ Could not read file with any encoding")
    return None, None


def safe_write_file(file_path, content, encoding="utf-8"):
    """Safely write file with proper encoding"""
    try:
        with open(file_path, "w", encoding=encoding, newline="") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"   ❌ Error writing file: {e}")
        return False


def update_user_settings():
    """Update user settings with proper encoding handling"""
    print("\n👤 UPDATING USER SETTINGS")
    print("=" * 50)

    settings_file = "user_settings.py"

    if os.path.exists(settings_file):
        # Read with safe encoding
        content, encoding = safe_read_file(settings_file)
        if content is None:
            print(f"❌ Could not read {settings_file}")
            return False

        # Update default strategy
        if '"strategy": "RSI_EMA"' in content:
            content = content.replace(
                '"strategy": "RSI_EMA"', '"strategy": "ENHANCED_RSI_EMA"'
            )
            print("✅ Updated default strategy in user_settings.py")

            # Write back with same encoding
            if safe_write_file(settings_file, content, encoding):
                print("✅ Successfully saved user_settings.py")
            else:
                return False
        else:
            print("ℹ️ Strategy already set or not found in user_settings.py")
    else:
        print("ℹ️ user_settings.py not found - will use config default")

    # Check for user_settings.json
    json_settings_file = "user_settings.json"
    if os.path.exists(json_settings_file):
        try:
            with open(json_settings_file, "r", encoding="utf-8") as f:
                settings = json.load(f)

            settings["strategy"] = "ENHANCED_RSI_EMA"
            settings["last_updated"] = datetime.now().isoformat()

            with open(json_settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)

            print("✅ Updated user_settings.json")
        except Exception as e:
            print(f"⚠️ Error updating user_settings.json: {e}")

    return True


def create_enhanced_strategy_file():
    """Create the enhanced strategy file with the complete implementation"""
    print("\n📁 CREATING ENHANCED STRATEGY FILE")
    print("=" * 50)

    strategy_file = "strategy/strategies/enhanced_rsi_ema_strategy.py"

    # The complete Enhanced RSI EMA Strategy implementation
    enhanced_strategy_code = '''#!/usr/bin/env python3
"""
Enhanced RSI EMA Strategy - Production Implementation
Enhanced version with practical thresholds and weighted scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from strategy.strategies.strategy import Strategy
from models.models import Signal
from indicators import calculate_rsi, calculate_ema, calculate_atr


class EnhancedRSIEMAStrategy(Strategy):
    """
    Enhanced RSI + EMA Strategy with more practical signal generation
    
    Key Improvements over original RSI_EMA Strategy:
    - More practical RSI thresholds (40/60 instead of extreme 35/65)
    - Weighted scoring system instead of all-or-nothing approach
    - Lower confidence threshold (0.4 instead of 0.7) for more signals
    - Less strict volume requirements (60% instead of 80% of average)
    - Multiple signal types with crossover detection
    - Enhanced momentum analysis and trend confirmation
    - Better risk management with ATR-based stops
    """
    
    def __init__(self, rsi_period: int = 14, ema_fast: int = 9, ema_slow: int = 21):
        super().__init__("Enhanced RSI + EMA Strategy")
        self.rsi_period = rsi_period
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow
        
        # Enhanced RSI thresholds - more practical than extreme levels
        self.rsi_oversold = 40          # More practical than 35
        self.rsi_overbought = 60        # More practical than 65
        self.rsi_strong_oversold = 30   # For high confidence signals
        self.rsi_strong_overbought = 70 # For high confidence signals
        
        # Lower confidence threshold for more signal generation
        self.min_confidence = 0.4       # Reduced from 0.7
        
        # Less strict volume requirement
        self.volume_multiplier = 0.6    # Reduced from 0.8

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate RSI and EMA indicators with enhanced analysis"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format - missing required columns")

        close_prices = data["close"]

        # Calculate core indicators
        rsi = calculate_rsi(close_prices, self.rsi_period)
        ema_fast = calculate_ema(close_prices, self.ema_fast_period)
        ema_slow = calculate_ema(close_prices, self.ema_slow_period)

        # Volume analysis
        volume_sma = data["volume"].rolling(window=20).mean()

        # Previous values for trend analysis and crossover detection
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else rsi.iloc[-1]
        prev_ema_fast = ema_fast.iloc[-2] if len(ema_fast) > 1 else ema_fast.iloc[-1]
        prev_ema_slow = ema_slow.iloc[-2] if len(ema_slow) > 1 else ema_slow.iloc[-1]

        # Store all indicators for signal generation
        self.indicators = {
            "rsi": rsi.iloc[-1],
            "prev_rsi": prev_rsi,
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "prev_ema_fast": prev_ema_fast,
            "prev_ema_slow": prev_ema_slow,
            "current_price": close_prices.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "avg_volume": volume_sma.iloc[-1],
        }

        return self.indicators

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal with enhanced weighted scoring system"""
        indicators = self.calculate_indicators(data)

        # Initialize signal strength scoring system
        signal_strength = {"buy": 0, "sell": 0, "reasons": []}

        # 1. RSI Signals with multiple confidence levels
        if indicators["rsi"] <= self.rsi_strong_oversold:
            signal_strength["buy"] += 0.4
            signal_strength["reasons"].append("RSI strongly oversold")
        elif indicators["rsi"] <= self.rsi_oversold:
            signal_strength["buy"] += 0.2
            signal_strength["reasons"].append("RSI oversold")

        if indicators["rsi"] >= self.rsi_strong_overbought:
            signal_strength["sell"] += 0.4
            signal_strength["reasons"].append("RSI strongly overbought")
        elif indicators["rsi"] >= self.rsi_overbought:
            signal_strength["sell"] += 0.2
            signal_strength["reasons"].append("RSI overbought")

        # 2. EMA Trend Analysis with crossover detection
        ema_bullish = indicators["ema_fast"] > indicators["ema_slow"]
        ema_bearish = indicators["ema_fast"] < indicators["ema_slow"]

        # Detect EMA crossovers for stronger signals
        prev_ema_bullish = indicators["prev_ema_fast"] > indicators["prev_ema_slow"]
        ema_bullish_crossover = ema_bullish and not prev_ema_bullish
        ema_bearish_crossover = ema_bearish and not prev_ema_bullish

        if ema_bullish_crossover:
            signal_strength["buy"] += 0.3
            signal_strength["reasons"].append("EMA bullish crossover")
        elif ema_bullish:
            signal_strength["buy"] += 0.15
            signal_strength["reasons"].append("EMA bullish trend")

        if ema_bearish_crossover:
            signal_strength["sell"] += 0.3
            signal_strength["reasons"].append("EMA bearish crossover")
        elif ema_bearish:
            signal_strength["sell"] += 0.15
            signal_strength["reasons"].append("EMA bearish trend")

        # 3. Price position analysis relative to EMAs
        price_above_fast_ema = indicators["current_price"] > indicators["ema_fast"]
        price_below_fast_ema = indicators["current_price"] < indicators["ema_fast"]

        if price_above_fast_ema and ema_bullish:
            signal_strength["buy"] += 0.1
            signal_strength["reasons"].append("Price above bullish EMA")

        if price_below_fast_ema and ema_bearish:
            signal_strength["sell"] += 0.1
            signal_strength["reasons"].append("Price below bearish EMA")

        # 4. RSI momentum analysis (improving conditions)
        rsi_improving_for_buy = (
            indicators["rsi"] > indicators["prev_rsi"] and indicators["rsi"] < 50
        )
        rsi_improving_for_sell = (
            indicators["rsi"] < indicators["prev_rsi"] and indicators["rsi"] > 50
        )

        if rsi_improving_for_buy:
            signal_strength["buy"] += 0.1
            signal_strength["reasons"].append("RSI recovering from oversold")

        if rsi_improving_for_sell:
            signal_strength["sell"] += 0.1
            signal_strength["reasons"].append("RSI declining from overbought")

        # 5. Volume confirmation (less strict than original strategy)
        volume_ok = (
            indicators["volume"] > indicators["avg_volume"] * self.volume_multiplier
        )

        if volume_ok:
            signal_strength["buy"] += 0.05
            signal_strength["sell"] += 0.05
            signal_strength["reasons"].append("Volume confirmation")

        # Calculate final confidence scores
        buy_confidence = min(signal_strength["buy"], 1.0)
        sell_confidence = min(signal_strength["sell"], 1.0)

        # Determine final action based on enhanced confidence threshold
        if buy_confidence >= self.min_confidence and buy_confidence > sell_confidence:
            action = "BUY"
            confidence = buy_confidence
        elif sell_confidence >= self.min_confidence and sell_confidence > buy_confidence:
            action = "SELL"
            confidence = sell_confidence
        else:
            action = "HOLD"
            confidence = max(buy_confidence, sell_confidence)

        # Calculate risk management levels using ATR
        atr = calculate_atr(data["high"], data["low"], data["close"]).iloc[-1]

        stop_loss = None
        take_profit = None
        risk_reward = None

        if action == "BUY":
            stop_loss = indicators["current_price"] - (1.5 * atr)  # Tighter stop
            take_profit = indicators["current_price"] + (2.5 * atr)  # Better R:R
            risk_reward = 1.67
        elif action == "SELL":
            stop_loss = indicators["current_price"] + (1.5 * atr)
            take_profit = indicators["current_price"] - (2.5 * atr)
            risk_reward = 1.67

        # Create comprehensive signal with all metadata
        return Signal(
            pair=data.attrs.get("pair", "UNKNOWN"),
            action=action,
            confidence=confidence,
            current_price=indicators["current_price"],
            timestamp=datetime.now(),
            indicators={
                "rsi": indicators["rsi"],
                "prev_rsi": indicators["prev_rsi"],
                "ema_trend": "bullish" if ema_bullish else "bearish",
                "ema_fast": indicators["ema_fast"],
                "ema_slow": indicators["ema_slow"],
                "volume_confirmation": volume_ok,
                "volatility": "high" if atr > data["close"].std() else "normal",
                "signal_reasons": signal_strength["reasons"],
                "buy_strength": buy_confidence,
                "sell_strength": sell_confidence,
                "crossover_detected": ema_bullish_crossover or ema_bearish_crossover,
                "rsi_momentum": "improving" if rsi_improving_for_buy or rsi_improving_for_sell else "stable"
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )

    def get_strategy_info(self) -> Dict:
        """Get comprehensive strategy information for monitoring and debugging"""
        return {
            "name": self.name,
            "type": "ENHANCED_RSI_EMA",
            "version": "2.0",
            "parameters": {
                "rsi_period": self.rsi_period,
                "ema_fast": self.ema_fast_period,
                "ema_slow": self.ema_slow_period,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "rsi_strong_oversold": self.rsi_strong_oversold,
                "rsi_strong_overbought": self.rsi_strong_overbought,
                "min_confidence": self.min_confidence,
                "volume_multiplier": self.volume_multiplier
            },
            "features": [
                "Practical RSI thresholds (40/60)",
                "Weighted scoring system",
                "Multiple signal confidence levels",
                "EMA crossover detection",
                "Volume confirmation",
                "ATR-based risk management",
                "Enhanced momentum analysis",
                "Improved signal frequency"
            ],
            "improvements": [
                "More practical RSI levels vs extreme 35/65",
                "Lower confidence threshold (0.4 vs 0.7)",
                "Less strict volume requirements (60% vs 80%)",
                "Multiple signal types instead of all-or-nothing",
                "Better crossover detection and trend analysis",
                "Enhanced risk-reward ratios with ATR stops"
            ]
        }

    def validate_signal(self, signal: Signal) -> bool:
        """Validate generated signal for consistency"""
        try:
            # Basic validation
            if signal.action not in ["BUY", "SELL", "HOLD"]:
                return False
            
            if not (0 <= signal.confidence <= 1):
                return False
            
            # Ensure signal has required indicators
            required_indicators = ["rsi", "ema_trend", "signal_reasons"]
            if not all(key in signal.indicators for key in required_indicators):
                return False
            
            # Validate RSI range
            if not (0 <= signal.indicators["rsi"] <= 100):
                return False
            
            # Validate risk management levels
            if signal.action in ["BUY", "SELL"]:
                if signal.stop_loss is None or signal.take_profit is None:
                    return False
            
            return True
            
        except Exception:
            return False
'''

    # Write the enhanced strategy file
    try:
        os.makedirs(os.path.dirname(strategy_file), exist_ok=True)
        with open(strategy_file, "w", encoding="utf-8") as f:
            f.write(enhanced_strategy_code)
        print(f"✅ Created enhanced strategy file: {strategy_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating strategy file: {e}")
        return False


def test_integration():
    """Test the Enhanced strategy integration"""
    print("\n🧪 TESTING INTEGRATION")
    print("=" * 50)

    try:
        # Test strategy import
        from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy

        print("✅ Enhanced strategy import successful")

        # Test strategy creation
        strategy = EnhancedRSIEMAStrategy()
        print(f"✅ Strategy creation successful: {strategy.name}")

        # Test strategy info
        info = strategy.get_strategy_info()
        print(f"✅ Strategy info: {info['type']} v{info['version']}")

        # Test with sample data
        import pandas as pd
        import numpy as np

        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(105, 115, 100),
                "low": np.random.uniform(95, 105, 100),
                "close": np.random.uniform(98, 108, 100),
                "volume": np.random.uniform(1000000, 5000000, 100),
            }
        )
        data.attrs["pair"] = "BTCUSDT"

        # Test signal generation
        signal = strategy.generate_signal(data)
        print(
            f"✅ Signal generation successful: {signal.action} (confidence: {signal.confidence:.2%})"
        )

        # Test strategy factory (if available)
        try:
            from services.trading_orchestrator import StrategyFactory

            factory_strategy = StrategyFactory.create_strategy("ENHANCED_RSI_EMA")
            print(f"✅ StrategyFactory creation successful: {factory_strategy.name}")
        except Exception as e:
            print(f"⚠️ StrategyFactory test failed: {e}")
            print("   You may need to manually update the StrategyFactory")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(
            "   Make sure the enhanced_rsi_ema_strategy.py file is in the correct location"
        )
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def create_final_test_script():
    """Create a comprehensive test script"""
    test_script = '''#!/usr/bin/env python3
"""
Enhanced RSI EMA Strategy - Final Integration Test
Run this to verify everything is working correctly
"""

def main():
    print("🧪 ENHANCED RSI EMA STRATEGY - FINAL TEST")
    print("=" * 55)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Strategy Import
    print("\\n1. Testing strategy import...")
    total_tests += 1
    try:
        from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
        print("   ✅ Import successful")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
    
    # Test 2: Strategy Creation
    print("\\n2. Testing strategy creation...")
    total_tests += 1
    try:
        strategy = EnhancedRSIEMAStrategy()
        print(f"   ✅ Created: {strategy.name}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Creation failed: {e}")
        return
    
    # Test 3: Configuration
    print("\\n3. Testing configuration...")
    total_tests += 1
    try:
        from config.config import Config
        active_strategy = getattr(Config, 'ACTIVE_STRATEGY', 'Not found')
        print(f"   ✅ Active strategy: {active_strategy}")
        if active_strategy == "ENHANCED_RSI_EMA":
            tests_passed += 1
        else:
            print("   ⚠️ Active strategy is not ENHANCED_RSI_EMA")
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
    
    # Test 4: Strategy Factory
    print("\\n4. Testing strategy factory...")
    total_tests += 1
    try:
        from services.trading_orchestrator import StrategyFactory
        available = StrategyFactory.get_available_strategies()
        if "ENHANCED_RSI_EMA" in available:
            print("   ✅ Strategy available in factory")
            tests_passed += 1
        else:
            print("   ⚠️ Strategy not in factory - manual update needed")
    except Exception as e:
        print(f"   ❌ Factory test failed: {e}")
    
    # Test 5: Signal Generation
    print("\\n5. Testing signal generation...")
    total_tests += 1
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
            'close': np.random.uniform(98, 108, 50),
            'volume': np.random.uniform(1000000, 5000000, 50)
        })
        data.attrs['pair'] = 'BTCUSDT'
        
        signal = strategy.generate_signal(data)
        print(f"   ✅ Signal: {signal.action} (confidence: {signal.confidence:.1%})")
        print(f"      Reasons: {len(signal.indicators.get('signal_reasons', []))} reasons")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Signal test failed: {e}")
    
    # Summary
    print("\\n" + "=" * 55)
    print(f"📊 TEST SUMMARY: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Enhanced RSI EMA Strategy is fully integrated!")
        print("\\n📋 Next Steps:")
        print("   1. Restart your trading bot")
        print("   2. Check Telegram bot settings")
        print("   3. Monitor signal generation")
        print("   4. Test with paper trading first")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("   Please review the failed tests above")
        if tests_passed >= 3:
            print("   Most tests passed - strategy should work")

if __name__ == "__main__":
    main()
'''

    with open("test_enhanced_final.py", "w", encoding="utf-8") as f:
        f.write(test_script)

    print(f"📄 Created final test script: test_enhanced_final.py")


async def main():
    """Run complete integration with fixed encoding handling"""
    print("🚀 ENHANCED RSI EMA STRATEGY - FIXED INTEGRATION")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Create the enhanced strategy file
    if not create_enhanced_strategy_file():
        print("❌ Failed to create strategy file. Stopping integration.")
        return

    # Step 2: Update user settings with proper encoding
    if not update_user_settings():
        print("⚠️ User settings update had issues, but continuing...")

    # Step 3: Test integration
    if test_integration():
        print("✅ Integration tests passed")
    else:
        print("❌ Integration tests failed")
        return

    # Step 4: Create final test script
    create_final_test_script()

    print("\n🎉 INTEGRATION COMPLETED!")
    print()
    print("📋 Summary of Changes:")
    print("   ✅ Created enhanced_rsi_ema_strategy.py")
    print("   ✅ Updated user settings (if found)")
    print("   ✅ Created test scripts")
    print()
    print("📋 Next Steps:")
    print("   1. Run: python test_enhanced_final.py")
    print("   2. Manually update StrategyFactory if needed")
    print("   3. Restart your trading bot")
    print("   4. Test in Telegram bot settings")
    print()
    print("🚀 Enhanced RSI EMA Strategy is ready!")


if __name__ == "__main__":
    asyncio.run(main())
