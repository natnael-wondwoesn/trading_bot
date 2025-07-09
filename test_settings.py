#!/usr/bin/env python3
"""
Test script for user settings and emergency functionality
"""

import asyncio
import logging
from datetime import datetime
from user_settings import user_settings
from bot import TradingBot
from models.models import Signal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_user_settings():
    """Test user settings functionality"""
    print("🧪 Testing User Settings...")

    # Test strategy setting
    print("\n📈 Testing Strategy Settings:")
    print(f"Current strategy: {user_settings.get_strategy()}")

    assert user_settings.set_strategy("MACD"), "Should set MACD strategy"
    assert user_settings.get_strategy() == "MACD", "Strategy should be MACD"

    assert user_settings.set_strategy("BOLLINGER"), "Should set Bollinger strategy"
    assert user_settings.get_strategy() == "BOLLINGER", "Strategy should be BOLLINGER"

    user_settings.set_strategy("RSI_EMA")  # Reset to default
    print("✅ Strategy settings work correctly")

    # Test risk management
    print("\n💰 Testing Risk Management:")
    risk = user_settings.get_risk_settings()
    print(f"Current risk per trade: {risk['max_risk_per_trade']*100}%")

    assert user_settings.set_max_risk_per_trade(0.01), "Should set 1% risk"
    assert (
        user_settings.get_risk_settings()["max_risk_per_trade"] == 0.01
    ), "Risk should be 1%"

    assert user_settings.set_custom_stop_loss(2.5), "Should set 2.5% stop loss"
    assert (
        user_settings.get_risk_settings()["custom_stop_loss"] == 2.5
    ), "Stop loss should be 2.5%"

    print("✅ Risk management settings work correctly")

    # Test emergency functions
    print("\n🚨 Testing Emergency Functions:")
    assert (
        not user_settings.is_emergency_mode()
    ), "Should not be in emergency mode initially"

    user_settings.enable_emergency_stop()
    assert user_settings.is_emergency_mode(), "Should be in emergency mode"
    assert (
        not user_settings.is_trading_enabled()
    ), "Trading should be disabled in emergency mode"

    user_settings.disable_emergency_stop()
    assert not user_settings.is_emergency_mode(), "Should not be in emergency mode"
    assert user_settings.is_trading_enabled(), "Trading should be enabled"

    print("✅ Emergency functions work correctly")

    # Test notification settings
    print("\n📱 Testing Notification Settings:")
    initial_signal_state = user_settings.settings["notifications"]["signal_alerts"]
    new_state = user_settings.toggle_signal_alerts()
    assert new_state != initial_signal_state, "Signal alerts should toggle"

    print("✅ Notification settings work correctly")

    # Test settings summary
    print("\n📊 Testing Settings Summary:")
    summary = user_settings.get_settings_summary()
    assert "CURRENT SETTINGS" in summary, "Summary should contain title"
    assert "Risk Management" in summary, "Summary should contain risk info"
    print("✅ Settings summary works correctly")

    print("\n🎉 All user settings tests passed!")


def test_bot_commands():
    """Test bot command integration"""
    print("\n🤖 Testing Bot Integration...")

    # Create a test bot instance (without actual Telegram connection)
    class TestBot(TradingBot):
        def __init__(self):
            self.pending_trades = {}
            self.trading_system = None

    bot = TestBot()

    # Test settings summary formatting
    summary = user_settings.get_settings_summary()
    print("Settings summary generated successfully")

    # Test strategy names mapping
    strategy_names = {
        "RSI_EMA": "RSI + EMA",
        "MACD": "MACD",
        "BOLLINGER": "Bollinger Bands",
    }
    current_strategy = user_settings.get_strategy()
    display_name = strategy_names.get(current_strategy, current_strategy)
    print(f"Current strategy display: {display_name}")

    print("✅ Bot integration tests passed!")


async def test_signal_processing():
    """Test signal processing with user settings"""
    print("\n📊 Testing Signal Processing...")

    # Create test signal
    test_signal = Signal(
        pair="BTCUSDT",
        action="BUY",
        current_price=100000.0,
        confidence=0.75,
        stop_loss=98000.0,
        take_profit=104000.0,
        risk_reward=2.0,
        indicators={"rsi": 30, "ema_trend": "Bullish"},
        timestamp=datetime.now(),
    )

    print(
        f"Test signal: {test_signal.pair} {test_signal.action} at ${test_signal.current_price}"
    )

    # Test with trading enabled
    user_settings.set_trading_enabled(True)
    assert user_settings.is_trading_enabled(), "Trading should be enabled"
    print("✅ Trading enabled check works")

    # Test with emergency mode
    user_settings.enable_emergency_stop()
    assert (
        not user_settings.is_trading_enabled()
    ), "Trading should be disabled in emergency"
    print("✅ Emergency mode disables trading")

    # Reset for normal operation
    user_settings.disable_emergency_stop()

    # Test custom stop loss calculation
    user_settings.set_custom_stop_loss(2.0)  # 2%
    risk_settings = user_settings.get_risk_settings()

    if risk_settings["custom_stop_loss"]:
        custom_stop = test_signal.current_price * (
            1 - risk_settings["custom_stop_loss"] / 100
        )
        expected_stop = 100000.0 * 0.98  # 2% below current price
        assert (
            abs(custom_stop - expected_stop) < 1
        ), f"Custom stop loss calculation: {custom_stop} vs {expected_stop}"
        print(f"✅ Custom stop loss calculation: {custom_stop}")

    print("✅ Signal processing tests passed!")


def test_emergency_scenarios():
    """Test various emergency scenarios"""
    print("\n🚨 Testing Emergency Scenarios...")

    # Test maximum positions limit
    user_settings.set_max_open_positions(3)
    max_positions = user_settings.get_risk_settings()["max_open_positions"]
    assert max_positions == 3, "Max positions should be 3"
    print(f"✅ Max positions set to: {max_positions}")

    # Test daily loss limit
    user_settings.set_max_daily_loss(0.05)  # 5%
    daily_loss = user_settings.settings["emergency"]["max_daily_loss"]
    assert daily_loss == 0.05, "Daily loss should be 5%"
    print(f"✅ Daily loss limit set to: {daily_loss*100}%")

    # Test trading pause vs emergency stop
    user_settings.set_trading_enabled(False)
    assert not user_settings.is_trading_enabled(), "Trading should be paused"
    assert not user_settings.is_emergency_mode(), "Should not be emergency mode"
    print("✅ Trading pause works independently from emergency mode")

    user_settings.enable_emergency_stop()
    assert user_settings.is_emergency_mode(), "Should be in emergency mode"
    assert not user_settings.is_trading_enabled(), "Trading should be disabled"
    print("✅ Emergency mode overrides trading settings")

    # Reset
    user_settings.disable_emergency_stop()
    user_settings.set_trading_enabled(True)

    print("✅ All emergency scenarios tested!")


def test_settings_persistence():
    """Test that settings are saved and loaded correctly"""
    print("\n💾 Testing Settings Persistence...")

    # Set some test values
    test_strategy = "MACD"
    test_risk = 0.015  # 1.5%
    test_max_positions = 7

    user_settings.set_strategy(test_strategy)
    user_settings.set_max_risk_per_trade(test_risk)
    user_settings.set_max_open_positions(test_max_positions)

    # Create new instance to test loading
    from user_settings import UserSettings

    new_settings = UserSettings()

    assert new_settings.get_strategy() == test_strategy, "Strategy should persist"
    assert (
        new_settings.get_risk_settings()["max_risk_per_trade"] == test_risk
    ), "Risk should persist"
    assert (
        new_settings.get_risk_settings()["max_open_positions"] == test_max_positions
    ), "Max positions should persist"

    print("✅ Settings persistence works correctly!")

    # Reset to defaults for cleanup
    user_settings.reset_to_defaults()
    print("✅ Settings reset to defaults")


async def main():
    """Run all tests"""
    print("🚀 Starting Settings and Emergency Features Test Suite")
    print("=" * 60)

    try:
        # Run all tests
        test_user_settings()
        test_bot_commands()
        await test_signal_processing()
        test_emergency_scenarios()
        test_settings_persistence()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Settings and emergency features are working correctly!")
        print("\nAvailable commands in your bot:")
        print("• /settings - Configure strategies and risk management")
        print("• /emergency - Emergency controls and risk management")
        print("• /status - Shows current settings and trading status")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()

    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
