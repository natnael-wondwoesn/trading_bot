#!/usr/bin/env python3
"""
Simple test to debug bot settings functionality
"""

import asyncio
import logging
from datetime import datetime

# Test imports
try:
    from user_settings import user_settings

    print("✅ user_settings imported successfully")
except Exception as e:
    print(f"❌ Error importing user_settings: {e}")

try:
    from bot import TradingBot

    print("✅ TradingBot imported successfully")
except Exception as e:
    print(f"❌ Error importing TradingBot: {e}")

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    print("✅ Telegram imports successful")
except Exception as e:
    print(f"❌ Error importing Telegram: {e}")


def test_settings_functionality():
    """Test the core settings functionality"""
    print("\n🧪 Testing Settings Core Functionality:")

    # Test basic settings
    print(f"Current strategy: {user_settings.get_strategy()}")
    print(f"Trading enabled: {user_settings.is_trading_enabled()}")
    print(f"Emergency mode: {user_settings.is_emergency_mode()}")

    # Test settings summary
    try:
        summary = user_settings.get_settings_summary()
        print("✅ Settings summary generated successfully")
    except Exception as e:
        print(f"❌ Error generating settings summary: {e}")
        return False

    # Test strategy setting
    try:
        user_settings.set_strategy("MACD")
        assert user_settings.get_strategy() == "MACD"
        user_settings.set_strategy("RSI_EMA")  # Reset
        print("✅ Strategy setting works")
    except Exception as e:
        print(f"❌ Error setting strategy: {e}")
        return False

    return True


async def test_settings_command():
    """Test the settings command specifically"""
    print("\n🤖 Testing Settings Command:")

    try:
        # Create mock update and context
        class MockMessage:
            async def reply_text(self, text, parse_mode=None, reply_markup=None):
                print(f"Bot would send: {text[:100]}...")
                if reply_markup:
                    print("✅ Inline keyboard created successfully")
                return True

        class MockUpdate:
            def __init__(self):
                self.message = MockMessage()

        class MockContext:
            pass

        # Test the settings command
        from config.config import Config

        bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

        mock_update = MockUpdate()
        mock_context = MockContext()

        # Call the settings command
        await bot.settings_command(mock_update, mock_context)
        print("✅ Settings command executed successfully")

    except Exception as e:
        print(f"❌ Error in settings command: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("🚀 Testing Bot Settings Functionality")
    print("=" * 50)

    # Test core functionality
    if not test_settings_functionality():
        print("❌ Core settings functionality failed")
        exit(1)

    # Test settings command
    asyncio.run(test_settings_command())

    print("\n" + "=" * 50)
    print("🎉 All tests completed! Bot should work correctly.")
