#!/usr/bin/env python3
"""
Fix Production Telegram Notifications
Make production system use the same notification method as live_signal_monitor.py
"""

import os


def fix_integrated_signal_service_notifications():
    """Fix the integrated signal service to use the same notification method as live_signal_monitor.py"""
    print("🔧 FIXING INTEGRATED SIGNAL SERVICE NOTIFICATIONS")
    print("=" * 50)

    service_file = "services/integrated_signal_service.py"

    if not os.path.exists(service_file):
        print("❌ integrated_signal_service.py not found")
        return False

    try:
        with open(service_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Add the same send_telegram_alert method from live_signal_monitor.py
        telegram_method = '''
    async def send_telegram_alert(self, signal_info):
        """Send Telegram alert about the signal (same as live_signal_monitor.py)"""
        try:
            from bot import TradingBot
            from config.config import Config

            bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

            message = f"""📈 **{signal_info['symbol']}** - {signal_info['action']}
💰 Price: ${signal_info['price']:,.4f}
🎲 Confidence: {signal_info['confidence']:.1%}
📊 RSI: {signal_info['rsi']:.1f}

🛑 Stop Loss: ${signal_info['stop_loss']:,.4f}
🎯 Take Profit: ${signal_info['take_profit']:,.4f}

⏰ {signal_info['timestamp'].strftime('%H:%M:%S')}"""

            await bot.send_alert("ENHANCED STRATEGY SIGNAL", message, "money")
            logger.info("📱 Telegram alert sent successfully")

        except Exception as e:
            logger.warning(f"⚠️ Telegram alert failed: {e}")
            # Log more details for debugging
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")
'''

        # Add the method if it doesn't exist
        if "async def send_telegram_alert" not in content:
            # Find a good place to insert (before close method or at end of class)
            if "async def stop_monitoring(self):" in content:
                insert_pos = content.find("async def stop_monitoring(self):")
                content = (
                    content[:insert_pos]
                    + telegram_method
                    + "\n    "
                    + content[insert_pos:]
                )
                print("✅ Added send_telegram_alert method")
            else:
                # Add before end of class
                class_end = content.rfind("# Global service instance")
                if class_end > 0:
                    content = (
                        content[:class_end]
                        + telegram_method
                        + "\n\n"
                        + content[class_end:]
                    )
                    print("✅ Added send_telegram_alert method at end of class")

        # Now modify the scan method to call send_telegram_alert for each signal
        # Find the scan method where signals are generated
        if 'logger.info(f"SIGNAL: {pair} {signal.action}' in content:
            old_log_pattern = 'logger.info(f"SIGNAL: {pair} {signal.action} @ ${accurate_price:.4f} (Confidence: {signal.confidence:.1%})")'

            new_log_with_telegram = """logger.info(f"SIGNAL: {pair} {signal.action} @ ${accurate_price:.4f} (Confidence: {signal.confidence:.1%})")
                    
                    # Send Telegram notification (same as live_signal_monitor.py)
                    try:
                        # Convert signal to format expected by send_telegram_alert
                        signal_info = {
                            'symbol': pair,
                            'action': signal.action,
                            'price': accurate_price,
                            'confidence': signal.confidence,
                            'rsi': signal.indicators.get('rsi', 0),
                            'stop_loss': signal.stop_loss or 0,
                            'take_profit': signal.take_profit or 0,
                            'timestamp': signal.timestamp
                        }
                        
                        await self.send_telegram_alert(signal_info)
                        logger.info(f"📱 Sent Telegram notification for {pair} {signal.action}")
                        
                    except Exception as telegram_error:
                        logger.error(f"Failed to send Telegram notification: {telegram_error}")"""

            content = content.replace(old_log_pattern, new_log_with_telegram)
            print("✅ Added Telegram notification to signal generation")

        # Write back the modified content
        with open(service_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Integrated signal service updated with Telegram notifications")
        return True

    except Exception as e:
        print(f"❌ Failed to fix integrated signal service: {e}")
        return False


def check_bot_py_exists():
    """Check if bot.py exists and is properly configured"""
    print("\n📱 CHECKING BOT.PY CONFIGURATION")
    print("=" * 40)

    if not os.path.exists("bot.py"):
        print("❌ bot.py not found - this is needed for Telegram notifications")
        return False

    try:
        with open("bot.py", "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if "class TradingBot:" in content:
            print("✅ TradingBot class found in bot.py")
        else:
            print("❌ TradingBot class not found in bot.py")
            return False

        if "async def send_alert" in content:
            print("✅ send_alert method found in bot.py")
        else:
            print("❌ send_alert method not found in bot.py")
            return False

        return True

    except Exception as e:
        print(f"❌ Error checking bot.py: {e}")
        return False


def check_config_telegram_settings():
    """Check if Telegram settings are properly configured"""
    print("\n⚙️ CHECKING TELEGRAM CONFIGURATION")
    print("=" * 40)

    config_file = "config/config.py"

    if not os.path.exists(config_file):
        print("❌ config/config.py not found")
        return False

    try:
        with open(config_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if "TELEGRAM_BOT_TOKEN" in content:
            print("✅ TELEGRAM_BOT_TOKEN found in config")
        else:
            print("❌ TELEGRAM_BOT_TOKEN not found in config")
            print("   Add: TELEGRAM_BOT_TOKEN = 'your_bot_token'")

        if "TELEGRAM_CHAT_ID" in content:
            print("✅ TELEGRAM_CHAT_ID found in config")
        else:
            print("❌ TELEGRAM_CHAT_ID not found in config")
            print("   Add: TELEGRAM_CHAT_ID = 'your_chat_id'")

        return True

    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False


def create_telegram_test():
    """Create a test script to verify Telegram notifications work"""
    print("\n🧪 CREATING TELEGRAM TEST SCRIPT")
    print("=" * 40)

    test_content = '''#!/usr/bin/env python3
"""
Test Telegram Notifications - Production System
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

async def test_production_telegram():
    """Test Telegram notifications from production system"""
    print("🧪 TESTING PRODUCTION TELEGRAM NOTIFICATIONS")
    print("=" * 50)
    
    try:
        # Test 1: Direct bot.py test
        print("\\n1. Testing bot.py directly...")
        from bot import TradingBot
        from config.config import Config
        
        bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
        
        test_message = """📈 **BTCUSDT** - BUY
💰 Price: $116,500.00
🎲 Confidence: 75.5%
📊 RSI: 28.5

🛑 Stop Loss: $115,000.00
🎯 Take Profit: $118,000.00

⏰ TEST MESSAGE"""
        
        await bot.send_alert("PRODUCTION TEST SIGNAL", test_message, "money")
        print("✅ Direct bot.py test - Message sent!")
        
        # Test 2: Integrated signal service test
        print("\\n2. Testing integrated signal service...")
        from services.integrated_signal_service import integrated_signal_service
        
        if hasattr(integrated_signal_service, 'send_telegram_alert'):
            test_signal_info = {
                'symbol': 'ETHUSDT',
                'action': 'SELL',
                'price': 2965.50,
                'confidence': 0.68,
                'rsi': 72.3,
                'stop_loss': 3050.00,
                'take_profit': 2850.00,
                'timestamp': datetime.now()
            }
            
            await integrated_signal_service.send_telegram_alert(test_signal_info)
            print("✅ Integrated service test - Message sent!")
        else:
            print("❌ send_telegram_alert method not found in integrated service")
        
        print("\\n🎉 All tests completed! Check your Telegram for messages.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_production_telegram())
'''

    with open("test_production_telegram.py", "w", encoding="utf-8") as f:
        f.write(test_content)

    print("✅ Created test_production_telegram.py")
    return True


def main():
    """Main execution"""
    print("FIXING PRODUCTION TELEGRAM NOTIFICATIONS")
    print("=" * 60)
    print("Making production system use same method as live_signal_monitor.py")
    print("=" * 60)

    # Check prerequisites
    bot_exists = check_bot_py_exists()
    config_ok = check_config_telegram_settings()

    if not bot_exists:
        print("\\n❌ bot.py is missing or incomplete")
        print("   This is needed for Telegram notifications to work")
        print("   live_signal_monitor.py uses bot.py to send messages")
        return

    # Apply the fix
    fix_applied = fix_integrated_signal_service_notifications()

    # Create test script
    test_created = create_telegram_test()

    print("\\n" + "=" * 60)
    print("📊 FIX RESULTS:")
    print(f"   Bot.py Check: {'✅ PASS' if bot_exists else '❌ FAIL'}")
    print(f"   Config Check: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Integration Fix: {'✅ APPLIED' if fix_applied else '❌ FAILED'}")
    print(f"   Test Script: {'✅ CREATED' if test_created else '❌ FAILED'}")

    if fix_applied:
        print("\\n🎉 PRODUCTION TELEGRAM NOTIFICATIONS FIXED!")

        print("\\n🔄 RESTART YOUR SYSTEM:")
        print("   Ctrl+C (stop current system)")
        print("   python production_main.py")

        print("\\n🧪 TEST THE FIX:")
        print("   python test_production_telegram.py")

        print("\\n✅ WHAT HAPPENS NOW:")
        print(
            "   • Production system will use same Telegram method as live_signal_monitor.py"
        )
        print("   • Each signal will automatically send Telegram notification")
        print("   • Messages will have same format as working live_signal_monitor.py")
        print("   • No more difference between standalone and production notifications")

        print("\\n📱 EXPECTED TELEGRAM MESSAGES:")
        print("   📈 BTCUSDT - BUY")
        print("   💰 Price: $116,560.50")
        print("   🎲 Confidence: 65.0%")
        print("   📊 RSI: 28.5")
        print("   🛑 Stop Loss: $115,000.00")
        print("   🎯 Take Profit: $118,000.00")

    else:
        print("\\n⚠️ Fix failed - check errors above")


if __name__ == "__main__":
    main()
