#!/usr/bin/env python3
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
        print("\n1. Testing bot.py directly...")
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
        print("\n2. Testing integrated signal service...")
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
        
        print("\n🎉 All tests completed! Check your Telegram for messages.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_production_telegram())
