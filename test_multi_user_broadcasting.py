#!/usr/bin/env python3
"""
Test Multi-User Signal Broadcasting
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

async def test_multi_user_broadcasting():
    """Test broadcasting signals to all users"""
    print("🧪 TESTING MULTI-USER SIGNAL BROADCASTING")
    print("=" * 50)
    
    try:
        # Import the production service
        import sys
        sys.path.append('.')
        
        from services.integrated_signal_service import integrated_signal_service
        
        # Create test signals
        test_signals = [
            {
                'pair': 'BTCUSDT',
                'action': 'BUY',
                'price': 116800.0,
                'confidence': 0.78,  # 78%
                'stop_loss': 115500.0,
                'take_profit': 118200.0,
                'timestamp': datetime.now()
            },
            {
                'pair': 'ETHUSDT',
                'action': 'SELL',
                'price': 2965.0,
                'confidence': 0.72,  # 72%
                'stop_loss': 3020.0,
                'take_profit': 2850.0,
                'timestamp': datetime.now()
            }
        ]
        
        print(f"📤 Broadcasting {len(test_signals)} test signals to all active users...")
        
        # Trigger the callback manually
        for callback in integrated_signal_service.signal_callbacks:
            try:
                await callback(test_signals)
                print("✅ Multi-user broadcast callback executed successfully")
            except Exception as e:
                print(f"❌ Callback failed: {e}")
        
        print("\n🎯 Check all user Telegram chats for the test signals!")
        print("\n📊 Expected behavior:")
        print("   • All users who have started your bot should receive signals")
        print("   • Each signal will be formatted with emojis and details")
        print("   • Users will see BTC BUY and ETH SELL signals")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def show_active_users():
    """Show how many users are currently active"""
    try:
        # Try to access the multi-user bot
        from production_main import service_instance
        
        if hasattr(service_instance, 'services') and 'bot' in service_instance.services:
            bot = service_instance.services['bot']
            active_count = len(bot.active_users)
            print(f"\n👥 Currently active users: {active_count}")
            
            if active_count > 0:
                print("\n📱 Active user IDs:")
                for telegram_id in bot.active_users.keys():
                    print(f"   • User ID: {telegram_id}")
            else:
                print("\n⚠️ No active users found")
                print("   Users need to send /start to your bot first")
                
        else:
            print("\n⚠️ Production service not accessible for user count")
            
    except Exception as e:
        print(f"\n⚠️ Could not check active users: {e}")


if __name__ == "__main__":
    asyncio.run(test_multi_user_broadcasting())
    asyncio.run(show_active_users())
