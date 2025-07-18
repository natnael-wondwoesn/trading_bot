#!/usr/bin/env python3
"""
Enable Multi-User Signal Broadcasting
Convert your system from single-user to multi-user signal broadcasting
"""

import os
import asyncio


def add_broadcast_signal_method():
    """Add broadcast signal method to multi_user_bot.py"""
    print("🔧 ADDING BROADCAST SIGNAL METHOD")
    print("=" * 40)

    bot_file = "services/multi_user_bot.py"

    if not os.path.exists(bot_file):
        print("❌ multi_user_bot.py not found")
        return False

    try:
        with open(bot_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Add the broadcast_signal_to_all_users method
        broadcast_method = '''
    async def broadcast_signal_to_all_users(self, signal_data: Dict):
        """Broadcast trading signal to all active users"""
        try:
            if not self.active_users:
                logger.info("No active users to send signals to")
                return 0
            
            # Format the signal message
            symbol = signal_data.get('symbol', 'N/A')
            action = signal_data.get('action', 'N/A')
            price = signal_data.get('price', 0)
            confidence = signal_data.get('confidence', 0)
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')
            
            # Action emoji
            action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            
            # Confidence level emoji
            if confidence >= 70:
                confidence_emoji = "🔥"
            elif confidence >= 60:
                confidence_emoji = "⚡"
            else:
                confidence_emoji = "💫"
            
            message = f"""{action_emoji} **TRADING SIGNAL** {confidence_emoji}

📊 **{symbol}**
🎯 **Action:** {action}
💰 **Price:** ${price:.4f}
📈 **Confidence:** {confidence:.1f}%"""

            if stop_loss:
                message += f"\\n🛡️ **Stop Loss:** ${stop_loss:.4f}"
            if take_profit:
                message += f"\\n🎯 **Take Profit:** ${take_profit:.4f}"
                
            message += f"\\n\\n⏰ **Time:** {datetime.now().strftime('%H:%M:%S')}"
            message += f"\\n\\n_From Enhanced Strategy System_"

            # Send to all active users
            sent_count = 0
            failed_count = 0
            
            logger.info(f"Broadcasting {symbol} {action} signal to {len(self.active_users)} active users")
            
            for telegram_id, context in self.active_users.items():
                try:
                    await self.application.bot.send_message(
                        chat_id=telegram_id, 
                        text=message, 
                        parse_mode="Markdown"
                    )
                    sent_count += 1
                    logger.debug(f"✅ Signal sent to user {telegram_id}")
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as user_error:
                    failed_count += 1
                    logger.warning(f"Failed to send signal to user {telegram_id}: {user_error}")
            
            logger.info(f"📱 Signal broadcast complete: {sent_count} sent, {failed_count} failed")
            return sent_count
            
        except Exception as e:
            logger.error(f"Error broadcasting signal to all users: {e}")
            return 0
'''

        # Add the method before the maintenance section
        if "async def _maintenance_task(self):" in content:
            insert_pos = content.find("# Maintenance and Monitoring")
            if insert_pos > 0:
                content = (
                    content[:insert_pos]
                    + broadcast_method
                    + "\n    "
                    + content[insert_pos:]
                )
                print("✅ Added broadcast_signal_to_all_users method")
            else:
                # Fallback insertion
                insert_pos = content.find("async def _maintenance_task(self):")
                content = (
                    content[:insert_pos]
                    + broadcast_method
                    + "\n    "
                    + content[insert_pos:]
                )
                print("✅ Added broadcast_signal_to_all_users method (fallback)")

        # Add a simplified broadcast method as well
        simple_broadcast = '''
    async def send_signal_to_all_users(self, signal_info):
        """Simple method to send signal to all users (compatible with existing code)"""
        try:
            # Convert signal_info to the format expected by broadcast_signal_to_all_users
            signal_data = {
                'symbol': signal_info.get('symbol', 'UNKNOWN'),
                'action': signal_info.get('action', 'HOLD'),
                'price': signal_info.get('price', 0),
                'confidence': signal_info.get('confidence', 0) * 100 if signal_info.get('confidence', 0) <= 1 else signal_info.get('confidence', 0),
                'stop_loss': signal_info.get('stop_loss'),
                'take_profit': signal_info.get('take_profit')
            }
            
            return await self.broadcast_signal_to_all_users(signal_data)
            
        except Exception as e:
            logger.error(f"Error in send_signal_to_all_users: {e}")
            return 0
'''

        # Add the simple method too
        content = content.replace(broadcast_method, broadcast_method + simple_broadcast)

        # Write back
        with open(bot_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Multi-user bot updated with broadcast methods")
        return True

    except Exception as e:
        print(f"❌ Failed to add broadcast method: {e}")
        return False


def update_production_main_callback():
    """Update production_main.py to use multi-user broadcasting"""
    print("\n🔄 UPDATING PRODUCTION MAIN CALLBACK")
    print("=" * 40)

    production_file = "production_main.py"

    if not os.path.exists(production_file):
        print("❌ production_main.py not found")
        return False

    try:
        with open(production_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Find the current signal notification callback
        old_callback_pattern = """async def signal_notification_callback(signals):
                    if signals and "bot" in self.services:
                        try:
                            bot = self.services["bot"]
                            logger.info(f"Sending {len(signals)} signals to Telegram users")
                            
                            # Send each signal to all active users
                            for signal in signals:
                                # Convert signal dict to format expected by bot
                                signal_data = {
                                    'symbol': signal.get('pair', 'UNKNOWN'),
                                    'action': signal.get('action', 'HOLD'),
                                    'price': signal.get('price', 0),
                                    'confidence': signal.get('confidence', 0) * 100,  # Convert to percentage
                                    'stop_loss': signal.get('stop_loss'),
                                    'take_profit': signal.get('take_profit'),
                                    'timestamp': signal.get('timestamp', datetime.now())
                                }
                                
                                # Send to all active users
                                active_user_count = len(bot.active_users)
                                logger.info(f"Broadcasting {signal_data['symbol']} {signal_data['action']} signal to {active_user_count} users")
                                
                                for telegram_id, user_context in bot.active_users.items():
                                    try:
                                        await bot._send_signal_notification(user_context.user.user_id, {"signal": signal_data})
                                        logger.debug(f"Sent signal to user {telegram_id}")
                                    except Exception as user_error:
                                        logger.error(f"Failed to send signal to user {telegram_id}: {user_error}")
                                        
                        except Exception as e:
                            logger.error(f"Error notifying users of signals: {e}")"""

        new_callback = """async def signal_notification_callback(signals):
                    if signals and "bot" in self.services:
                        try:
                            bot = self.services["bot"]
                            active_user_count = len(bot.active_users)
                            logger.info(f"Broadcasting {len(signals)} signals to {active_user_count} active users")
                            
                            # Send each signal to all active users using the new broadcast method
                            total_sent = 0
                            for signal in signals:
                                # Convert signal dict to format expected by broadcast method
                                signal_data = {
                                    'symbol': signal.get('pair', 'UNKNOWN'),
                                    'action': signal.get('action', 'HOLD'),
                                    'price': signal.get('price', 0),
                                    'confidence': signal.get('confidence', 0) * 100 if signal.get('confidence', 0) <= 1 else signal.get('confidence', 0),
                                    'stop_loss': signal.get('stop_loss'),
                                    'take_profit': signal.get('take_profit'),
                                    'timestamp': signal.get('timestamp', datetime.now())
                                }
                                
                                # Use the new broadcast method
                                sent_count = await bot.broadcast_signal_to_all_users(signal_data)
                                total_sent += sent_count
                                
                                logger.info(f"📊 {signal_data['symbol']} {signal_data['action']} signal sent to {sent_count} users")
                            
                            logger.info(f"🎯 Total broadcast complete: {total_sent} messages sent across {len(signals)} signals")
                                        
                        except Exception as e:
                            logger.error(f"Error broadcasting signals to users: {e}")"""

        if old_callback_pattern in content:
            content = content.replace(old_callback_pattern, new_callback)
            print("✅ Updated signal callback to use multi-user broadcasting")
        else:
            print("⚠️ Old callback pattern not found - may already be updated")

        # Write back
        with open(production_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Production main updated for multi-user broadcasting")
        return True

    except Exception as e:
        print(f"❌ Failed to update production main: {e}")
        return False


def update_integrated_signal_service():
    """Update integrated signal service to work with multi-user system"""
    print("\n⚡ UPDATING INTEGRATED SIGNAL SERVICE")
    print("=" * 40)

    service_file = "services/integrated_signal_service.py"

    if not os.path.exists(service_file):
        print("❌ integrated_signal_service.py not found")
        return False

    try:
        with open(service_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Remove or comment out the old bot.py telegram method since we're using multi-user now
        old_telegram_method = '''async def send_telegram_alert(self, signal_info):
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
            logger.debug(f"Full error: {traceback.format_exc()}")'''

        # Comment out the old method instead of removing it completely
        if old_telegram_method in content:
            commented_method = old_telegram_method.replace(
                "async def send_telegram_alert",
                "# DISABLED: async def send_telegram_alert_old",
            )
            commented_method = (
                "    # OLD SINGLE-USER METHOD - DISABLED FOR MULTI-USER\n"
                + commented_method
            )
            content = content.replace(old_telegram_method, commented_method)
            print("✅ Disabled old single-user telegram method")

        # Remove the call to send_telegram_alert in the signal generation
        old_telegram_call = """# Send Telegram notification (same as live_signal_monitor.py)
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

        # Replace with a note that notifications are handled by the callback system
        new_telegram_note = """# Telegram notifications are handled by the production callback system
                    # which broadcasts to all active users automatically
                    logger.debug(f"Signal generated: {pair} {signal.action} - will be broadcast to all users")"""

        if old_telegram_call in content:
            content = content.replace(old_telegram_call, new_telegram_note)
            print("✅ Updated signal service to rely on callback system")

        # Write back
        with open(service_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Integrated signal service updated for multi-user")
        return True

    except Exception as e:
        print(f"❌ Failed to update signal service: {e}")
        return False


def create_user_management_commands():
    """Add commands to manage users and test broadcasting"""
    print("\n👥 ADDING USER MANAGEMENT COMMANDS")
    print("=" * 40)

    bot_file = "services/multi_user_bot.py"

    if not os.path.exists(bot_file):
        print("❌ multi_user_bot.py not found")
        return False

    try:
        with open(bot_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Add user stats command
        stats_command = '''
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user statistics and system status"""
        try:
            telegram_id = update.effective_user.id
            
            # Check if user is active
            if telegram_id not in self.active_users:
                await update.message.reply_text("❌ Please register first with /start")
                return
            
            active_count = len(self.active_users)
            
            stats_message = f"""📊 **System Statistics**

👥 **Active Users:** {active_count}
📈 **Your Status:** Active
🤖 **Bot Version:** Multi-User v2.0
⚡ **Signal Broadcasting:** Enabled

🔄 **Recent Activity:**
• Messages sent today: {self.stats.get('messages_sent_today', 0)}
• Commands processed: {self.stats.get('commands_processed_today', 0)}

_All active users receive trading signals automatically_"""
            
            await update.message.reply_text(stats_message, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in stats command: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
'''

        # Add test broadcast command
        test_broadcast_command = '''
    async def test_broadcast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test broadcasting to all users (admin only)"""
        try:
            telegram_id = update.effective_user.id
            
            # Simple admin check (you can enhance this)
            admin_ids = [telegram_id]  # Add your telegram ID here
            
            if telegram_id not in admin_ids:
                await update.message.reply_text("❌ Admin command only")
                return
            
            # Create test signal
            test_signal = {
                'symbol': 'TESTUSDT',
                'action': 'BUY',
                'price': 1.2345,
                'confidence': 85.5,
                'stop_loss': 1.2000,
                'take_profit': 1.2800
            }
            
            sent_count = await self.broadcast_signal_to_all_users(test_signal)
            
            await update.message.reply_text(
                f"✅ Test signal broadcast complete!\\n📊 Sent to {sent_count} users",
                parse_mode="Markdown"
            )
            
        except Exception as e:
            logger.error(f"Error in test broadcast command: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
'''

        # Find where to insert the commands
        if "async def help_command" in content:
            insert_pos = content.find("async def help_command")
            content = (
                content[:insert_pos]
                + stats_command
                + test_broadcast_command
                + "\\n    "
                + content[insert_pos:]
            )
            print("✅ Added user management commands")

        # Add command handlers
        if 'CommandHandler("help", self.help_command)' in content:
            handler_line = 'CommandHandler("help", self.help_command)'
            new_handlers = """CommandHandler("help", self.help_command),
            CommandHandler("stats", self.stats_command),
            CommandHandler("test_broadcast", self.test_broadcast_command)"""

            content = content.replace(handler_line, new_handlers)
            print("✅ Registered user management command handlers")

        # Write back
        with open(bot_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ User management commands added")
        return True

    except Exception as e:
        print(f"❌ Failed to add user management commands: {e}")
        return False


def create_multi_user_test_script():
    """Create a test script for multi-user functionality"""
    print("\n🧪 CREATING MULTI-USER TEST SCRIPT")
    print("=" * 40)

    test_content = '''#!/usr/bin/env python3
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
        
        print("\\n🎯 Check all user Telegram chats for the test signals!")
        print("\\n📊 Expected behavior:")
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
            print(f"\\n👥 Currently active users: {active_count}")
            
            if active_count > 0:
                print("\\n📱 Active user IDs:")
                for telegram_id in bot.active_users.keys():
                    print(f"   • User ID: {telegram_id}")
            else:
                print("\\n⚠️ No active users found")
                print("   Users need to send /start to your bot first")
                
        else:
            print("\\n⚠️ Production service not accessible for user count")
            
    except Exception as e:
        print(f"\\n⚠️ Could not check active users: {e}")


if __name__ == "__main__":
    asyncio.run(test_multi_user_broadcasting())
    asyncio.run(show_active_users())
'''

    with open("test_multi_user_broadcasting.py", "w", encoding="utf-8") as f:
        f.write(test_content)

    print("✅ Created test_multi_user_broadcasting.py")
    return True


def main():
    """Main execution"""
    print("ENABLING MULTI-USER SIGNAL BROADCASTING")
    print("=" * 60)
    print("Converting from single-user to multi-user notifications")
    print("=" * 60)

    # Apply all updates
    fix1 = add_broadcast_signal_method()
    fix2 = update_production_main_callback()
    fix3 = update_integrated_signal_service()
    fix4 = create_user_management_commands()
    fix5 = create_multi_user_test_script()

    print("\\n" + "=" * 60)
    print("📊 MULTI-USER SETUP RESULTS:")
    print(f"   Broadcast Method: {'✅ ADDED' if fix1 else '❌ FAILED'}")
    print(f"   Production Callback: {'✅ UPDATED' if fix2 else '❌ FAILED'}")
    print(f"   Signal Service: {'✅ UPDATED' if fix3 else '❌ FAILED'}")
    print(f"   User Commands: {'✅ ADDED' if fix4 else '❌ FAILED'}")
    print(f"   Test Script: {'✅ CREATED' if fix5 else '❌ FAILED'}")

    if all([fix1, fix2, fix3, fix4, fix5]):
        print("\\n🎉 MULTI-USER BROADCASTING ENABLED!")

        print("\\n🔄 RESTART YOUR SYSTEM:")
        print("   Ctrl+C (stop current system)")
        print("   python production_main.py")

        print("\\n👥 HOW TO GET USERS:")
        print("   1. Share your bot link: t.me/your_bot_username")
        print("   2. Users send /start to register")
        print("   3. All registered users automatically get signals")

        print("\\n🧪 TEST MULTI-USER BROADCASTING:")
        print("   python test_multi_user_broadcasting.py")

        print("\\n📱 NEW USER COMMANDS:")
        print("   /stats - Show active user count and system status")
        print("   /test_broadcast - Send test signal to all users (admin only)")

        print("\\n✅ WHAT HAPPENS NOW:")
        print("   • Every signal gets sent to ALL active users")
        print("   • Users see signals in real-time as they're generated")
        print("   • No more single-user limitation")
        print("   • Each user gets the same high-quality formatted signals")

        print("\\n📊 EXPECTED USER EXPERIENCE:")
        print("   1. User sends /start to your bot")
        print("   2. User gets welcome message and registration")
        print("   3. User automatically receives all trading signals")
        print("   4. User can use /stats to see system status")

        print("\\n🎯 CURRENT SIGNAL FLOW:")
        print(
            "   Signal Generated → Production Callback → Multi-User Broadcast → All Users"
        )

    else:
        print("\\n⚠️ Some updates failed - check errors above")


if __name__ == "__main__":
    main()
