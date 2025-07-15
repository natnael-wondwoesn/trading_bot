#!/usr/bin/env python3
"""
Debug version of the main trading bot with comprehensive error handling
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from config.config import Config
from bot import TradingBot
from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
from strategy.strategies.macd_strategy import MACDStrategy
from strategy.strategies.bollinger_strategy import BollingerStrategy
from models.models import PerformanceStats, Signal, TradeSetup
from mexc.mexc_client import MEXCClient, MEXCTradeExecutor
from mexc.data_feed import MEXCDataFeed
from user_settings import user_settings

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Reduce telegram library logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)


class DebugTradingSystem:
    """Debug version of the trading system with enhanced error handling"""

    def __init__(self):
        logger.info("🚀 Initializing debug trading system...")

        # Validate configuration first
        try:
            Config.validate()
            logger.info("✅ Configuration validation passed")
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise

        # Initialize MEXC client with error handling
        try:
            self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
            self.trade_executor = MEXCTradeExecutor(self.mexc_client)
            self.data_feed = MEXCDataFeed(self.mexc_client)
            logger.info("✅ MEXC clients initialized")
        except Exception as e:
            logger.error(f"❌ MEXC initialization failed: {e}")
            raise

        # Initialize bot with error handling
        try:
            self.bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
            logger.info("✅ TradingBot instance created")
        except Exception as e:
            logger.error(f"❌ TradingBot initialization failed: {e}")
            raise

        # Create Telegram application
        try:
            from telegram.ext import Application

            self.bot.application = (
                Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            )
            logger.info("✅ Telegram application created")
        except Exception as e:
            logger.error(f"❌ Telegram application creation failed: {e}")
            raise

        # Initialize strategy
        try:
            self.strategy = self.get_current_strategy()
            logger.info(f"✅ Strategy initialized: {user_settings.get_strategy()}")
        except Exception as e:
            logger.error(f"❌ Strategy initialization failed: {e}")
            raise

        self.running = False
        self.active_trades = {}
        self.performance_tracker = PerformanceTracker()

        # Set up trade callback for bot
        self.bot.trade_callback = self.execute_trade
        self.bot.trading_system = self

        logger.info("✅ Debug trading system initialization complete")

    def get_current_strategy(self):
        """Get the current strategy based on user settings"""
        strategy_type = user_settings.get_strategy()
        logger.info(f"Loading strategy: {strategy_type}")

        if strategy_type == "MACD":
            return MACDStrategy()
        elif strategy_type == "BOLLINGER":
            return BollingerStrategy()
        elif strategy_type == "ENHANCED_RSI_EMA":
            return EnhancedRSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )
        else:  # Default to ENHANCED_RSI_EMA
            return EnhancedRSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )

    async def initialize(self):
        """Initialize the trading system with comprehensive error handling"""
        logger.info("🔧 Starting system initialization...")

        try:
            # Check account balance
            balance = await self.mexc_client.get_balance()
            logger.info(f"Account balance retrieved: {balance}")

            # Send startup message
            await self.bot.send_alert(
                "Debug System Started",
                f"🧪 Debug trading bot connected\n"
                f"📊 Strategy: {user_settings.get_strategy()}\n"
                f"🛡️ Risk Management: Active\n"
                f"📱 Settings & Emergency: Available\n"
                f"Monitoring pairs: {', '.join(Config.TRADING_PAIRS)}",
                "success",
            )
            logger.info("✅ Startup message sent")

            # Start data feed
            await self.data_feed.start(Config.TRADING_PAIRS, Config.KLINE_INTERVAL)
            logger.info("✅ Data feed started")

            # Subscribe to kline updates for signal generation
            for pair in Config.TRADING_PAIRS:
                self.data_feed.subscribe(pair, "kline_closed", self.on_kline_update)
                logger.info(f"✅ Subscribed to {pair} updates")

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            await self.bot.send_alert(
                "System Initialization Failed",
                f"Error: {str(e)}\nPlease check configuration and restart.",
                "error",
            )
            raise

    async def start_bot_only(self):
        """Start only the Telegram bot for testing settings"""
        logger.info("🤖 Starting bot-only mode for testing...")

        try:
            # Send test message
            await self.bot.send_alert(
                "Bot Test Mode",
                "🧪 Bot running in test mode\n\n"
                "Available commands:\n"
                "• /start - Show welcome\n"
                "• /status - System status\n"
                "• /settings - Configure bot\n"
                "• /emergency - Emergency controls\n"
                "• /test_signal - Send test signal\n\n"
                "✅ All settings features should work!",
                "info",
            )

            # Set up command handlers with error wrapping
            from telegram.ext import CommandHandler, CallbackQueryHandler

            # Wrap commands with error handling
            async def safe_start(update, context):
                try:
                    await self.bot.start_command(update, context)
                except Exception as e:
                    logger.error(f"Error in start command: {e}")
                    await update.message.reply_text("❌ Error in start command")

            async def safe_settings(update, context):
                try:
                    logger.info("📱 Settings command received")
                    await self.bot.settings_command(update, context)
                    logger.info("✅ Settings command completed")
                except Exception as e:
                    logger.error(f"❌ Error in settings command: {e}")
                    import traceback

                    traceback.print_exc()
                    await update.message.reply_text(f"❌ Settings error: {str(e)}")

            async def safe_emergency(update, context):
                try:
                    logger.info("🚨 Emergency command received")
                    await self.bot.emergency_command(update, context)
                    logger.info("✅ Emergency command completed")
                except Exception as e:
                    logger.error(f"❌ Error in emergency command: {e}")
                    await update.message.reply_text(f"❌ Emergency error: {str(e)}")

            async def safe_callback(update, context):
                try:
                    logger.info(f"🔘 Callback received: {update.callback_query.data}")
                    await self.bot.handle_callback(update, context)
                    logger.info("✅ Callback handled successfully")
                except Exception as e:
                    logger.error(f"❌ Error in callback: {e}")
                    await update.callback_query.message.reply_text(
                        f"❌ Callback error: {str(e)}"
                    )

            # Add handlers
            self.bot.application.add_handler(CommandHandler("start", safe_start))
            self.bot.application.add_handler(
                CommandHandler("status", self.bot.status_command)
            )
            self.bot.application.add_handler(CommandHandler("settings", safe_settings))
            self.bot.application.add_handler(
                CommandHandler("emergency", safe_emergency)
            )
            self.bot.application.add_handler(
                CommandHandler("test_signal", self.bot.test_signal_command)
            )
            self.bot.application.add_handler(CallbackQueryHandler(safe_callback))

            logger.info("✅ All handlers added with error protection")

            # Start polling
            logger.info("🔄 Starting bot polling...")
            await self.bot.application.initialize()
            await self.bot.application.start()

            logger.info("🎉 Bot is now running! Try /settings in Telegram")
            await self.bot.application.updater.start_polling()

            # Keep running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("⏹️ Bot stopped by user")
            finally:
                await self.bot.application.stop()

        except Exception as e:
            logger.error(f"❌ Bot startup failed: {e}")
            import traceback

            traceback.print_exc()


class PerformanceTracker:
    """Simple performance tracker for debug mode"""

    def __init__(self):
        self.trades = []
        self.daily_pnl = 0.0


async def main():
    """Main function for debug mode"""
    logger.info("🚀 Starting DEBUG Trading Bot")
    logger.info("=" * 60)

    try:
        # Create system
        system = DebugTradingSystem()

        # Start in bot-only mode for testing
        await system.start_bot_only()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 DEBUG MODE - Trading Bot")
    print("This version has enhanced error handling and logging")
    print("Check debug_bot.log for detailed logs")
    print("=" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Debug bot stopped")
    except Exception as e:
        print(f"\n❌ Debug bot crashed: {e}")
        import traceback

        traceback.print_exc()
