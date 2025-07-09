#!/usr/bin/env python3
"""
Simple bot focused on testing settings functionality
"""

import asyncio
import logging
from config.config import Config
from user_settings import user_settings
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Simple logging without unicode
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SimpleSettingsBot:
    def __init__(self):
        # Validate config
        Config.validate()
        logger.info("Configuration validated")

        # Create application
        self.application = (
            Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        )
        logger.info("Bot application created")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        logger.info("Received /start command")

        current_strategy = user_settings.get_strategy()
        strategy_names = {
            "RSI_EMA": "RSI + EMA",
            "MACD": "MACD",
            "BOLLINGER": "Bollinger Bands",
        }

        message = f"""🤖 Trading Bot Settings Test

Current Strategy: {strategy_names.get(current_strategy, current_strategy)}
Trading Status: {'Enabled' if user_settings.is_trading_enabled() else 'Disabled'}
Emergency Mode: {'ACTIVE' if user_settings.is_emergency_mode() else 'Normal'}

Available Commands:
• /start - Show this message
• /settings - Configure bot settings
• /emergency - Emergency controls
• /status - System status

Test the /settings command to see if it works!"""

        await update.message.reply_text(message, parse_mode="Markdown")

    async def settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /settings command"""
        logger.info("Received /settings command")

        try:
            keyboard = [
                [
                    InlineKeyboardButton("Strategy", callback_data="settings_strategy"),
                    InlineKeyboardButton(
                        "Risk Management", callback_data="settings_risk"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Notifications", callback_data="settings_notifications"
                    ),
                    InlineKeyboardButton(
                        "Emergency", callback_data="settings_emergency"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "View Settings", callback_data="settings_view"
                    ),
                    InlineKeyboardButton(
                        "Reset Defaults", callback_data="settings_reset"
                    ),
                ],
                [InlineKeyboardButton("Close", callback_data="settings_close")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "⚙️ TRADING BOT SETTINGS\n\nChoose a category to configure:",
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
            logger.info("Settings menu sent successfully")

        except Exception as e:
            logger.error(f"Error in settings command: {e}")
            await update.message.reply_text(f"Error in settings: {str(e)}")

    async def emergency_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /emergency command"""
        logger.info("Received /emergency command")

        try:
            is_emergency = user_settings.is_emergency_mode()
            trading_enabled = user_settings.is_trading_enabled()

            keyboard = [
                [
                    InlineKeyboardButton(
                        "EMERGENCY STOP" if not is_emergency else "Disable Emergency",
                        callback_data="emergency_toggle",
                    )
                ],
                [
                    InlineKeyboardButton(
                        "Close All Positions", callback_data="emergency_close_all"
                    ),
                    InlineKeyboardButton(
                        "Pause Trading", callback_data="emergency_pause"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Risk Status", callback_data="emergency_status"
                    ),
                    InlineKeyboardButton("Close", callback_data="emergency_close"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            message = f"""🚨 EMERGENCY CONTROL CENTER

System Status: {'EMERGENCY MODE' if is_emergency else 'Normal Operations'}
Trading: {'Enabled' if trading_enabled else 'Disabled'}

Emergency Actions Available:
• Emergency Stop: Halt all trading immediately
• Close All: Close all open positions
• Pause Trading: Stop new trades, keep existing

Use these controls for immediate risk management."""

            await update.message.reply_text(
                message, parse_mode="Markdown", reply_markup=reply_markup
            )
            logger.info("Emergency menu sent successfully")

        except Exception as e:
            logger.error(f"Error in emergency command: {e}")
            await update.message.reply_text(f"Error in emergency: {str(e)}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        logger.info("Received /status command")

        current_strategy = user_settings.get_strategy()
        strategy_names = {
            "RSI_EMA": "RSI + EMA",
            "MACD": "MACD",
            "BOLLINGER": "Bollinger Bands",
        }
        is_emergency = user_settings.is_emergency_mode()
        trading_enabled = user_settings.is_trading_enabled()
        risk_settings = user_settings.get_risk_settings()

        message = f"""📊 Bot Status

System: {'EMERGENCY MODE' if is_emergency else 'Normal Operations'}
Trading: {'Active' if trading_enabled else 'Paused'}

Current Strategy: {strategy_names.get(current_strategy, current_strategy)}

Risk Settings:
• Max Risk per Trade: {risk_settings['max_risk_per_trade']*100:.1f}%
• Max Positions: {risk_settings['max_open_positions']}
• Emergency Mode: {'ACTIVE' if is_emergency else 'Normal'}

Use /settings to modify configuration
Use /emergency for immediate controls"""

        await update.message.reply_text(message, parse_mode="Markdown")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data
        logger.info(f"Received callback: {data}")

        try:
            if data.startswith("settings_"):
                await self.handle_settings_callback(query, data)
            elif data.startswith("emergency_"):
                await self.handle_emergency_callback(query, data)
            elif data.startswith("strategy_set_"):
                strategy = data.split("_")[2]
                if user_settings.set_strategy(strategy):
                    strategy_names = {
                        "RSI_EMA": "RSI + EMA",
                        "MACD": "MACD",
                        "BOLLINGER": "Bollinger Bands",
                    }
                    await query.edit_message_text(
                        f"Strategy Updated\n\nActive strategy: {strategy_names.get(strategy, strategy)}\n\nThe system will use this strategy for new signals.",
                        parse_mode="Markdown",
                    )
                    logger.info(f"Strategy updated to: {strategy}")
                else:
                    await query.edit_message_text("Error updating strategy.")

        except Exception as e:
            logger.error(f"Error in callback: {e}")
            await query.edit_message_text(f"Error: {str(e)}")

    async def handle_settings_callback(self, query, data):
        """Handle settings callbacks"""
        action = data.split("_")[1] if len(data.split("_")) > 1 else None

        if action == "strategy":
            keyboard = [
                [
                    InlineKeyboardButton(
                        f"{'[ACTIVE]' if user_settings.get_strategy() == 'RSI_EMA' else ''} RSI + EMA",
                        callback_data="strategy_set_RSI_EMA",
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"{'[ACTIVE]' if user_settings.get_strategy() == 'MACD' else ''} MACD",
                        callback_data="strategy_set_MACD",
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"{'[ACTIVE]' if user_settings.get_strategy() == 'BOLLINGER' else ''} Bollinger Bands",
                        callback_data="strategy_set_BOLLINGER",
                    )
                ],
                [InlineKeyboardButton("Back", callback_data="settings_main")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "SELECT TRADING STRATEGY\n\nChoose your preferred strategy:",
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )

        elif action == "view":
            settings_text = user_settings.get_settings_summary()
            keyboard = [[InlineKeyboardButton("Back", callback_data="settings_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                settings_text, parse_mode="Markdown", reply_markup=reply_markup
            )

        elif action == "close":
            await query.edit_message_text("Settings closed. Use /settings to reopen.")

        else:
            # Return to main settings menu
            keyboard = [
                [
                    InlineKeyboardButton("Strategy", callback_data="settings_strategy"),
                    InlineKeyboardButton(
                        "Risk Management", callback_data="settings_risk"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Notifications", callback_data="settings_notifications"
                    ),
                    InlineKeyboardButton(
                        "Emergency", callback_data="settings_emergency"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "View Settings", callback_data="settings_view"
                    ),
                    InlineKeyboardButton(
                        "Reset Defaults", callback_data="settings_reset"
                    ),
                ],
                [InlineKeyboardButton("Close", callback_data="settings_close")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "TRADING BOT SETTINGS\n\nChoose a category to configure:",
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )

    async def handle_emergency_callback(self, query, data):
        """Handle emergency callbacks"""
        action = data.split("_")[1] if len(data.split("_")) > 1 else None

        if action == "toggle":
            if user_settings.is_emergency_mode():
                user_settings.disable_emergency_stop()
                await query.edit_message_text(
                    "Emergency Mode Disabled\n\nNormal trading operations resumed.",
                    parse_mode="Markdown",
                )
            else:
                user_settings.enable_emergency_stop()
                await query.edit_message_text(
                    "EMERGENCY MODE ACTIVATED\n\nAll trading has been stopped.\nUse /emergency to manage.",
                    parse_mode="Markdown",
                )

        elif action == "pause":
            user_settings.set_trading_enabled(False)
            await query.edit_message_text(
                "Trading Paused\n\nNew trades disabled.\nExisting positions remain active.",
                parse_mode="Markdown",
            )

        elif action == "status":
            risk = user_settings.get_risk_settings()
            emergency = user_settings.settings["emergency"]

            status_text = f"""RISK STATUS REPORT

Emergency Mode: {'ACTIVE' if emergency['emergency_mode'] else 'Normal'}
Trading Enabled: {'Yes' if risk['trading_enabled'] else 'No'}
Max Daily Loss: {emergency['max_daily_loss']*100:.1f}%

Current Risk:
• Max Risk per Trade: {risk['max_risk_per_trade']*100:.1f}%
• Max Open Positions: {risk['max_open_positions']}

Quick Actions Available:
Use /emergency for immediate risk controls"""

            await query.edit_message_text(status_text, parse_mode="Markdown")

        elif action == "close":
            await query.edit_message_text(
                "Emergency controls closed. Use /emergency to reopen."
            )

    def run(self):
        """Start the bot"""
        logger.info("Adding command handlers...")

        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(
            CommandHandler("emergency", self.emergency_command)
        )
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

        logger.info("Handlers added, starting bot polling...")

        # Start polling
        self.application.run_polling()


if __name__ == "__main__":
    print("Simple Settings Bot Test")
    print("Testing /settings and /emergency functionality")
    print("=" * 50)

    try:
        bot = SimpleSettingsBot()
        logger.info("Bot initialized successfully")

        print("\nBot is starting...")
        print("Try these commands in Telegram:")
        print("• /start")
        print("• /settings")
        print("• /emergency")
        print("• /status")
        print("\nPress Ctrl+C to stop")

        bot.run()

    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot error: {e}")
        import traceback

        traceback.print_exc()
