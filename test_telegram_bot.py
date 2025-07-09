#!/usr/bin/env python3
"""
Simple Telegram bot test to check connection and commands
"""

import asyncio
import logging
from config.config import Config
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleBotTest:
    def __init__(self):
        # Validate config
        try:
            Config.validate()
            print("✅ Configuration validation passed")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return

        # Create application
        self.application = (
            Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        )
        print(f"✅ Bot created with token: {Config.TELEGRAM_BOT_TOKEN[:20]}...")
        print(f"✅ Target chat ID: {Config.TELEGRAM_CHAT_ID}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        print("📱 Received /start command")
        await update.message.reply_text(
            "🤖 **Bot Test**\n\nCommands working:\n• /start\n• /test\n• /settings_test",
            parse_mode="Markdown",
        )

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command"""
        print("📱 Received /test command")
        await update.message.reply_text(
            "✅ Test command working!\n\nBot is responding correctly.",
            parse_mode="Markdown",
        )

    async def settings_test_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /settings_test command"""
        print("📱 Received /settings_test command")

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = [
            [InlineKeyboardButton("✅ Option 1", callback_data="test_1")],
            [InlineKeyboardButton("✅ Option 2", callback_data="test_2")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🧪 **Settings Test**\n\nThis tests the inline keyboard:",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()

        print(f"📱 Received callback: {query.data}")

        await query.edit_message_text(
            f"✅ Callback received: {query.data}\n\nInline keyboards working!",
            parse_mode="Markdown",
        )

    def run(self):
        """Start the test bot"""
        print("🚀 Starting simple bot test...")

        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("test", self.test_command))
        self.application.add_handler(
            CommandHandler("settings_test", self.settings_test_command)
        )

        from telegram.ext import CallbackQueryHandler

        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

        print("✅ Handlers added, starting polling...")

        # Start polling
        self.application.run_polling()


if __name__ == "__main__":
    print("🧪 Telegram Bot Connection Test")
    print("=" * 40)

    bot_test = SimpleBotTest()
    if hasattr(bot_test, "application"):
        print("\n🚀 Starting bot test...")
        print("Try these commands in Telegram:")
        print("• /start")
        print("• /test")
        print("• /settings_test")
        print("\nPress Ctrl+C to stop")
        print("=" * 40)

        try:
            bot_test.run()
        except KeyboardInterrupt:
            print("\n⏹️ Bot test stopped")
    else:
        print("❌ Bot test failed to initialize")
