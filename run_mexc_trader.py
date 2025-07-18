#!/usr/bin/env python3
"""
MEXC Automated Trader - Simple Setup & Run Script
Run this script to start the automated trading system
"""

import asyncio
import os
import sys
from pathlib import Path


def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")

    required_packages = ["telegram", "aiohttp", "pandas", "python-dotenv"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All requirements satisfied")
    return True


def check_configuration():
    """Check if configuration is properly set"""
    print("🔧 Checking configuration...")

    # Try to load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except:
        pass

    required_vars = {
        "MEXC_API_KEY": "MEXC Exchange API Key",
        "MEXC_API_SECRET": "MEXC Exchange API Secret",
        "TELEGRAM_BOT_TOKEN": "Telegram Bot Token",
        "TELEGRAM_CHAT_ID": "Telegram Chat ID",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append((var, description))

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var, desc in missing_vars:
            print(f"   • {var} - {desc}")

        print("\n📝 Create a .env file with these variables:")
        print("   MEXC_API_KEY=your_mexc_api_key")
        print("   MEXC_API_SECRET=your_mexc_api_secret")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")

        print("\n💡 How to get these:")
        print("   • MEXC API: https://mexc.com/user/api")
        print("   • Telegram Bot: https://t.me/BotFather")
        print("   • Chat ID: Send /start to @userinfobot")

        return False

    print("✅ Configuration complete")
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")

    directories = ["logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("✅ Directories created")


def show_welcome():
    """Show welcome message"""
    print("\n🤖 MEXC AUTOMATED TRADING SYSTEM")
    print("=" * 50)
    print("💰 Maximum trade volume: $5 per trade")
    print("🛡️ Risk-controlled trading with stop-loss")
    print("📱 Simple button interface via Telegram")
    print("🔍 Automatic signal detection")
    print("=" * 50)


def show_instructions():
    """Show usage instructions"""
    print("\n📋 USAGE INSTRUCTIONS:")
    print("=" * 30)
    print("1. 📱 Go to your Telegram bot")
    print("2. 💬 Send /start to begin")
    print("3. 📊 The bot will monitor markets")
    print("4. 🚨 You'll get signals with TRADE buttons")
    print("5. ✅ Click buttons to approve/reject trades")
    print("6. 💰 Maximum $5 per trade (safe trading)")
    print("7. 📈 Automatic stop-loss protection")
    print("\n🔧 Commands:")
    print("   /start - Start the trader")
    print("   /status - Check system status")
    print("   /balance - Show account balance")
    print("   /scan - Manual signal scan")
    print("   /stop - Stop the trader")
    print("\n⚠️  IMPORTANT:")
    print("   • This is for MEXC exchange only")
    print("   • Maximum $5 per trade for safety")
    print("   • Always monitor your trades")
    print("   • Use money you can afford to lose")


async def run_trader():
    """Run the MEXC automated trader"""
    try:
        # Import and run the trader
        from mexc_automated_trader import MEXCAutomatedTrader

        print("\n🚀 Starting MEXC Automated Trader...")
        print("⏰ Current time:", asyncio.get_event_loop().time())
        print("📡 Connecting to MEXC...")
        print("🤖 Starting Telegram bot...")

        trader = MEXCAutomatedTrader()
        await trader.start_monitoring()

        # Keep running
        while trader.running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Trader stopped by user")
        if "trader" in locals():
            await trader.stop_monitoring()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Check your configuration and try again")


def main():
    """Main setup and run function"""
    show_welcome()

    # Check requirements
    if not check_requirements():
        return

    # Check configuration
    if not check_configuration():
        return

    # Create directories
    create_directories()

    # Show instructions
    show_instructions()

    # Confirm start
    print("\n" + "=" * 50)
    print("✅ System ready!")

    start = input("\n🚀 Start the MEXC Automated Trader? (y/N): ").lower().strip()

    if start == "y":
        print("\n🎯 Starting trader...")
        print("💡 Press Ctrl+C to stop at any time")
        print("📱 Check your Telegram for the /start command")
        print("-" * 50)

        try:
            asyncio.run(run_trader())
        except KeyboardInterrupt:
            print("\n👋 Thanks for using MEXC Automated Trader!")
    else:
        print("\n👋 Setup complete. Run this script again when ready to trade!")


if __name__ == "__main__":
    main()
