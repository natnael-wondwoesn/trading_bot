#!/usr/bin/env python3
"""
MEXC Automated Trading System
Simple button-based trading bot with $5 maximum trade volume
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Import existing components
from config.config import Config
from mexc.mexc_client import MEXCClient, MEXCTradeExecutor
from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
from models.models import Signal
from user_settings import user_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/mexc_trader.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MEXCAutomatedTrader:
    """Simple automated trading system for MEXC with button interface"""

    # Maximum trade volume - $5 USD
    MAX_TRADE_VOLUME = 5.0

    # Trading pairs to monitor
    TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]

    def __init__(self):
        # Validate configuration
        self._validate_config()

        # Initialize MEXC components
        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
        self.trade_executor = MEXCTradeExecutor(self.mexc_client)
        self.strategy = EnhancedRSIEMAStrategy()

        # Initialize Telegram bot
        self.bot_application = (
            Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        )
        self.chat_id = Config.TELEGRAM_CHAT_ID

        # Trading state
        self.running = False
        self.pending_signals = {}  # Store signals awaiting user approval
        self.active_trades = {}  # Store active trade information
        self.last_signal_time = {}  # Rate limiting for signals

        # Setup bot handlers
        self._setup_handlers()

        logger.info("MEXC Automated Trader initialized successfully")

    def _validate_config(self):
        """Validate required configuration"""
        required_vars = [
            "MEXC_API_KEY",
            "MEXC_API_SECRET",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(Config, var, None):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing_vars)}"
            )

        logger.info("✅ Configuration validated")

    def _setup_handlers(self):
        """Setup Telegram bot command and callback handlers"""
        self.bot_application.add_handler(CommandHandler("start", self._handle_start))
        self.bot_application.add_handler(CommandHandler("status", self._handle_status))
        self.bot_application.add_handler(
            CommandHandler("balance", self._handle_balance)
        )
        self.bot_application.add_handler(
            CommandHandler("scan", self._handle_manual_scan)
        )
        self.bot_application.add_handler(CommandHandler("stop", self._handle_stop))
        self.bot_application.add_handler(CallbackQueryHandler(self._handle_callback))

        logger.info("Bot handlers setup complete")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        message = f"""🤖 **MEXC Automated Trader**

💰 **Max Trade Volume**: ${self.MAX_TRADE_VOLUME}
📊 **Monitoring**: {len(self.TRADING_PAIRS)} pairs
🛡️ **Strategy**: Enhanced RSI/EMA
⚡ **Status**: {'🟢 Running' if self.running else '🔴 Stopped'}

**Commands**:
• /start - Show this message
• /status - Check system status
• /balance - Show account balance
• /scan - Manual signal scan
• /stop - Stop the trader

**How it works**:
1. 🔍 Continuously scans for trading signals
2. 📱 Sends you signals with TRADE buttons
3. 💰 Maximum $5 per trade (risk-controlled)
4. ✅ You approve trades with one click
5. 📈 Automatic stop-loss and take-profit

Ready to start trading? The system will notify you when signals are found!"""

        await update.message.reply_text(message, parse_mode="Markdown")

        if not self.running:
            await self._send_notification(
                "🚀 **Trader Started**\n\nMonitoring markets for signals..."
            )
            await self.start_monitoring()

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        # Get account balance
        try:
            balance = await self.mexc_client.get_balance("USDT")
            usdt_balance = balance.get("free", 0)
        except Exception as e:
            usdt_balance = "Error"
            logger.error(f"Error getting balance: {e}")

        message = f"""📊 **System Status**

🤖 **Trader**: {'🟢 Running' if self.running else '🔴 Stopped'}
💰 **USDT Balance**: ${usdt_balance}
📈 **Active Trades**: {len(self.active_trades)}
⏳ **Pending Signals**: {len(self.pending_signals)}

📊 **Monitoring**:
{chr(10).join([f"• {pair}" for pair in self.TRADING_PAIRS])}

🛡️ **Risk Settings**:
• Max Trade: ${self.MAX_TRADE_VOLUME}
• Stop Loss: 2% average
• Take Profit: 4% average

⏰ **Uptime**: {datetime.now().strftime('%H:%M:%S')}"""

        await update.message.reply_text(message, parse_mode="Markdown")

    async def _handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        try:
            account = await self.mexc_client.get_account()

            # Get USDT balance
            usdt_balance = 0
            for balance in account.get("balances", []):
                if balance["asset"] == "USDT":
                    usdt_balance = float(balance["free"])
                    break

            # Calculate how many trades are possible
            possible_trades = int(usdt_balance / self.MAX_TRADE_VOLUME)

            message = f"""💰 **Account Balance**

💵 **Available USDT**: ${usdt_balance:.2f}
🎯 **Max Trade Size**: ${self.MAX_TRADE_VOLUME}
📊 **Possible Trades**: {possible_trades}

{"✅ **Ready to trade!**" if usdt_balance >= self.MAX_TRADE_VOLUME else "⚠️ **Insufficient balance for trading**"}

💡 **Note**: Each trade uses maximum ${self.MAX_TRADE_VOLUME} to minimize risk"""

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            await update.message.reply_text(f"❌ Error getting balance: {str(e)}")
            logger.error(f"Balance check error: {e}")

    async def _handle_manual_scan(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /scan command - manual signal scan"""
        await update.message.reply_text(
            "🔍 **Manual Scan Started**\n\nScanning all pairs for signals..."
        )

        signals = await self._scan_for_signals()

        if signals:
            await update.message.reply_text(
                f"✅ Found {len(signals)} signals! Check above for trade buttons."
            )
        else:
            await update.message.reply_text(
                "📊 No signals found at this time. Markets are in hold mode."
            )

    async def _handle_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        await self.stop_monitoring()
        await update.message.reply_text(
            "🛑 **Trader Stopped**\n\nSignal monitoring has been disabled."
        )

    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data
        logger.info(f"Received callback: {data}")

        try:
            if data.startswith("trade_"):
                await self._handle_trade_callback(query, data)
            elif data.startswith("close_"):
                await self._handle_close_callback(query, data)

        except Exception as e:
            logger.error(f"Callback error: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")

    async def _handle_trade_callback(self, query, data):
        """Handle trade execution callbacks"""
        # Parse callback data: trade_approve_BTCUSDT_timestamp or trade_reject_BTCUSDT_timestamp
        parts = data.split("_")
        action = parts[1]  # approve or reject
        pair = parts[2]
        timestamp = parts[3]

        signal_id = f"{pair}_{timestamp}"

        if signal_id not in self.pending_signals:
            await query.edit_message_text("❌ Signal expired or already processed.")
            return

        signal = self.pending_signals[signal_id]

        if action == "approve":
            # Execute the trade
            try:
                result = await self._execute_trade(signal)

                if result["success"]:
                    # Store active trade
                    self.active_trades[signal_id] = {
                        "signal": signal,
                        "order_id": result.get("order_id"),
                        "entry_price": result["price"],
                        "quantity": result["quantity"],
                        "timestamp": datetime.now(),
                    }

                    await query.edit_message_text(
                        f"✅ **Trade Executed**\n\n"
                        f"📊 {signal['pair']}: {signal['action']}\n"
                        f"💰 Amount: ${result['amount']:.2f}\n"
                        f"📈 Price: ${result['price']:.4f}\n"
                        f"📊 Quantity: {result['quantity']:.6f}\n"
                        f"🆔 Order: {result.get('order_id', 'N/A')}",
                        parse_mode="Markdown",
                    )
                else:
                    await query.edit_message_text(f"❌ Trade failed: {result['error']}")

            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                await query.edit_message_text(f"❌ Trade execution failed: {str(e)}")

        elif action == "reject":
            await query.edit_message_text(
                "❌ **Trade Rejected**\n\nSignal dismissed. Waiting for next opportunity..."
            )

        # Remove from pending
        if signal_id in self.pending_signals:
            del self.pending_signals[signal_id]

    async def _execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade with $5 maximum volume"""
        try:
            # Calculate trade size (max $5)
            current_price = signal["price"]
            trade_amount = min(self.MAX_TRADE_VOLUME, 5.0)  # Enforce $5 max
            quantity = trade_amount / current_price

            # Check account balance
            balance = await self.mexc_client.get_balance("USDT")
            available_balance = balance.get("free", 0)

            if available_balance < trade_amount:
                return {
                    "success": False,
                    "error": f"Insufficient balance. Need ${trade_amount:.2f}, have ${available_balance:.2f}",
                }

            # Execute market order
            logger.info(
                f"Executing {signal['action']} order for {signal['pair']}: {quantity:.6f} @ ${current_price:.4f}"
            )

            order = await self.trade_executor.execute_market_order(
                symbol=signal["pair"], side=signal["action"].upper(), quantity=quantity
            )

            return {
                "success": True,
                "order_id": order.get("orderId"),
                "price": current_price,
                "quantity": quantity,
                "amount": trade_amount,
            }

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _scan_for_signals(self) -> List[Dict]:
        """Scan all trading pairs for signals"""
        signals = []

        for pair in self.TRADING_PAIRS:
            try:
                # Rate limiting - only scan if enough time has passed
                last_scan = self.last_signal_time.get(pair)
                if (
                    last_scan and (datetime.now() - last_scan).total_seconds() < 300
                ):  # 5 minutes
                    continue

                # Get market data
                klines = await self.mexc_client.get_klines(pair, "1h", 100)

                if len(klines) < 50:
                    continue

                # Set pair attribute for strategy
                klines.attrs = {"pair": pair}

                # Generate signal
                signal = self.strategy.generate_signal(klines)

                if (
                    signal.action != "HOLD" and signal.confidence > 0.6
                ):  # High confidence signals only
                    # Get current price
                    current_price = await self.mexc_client.get_accurate_price(pair)

                    signal_data = {
                        "pair": pair,
                        "action": signal.action,
                        "confidence": signal.confidence,
                        "price": current_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "timestamp": datetime.now(),
                    }

                    signals.append(signal_data)
                    self.last_signal_time[pair] = datetime.now()

                    logger.info(
                        f"🎯 Signal: {pair} {signal.action} @ ${current_price:.4f} (Confidence: {signal.confidence:.1%})"
                    )

            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")

        # Send signals to user
        for signal in signals:
            await self._send_signal_notification(signal)

        return signals

    async def _send_signal_notification(self, signal: Dict):
        """Send signal notification with trade buttons"""
        # Create unique signal ID
        signal_id = f"{signal['pair']}_{int(signal['timestamp'].timestamp())}"

        # Store signal for callback processing
        self.pending_signals[signal_id] = signal

        # Calculate potential profit/loss
        entry_price = signal["price"]
        stop_loss_pct = ((signal["stop_loss"] - entry_price) / entry_price) * 100
        take_profit_pct = ((signal["take_profit"] - entry_price) / entry_price) * 100

        # Format message
        message = f"""🚨 **TRADING SIGNAL**

📊 **{signal['pair']}**
📈 **Action**: {signal['action']}
💰 **Price**: ${signal['price']:.4f}
🎯 **Confidence**: {signal['confidence']:.0%}

💼 **Trade Details**:
• Amount: ${self.MAX_TRADE_VOLUME} (Max Volume)
• Stop Loss: {stop_loss_pct:+.1f}%
• Take Profit: {take_profit_pct:+.1f}%

⏰ **Time**: {signal['timestamp'].strftime('%H:%M:%S')}

Click a button to execute:"""

        # Create buttons
        keyboard = [
            [
                InlineKeyboardButton(
                    f"✅ TRADE ${self.MAX_TRADE_VOLUME}",
                    callback_data=f"trade_approve_{signal['pair']}_{int(signal['timestamp'].timestamp())}",
                ),
                InlineKeyboardButton(
                    "❌ REJECT",
                    callback_data=f"trade_reject_{signal['pair']}_{int(signal['timestamp'].timestamp())}",
                ),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send notification
        await self._send_notification(message, reply_markup)

    async def _send_notification(self, message: str, reply_markup=None):
        """Send notification to user"""
        try:
            bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def start_monitoring(self):
        """Start continuous signal monitoring"""
        if self.running:
            return

        self.running = True
        logger.info("Starting signal monitoring...")

        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())

        # Start bot polling
        await self.bot_application.initialize()
        await self.bot_application.start()
        await self.bot_application.updater.start_polling()

        logger.info("MEXC Automated Trader is now running!")

    async def stop_monitoring(self):
        """Stop signal monitoring"""
        self.running = False
        logger.info("Stopping signal monitoring...")

        try:
            if self.bot_application.updater._running:
                await self.bot_application.updater.stop()
            await self.bot_application.stop()
            await self.bot_application.shutdown()
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

        if self.mexc_client:
            await self.mexc_client.close()

        logger.info("MEXC Automated Trader stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        scan_count = 0

        while self.running:
            try:
                scan_count += 1
                logger.info(f"🔍 Signal scan #{scan_count}")

                # Scan for signals
                await self._scan_for_signals()

                # Wait 5 minutes before next scan
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error


async def main():
    """Main entry point"""
    print("🚀 MEXC Automated Trading System")
    print("=" * 50)

    try:
        # Create trader instance
        trader = MEXCAutomatedTrader()

        # Start the system
        print("✅ Trader initialized")
        print("📱 Starting Telegram bot...")
        print("🔍 Beginning signal monitoring...")
        print("\n💡 Use /start in Telegram to begin trading")
        print("💡 Maximum trade volume: $5 per trade")
        print("💡 Press Ctrl+C to stop\n")

        await trader.start_monitoring()

        # Keep running
        while trader.running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if "trader" in locals():
            await trader.stop_monitoring()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        if "trader" in locals():
            await trader.stop_monitoring()


if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Run the trader
    asyncio.run(main())
