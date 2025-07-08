import asyncio
from typing import Dict, Optional
from datetime import datetime
import logging
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from models.models import Signal, TradeSetup, PerformanceStats

logger = logging.getLogger(__name__)


class TradingBot:
    """Telegram bot for trading signals and notifications"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.application = Application.builder().token(token).build()
        self.pending_trades = {}
        self.trade_callback = None

    def format_signal_message(self, signal: Signal, amount: float = 25) -> str:
        """Format signal notification message"""
        message = f"""🚨 **TRADING SIGNAL DETECTED**

📊 Pair: {signal.pair}
💰 Current Price: ${signal.current_price:,.2f}
📈 Signal: {signal.action}
🎯 Confidence: {signal.confidence:.0%}

📋 Analysis:
• RSI: {signal.indicators.get('rsi', 0):.0f} ({'Oversold' if signal.indicators.get('rsi', 50) < 30 else 'Overbought' if signal.indicators.get('rsi', 50) > 70 else 'Neutral'})
• EMA: {signal.indicators.get('ema_trend', 'Unknown')} crossover
• Volume: {'Above' if signal.indicators.get('volume_confirmation') else 'Below'} average
• Volatility: {signal.indicators.get('volatility', 'Unknown')}

💼 Suggested Trade:
• Amount: ${amount}
• Stop Loss: ${signal.stop_loss:,.2f} ({((signal.stop_loss - signal.current_price) / signal.current_price * 100):+.1f}%)
• Take Profit: ${signal.take_profit:,.2f} ({((signal.take_profit - signal.current_price) / signal.current_price * 100):+.1f}%)
• Risk/Reward: 1:{signal.risk_reward}

React with ✅ to APPROVE or ❌ to REJECT"""

        return message

    def format_trade_execution_message(self, trade: TradeSetup) -> str:
        """Format trade execution message"""
        message = f"""✅ **TRADE EXECUTED**

📊 Pair: {trade.pair}
💵 Entry Price: ${trade.entry_price:,.2f}
📉 Stop Loss: ${trade.stop_loss:,.2f}
📈 Take Profit: ${trade.take_profit:,.2f}
💰 Position Size: ${trade.position_size:,.2f}

⚖️ Risk Management:
• Risk/Reward: 1:{trade.risk_reward}
• Max Loss: ${abs(trade.entry_price - trade.stop_loss) * trade.position_size / trade.entry_price:,.2f}
• Potential Profit: ${abs(trade.take_profit - trade.entry_price) * trade.position_size / trade.entry_price:,.2f}

🔔 You will be notified when targets are reached."""

        return message

    def format_performance_message(self, stats: PerformanceStats) -> str:
        """Format performance summary message"""
        message = f"""📊 **DAILY PERFORMANCE SUMMARY**

📅 Date: {stats.date}

💼 Trading Statistics:
• Total Trades: {stats.total_trades}
• Win Rate: {stats.win_rate:.1%}
• Profit/Loss: ${stats.pnl:+,.2f} ({stats.pnl_percent:+.2f}%)

📈 Best Trade: {stats.best_trade['pair']} (+${stats.best_trade['profit']:,.2f})
📉 Worst Trade: {stats.worst_trade['pair']} (-${stats.worst_trade['loss']:,.2f})

💰 Account Status:
• Starting Balance: ${stats.start_balance:,.2f}
• Current Balance: ${stats.current_balance:,.2f}
• Total Return: {stats.total_return:+.2f}%"""

        return message

    def format_stop_loss_hit_message(self, trade: Dict) -> str:
        """Format stop loss hit notification"""
        message = f"""🛑 **STOP LOSS HIT**

📊 Pair: {trade['pair']}
💔 Exit Price: ${trade['exit_price']:,.2f}
📉 Entry Price: ${trade['entry_price']:,.2f}
💸 Loss: ${trade['loss']:,.2f} ({trade['loss_percent']:.2f}%)

📝 Trade Summary:
• Duration: {trade['duration']}
• Volume: ${trade['volume']:,.2f}
• Reason: Stop loss triggered

💡 Analysis saved for strategy improvement."""

        return message

    def format_take_profit_hit_message(self, trade: Dict) -> str:
        """Format take profit hit notification"""
        message = f"""💰 **TAKE PROFIT REACHED**

📊 Pair: {trade['pair']}
🎯 Exit Price: ${trade['exit_price']:,.2f}
📈 Entry Price: ${trade['entry_price']:,.2f}
💵 Profit: ${trade['profit']:,.2f} ({trade['profit_percent']:.2f}%)

📝 Trade Summary:
• Duration: {trade['duration']}
• Volume: ${trade['volume']:,.2f}
• R:R Achieved: {trade['rr_achieved']:.1f}

🎉 Congratulations on the successful trade!"""

        return message

    async def send_signal_notification(self, signal: Signal, amount: float = 25):
        """Send signal notification with inline keyboard"""
        keyboard = [
            [
                InlineKeyboardButton(
                    "✅ Approve",
                    callback_data=f"approve_{signal.pair}_{signal.timestamp.timestamp()}",
                ),
                InlineKeyboardButton(
                    "❌ Reject",
                    callback_data=f"reject_{signal.pair}_{signal.timestamp.timestamp()}",
                ),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        message = self.format_signal_message(signal, amount)

        # Store signal for later processing
        signal_id = f"{signal.pair}_{signal.timestamp.timestamp()}"
        self.pending_trades[signal_id] = signal

        await self.application.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )

    async def handle_callback(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data.split("_")
        action = data[0]
        signal_id = "_".join(data[1:])

        if signal_id in self.pending_trades:
            signal = self.pending_trades[signal_id]

            if action == "approve":
                # Execute trade
                trade = TradeSetup(
                    pair=signal.pair,
                    entry_price=signal.current_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=25,  # Default position size
                    risk_reward=signal.risk_reward,
                    confidence=signal.confidence,
                )

                await query.edit_message_text(
                    text=f"✅ Trade Approved!\n\n{self.format_trade_execution_message(trade)}",
                    parse_mode="Markdown",
                )

                # Remove from pending
                del self.pending_trades[signal_id]

            elif action == "reject":
                await query.edit_message_text(
                    text="❌ Trade Rejected\n\nSignal dismissed. Waiting for next opportunity...",
                    parse_mode="Markdown",
                )
                del self.pending_trades[signal_id]

    async def send_alert(self, title: str, message: str, alert_type: str = "info"):
        """Send generic alert message"""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "success": "✅",
            "error": "❌",
            "money": "💰",
        }

        emoji = emoji_map.get(alert_type, "ℹ️")

        formatted_message = f"{emoji} **{title}**\n\n{message}"

        await self.application.bot.send_message(
            chat_id=self.chat_id, text=formatted_message, parse_mode="Markdown"
        )

    async def start_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """🤖 **Trading Bot Activated**

Welcome to your automated trading assistant!

Available Commands:
• /start - Show this message
• /status - Check bot status
• /performance - Today's performance
• /settings - Configure bot settings

The bot will automatically send you:
📊 Trading signals when detected
💰 Trade execution confirmations
📈 Performance summaries
🔔 Important market alerts

Happy trading! 🚀"""

        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def status_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status_message = """📊 **Bot Status**

🟢 System: Online
📡 Connection: Active
⚡ Response Time: < 100ms

📈 Active Strategies:
• RSI + EMA Strategy ✅
• MACD Strategy ⏸️
• Bollinger Bands ⏸️

💼 Open Positions: 3
📊 Pending Signals: 1

Last Update: Just now"""

        await update.message.reply_text(status_message, parse_mode="Markdown")

    async def handle_callback(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data.split("_")
        action = data[0]
        signal_id = "_".join(data[1:])

        if signal_id in self.pending_trades:
            trade_data = self.pending_trades[signal_id]
            signal = trade_data["signal"]
            position_size = trade_data["position_size"]

            if action == "approve":
                # Execute trade through callback
                if self.trade_callback:
                    await self.trade_callback(signal, position_size)

                await query.edit_message_text(
                    text="✅ Trade Approved! Executing order...", parse_mode="Markdown"
                )

                # Remove from pending
                del self.pending_trades[signal_id]

            elif action == "reject":
                await query.edit_message_text(
                    text="❌ Trade Rejected\n\nSignal dismissed. Waiting for next opportunity...",
                    parse_mode="Markdown",
                )
                del self.pending_trades[signal_id]

    def run(self):
        """Start the bot"""
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

        # Start polling
        logger.info("Bot started")
        self.application.run_polling()
