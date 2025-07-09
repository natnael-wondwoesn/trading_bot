import asyncio
from typing import Dict, Optional
from datetime import datetime
import logging
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from models.models import Signal, TradeSetup, PerformanceStats
from user_settings import user_settings

logger = logging.getLogger(__name__)


class TradingBot:
    """Telegram bot for trading signals and notifications"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.application = None  # Will be created in the proper event loop
        self.pending_trades = {}
        self.trade_callback = None
        self.trading_system = None  # Will be set by the main system

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

        if self.application:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
        else:
            # Fallback: create a simple bot instance for sending messages
            bot = Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )

    async def handle_callback(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data
        data_parts = data.split("_")

        # Handle settings callbacks
        if data.startswith("settings_"):
            await self.handle_settings_callback(query, data_parts)
            return

        # Handle emergency callbacks
        elif data.startswith("emergency_"):
            await self.handle_emergency_callback(query, data_parts)
            return

        # Handle strategy selection callbacks
        elif data.startswith("strategy_set_"):
            strategy = data_parts[2]
            if user_settings.set_strategy(strategy):
                strategy_names = {
                    "RSI_EMA": "RSI + EMA",
                    "MACD": "MACD",
                    "BOLLINGER": "Bollinger Bands",
                }
                await query.edit_message_text(
                    f"✅ **Strategy Updated**\n\nActive strategy: {strategy_names.get(strategy, strategy)}\n\nThe system will use this strategy for new signals.",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text(
                    "❌ Error updating strategy. Please try again.",
                    parse_mode="Markdown",
                )
            return

        # Handle notification toggle callbacks
        elif data.startswith("notif_toggle_"):
            notif_type = data_parts[2]
            if notif_type == "signal":
                new_state = user_settings.toggle_signal_alerts()
                await query.edit_message_text(
                    f"📱 Signal alerts {'enabled' if new_state else 'disabled'}.",
                    parse_mode="Markdown",
                )
            elif notif_type == "execution":
                new_state = user_settings.toggle_trade_execution_alerts()
                await query.edit_message_text(
                    f"📱 Trade execution alerts {'enabled' if new_state else 'disabled'}.",
                    parse_mode="Markdown",
                )
            return

        # Handle settings main menu callback
        elif data == "settings_main":
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔧 Strategy", callback_data="settings_strategy"
                    ),
                    InlineKeyboardButton(
                        "💰 Risk Management", callback_data="settings_risk"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📱 Notifications", callback_data="settings_notifications"
                    ),
                    InlineKeyboardButton(
                        "🚨 Emergency", callback_data="settings_emergency"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📊 View Current Settings", callback_data="settings_view"
                    ),
                    InlineKeyboardButton(
                        "🔄 Reset to Defaults", callback_data="settings_reset"
                    ),
                ],
                [InlineKeyboardButton("❌ Close", callback_data="settings_close")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "⚙️ **TRADING BOT SETTINGS**\n\nChoose a category to configure:",
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
            return

        # Handle settings reset confirmation
        elif data == "settings_reset_confirm":
            user_settings.reset_to_defaults()
            await query.edit_message_text(
                "🔄 **Settings Reset**\n\nAll settings have been restored to defaults.",
                parse_mode="Markdown",
            )
            return

        # Original trade callback handling
        action = data_parts[0]
        signal_id = "_".join(data_parts[1:])

        if signal_id in self.pending_trades:
            signal = self.pending_trades[signal_id]

            if action == "approve":
                # Create trade setup
                trade = TradeSetup(
                    pair=signal.pair,
                    entry_price=signal.current_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=25,  # Default position size
                    risk_reward=signal.risk_reward,
                    confidence=signal.confidence,
                )

                # Execute trade through callback if trading system is connected
                if self.trading_system:
                    try:
                        # Use thread-safe execution to avoid event loop conflicts
                        trade_info = (
                            await self.trading_system.execute_trade_thread_safe(
                                signal, 25
                            )
                        )
                        logger.info(
                            f"Trade executed via thread-safe callback for {signal.pair}"
                        )

                        # Send success notification
                        await self.send_alert(
                            "✅ Trade Executed Successfully",
                            f"Executed {signal.action} order for {signal.pair}\n"
                            f"💰 Position Size: ${trade_info['position_size']:.2f}\n"
                            f"📊 Quantity: {trade_info['quantity']:.6f}\n"
                            f"🆔 Order ID: {trade_info.get('order_id', 'N/A')}",
                            "success",
                        )
                    except Exception as e:
                        logger.error(f"Trade execution failed: {str(e)}")
                        # Send error notification to user
                        await self.send_alert(
                            "❌ Trade Execution Failed",
                            f"Failed to execute {signal.action} order for {signal.pair}: {str(e)}",
                            "error",
                        )
                elif self.trade_callback:
                    try:
                        await self.trade_callback(
                            signal, 25
                        )  # Fallback to original callback
                        logger.info(f"Trade executed via callback for {signal.pair}")
                    except Exception as e:
                        logger.error(f"Trade execution failed: {str(e)}")
                        # Send error notification to user
                        await self.send_alert(
                            "❌ Trade Execution Failed",
                            f"Failed to execute {signal.action} order for {signal.pair}: {str(e)}",
                            "error",
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

        if self.application:
            await self.application.bot.send_message(
                chat_id=self.chat_id, text=formatted_message, parse_mode="Markdown"
            )
        else:
            # Fallback: create a simple bot instance for sending messages
            bot = Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id, text=formatted_message, parse_mode="Markdown"
            )

    async def start_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        current_strategy = user_settings.get_strategy()
        strategy_names = {
            "RSI_EMA": "RSI + EMA",
            "MACD": "MACD",
            "BOLLINGER": "Bollinger Bands",
        }

        welcome_message = f"""🤖 **Trading Bot Activated**

Welcome to your automated trading assistant!

📊 **Current Strategy**: {strategy_names.get(current_strategy, current_strategy)}
🛡️ **Risk Management**: Active
⚡ **Status**: {'✅ Ready' if user_settings.is_trading_enabled() else '🚨 Emergency Mode'}

**Available Commands**:
• /start - Show this message
• /status - Check bot status
• /performance - Today's performance
• /settings - Configure bot settings
• /emergency - Emergency controls
• /test_signal - Send a fake signal for testing

**Settings & Risk Management**:
• Configure trading strategies
• Set stop loss and take profit levels
• Manage risk parameters
• Emergency stop functionality

The bot will automatically send you:
📊 Trading signals when detected
💰 Trade execution confirmations
📈 Performance summaries
🔔 Important market alerts

Use /settings to customize your trading preferences.
Use /emergency for immediate risk controls.

Happy trading! 🚀"""

        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def performance_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        try:
            if self.trading_system:
                # Use thread-safe version to avoid event loop conflicts
                stats = await self.trading_system.get_daily_performance_thread_safe()
                message = self.format_performance_message(stats)
                await update.message.reply_text(message, parse_mode="Markdown")
            else:
                await update.message.reply_text(
                    "📊 Performance data unavailable - system not initialized.",
                    parse_mode="Markdown",
                )
        except Exception as e:
            logger.error(f"Error in performance command: {str(e)}")
            await update.message.reply_text(
                "❌ Error retrieving performance data. Please try again later.",
                parse_mode="Markdown",
            )

    async def status_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Count pending signals
            pending_count = len(self.pending_trades)

            # Get active positions count
            active_positions = 0
            if self.trading_system:
                active_positions = len(self.trading_system.active_trades)

            # Get current settings
            current_strategy = user_settings.get_strategy()
            strategy_names = {
                "RSI_EMA": "RSI + EMA",
                "MACD": "MACD",
                "BOLLINGER": "Bollinger Bands",
            }
            is_emergency = user_settings.is_emergency_mode()
            trading_enabled = user_settings.is_trading_enabled()

            # System status
            system_status = "🔴 EMERGENCY" if is_emergency else "🟢 Online"
            trading_status = "📈 Active" if trading_enabled else "⏸️ Paused"

            status_message = f"""📊 **Bot Status**

{system_status} System: {'Emergency Mode' if is_emergency else 'Normal Operations'}
📡 Connection: Active
⚡ Response Time: < 100ms
🔧 Trading: {trading_status}

📈 **Current Strategy**: {strategy_names.get(current_strategy, current_strategy)}

💼 **Trading Summary**:
• Open Positions: {active_positions}
• Pending Signals: {pending_count}
• Max Positions: {user_settings.get_risk_settings()['max_open_positions']}

🛡️ **Risk Settings**:
• Max Risk per Trade: {user_settings.get_risk_settings()['max_risk_per_trade']*100:.1f}%
• Emergency Mode: {'🔴 ACTIVE' if is_emergency else '✅ Normal'}

Last Update: {current_time}

💡 Use /settings to modify configuration
🚨 Use /emergency for immediate controls"""

            await update.message.reply_text(status_message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error in status command: {str(e)}")
            await update.message.reply_text(
                "❌ Error retrieving status. Please try again later.",
                parse_mode="Markdown",
            )

    async def test_signal_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test_signal command - sends a fake signal for testing"""
        try:
            from models.models import Signal
            import random

            # Create a realistic test signal
            pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
            prices = {
                "BTCUSDT": 108320.50,
                "ETHUSDT": 2549.77,
                "BNBUSDT": 660.83,
                "SOLUSDT": 149.73,
                "ADAUSDT": 0.5774,
            }

            pair = random.choice(pairs)
            action = random.choice(["BUY", "SELL"])
            current_price = prices[pair]
            confidence = random.uniform(0.65, 0.95)

            if action == "BUY":
                stop_loss = current_price * random.uniform(0.96, 0.98)
                take_profit = current_price * random.uniform(1.04, 1.08)
            else:
                stop_loss = current_price * random.uniform(1.02, 1.04)
                take_profit = current_price * random.uniform(0.92, 0.96)

            risk_reward = abs(take_profit - current_price) / abs(
                current_price - stop_loss
            )

            # Generate realistic indicators
            rsi = random.uniform(25, 35) if action == "BUY" else random.uniform(65, 80)

            test_signal = Signal(
                pair=pair,
                action=action,
                current_price=current_price,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                indicators={
                    "rsi": rsi,
                    "ema_trend": "Bullish" if action == "BUY" else "Bearish",
                    "volume_confirmation": random.choice([True, False]),
                    "volatility": random.choice(["Low", "Normal", "High"]),
                },
                timestamp=datetime.now(),
            )

            # Send the test signal
            await self.send_signal_notification(test_signal, amount=25)

            # Confirm to user
            await update.message.reply_text(
                f"🧪 **Test Signal Sent!**\n\n"
                f"📊 {pair} {action} signal generated\n"
                f"💰 Price: ${current_price:,.2f}\n"
                f"🎯 Confidence: {confidence:.0%}\n"
                f"📱 Check above for the signal notification!",
                parse_mode="Markdown",
            )

        except Exception as e:
            logger.error(f"Error in test_signal command: {str(e)}")
            await update.message.reply_text(
                "❌ Error generating test signal. Please try again later.",
                parse_mode="Markdown",
            )

    async def settings_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command - main settings menu"""
        try:
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔧 Strategy", callback_data="settings_strategy"
                    ),
                    InlineKeyboardButton(
                        "💰 Risk Management", callback_data="settings_risk"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📱 Notifications", callback_data="settings_notifications"
                    ),
                    InlineKeyboardButton(
                        "🚨 Emergency", callback_data="settings_emergency"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📊 View Current Settings", callback_data="settings_view"
                    ),
                    InlineKeyboardButton(
                        "🔄 Reset to Defaults", callback_data="settings_reset"
                    ),
                ],
                [InlineKeyboardButton("❌ Close", callback_data="settings_close")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "⚙️ **TRADING BOT SETTINGS**\n\nChoose a category to configure:",
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )

        except Exception as e:
            logger.error(f"Error in settings command: {str(e)}")
            await update.message.reply_text(
                "❌ Error opening settings. Please try again later.",
                parse_mode="Markdown",
            )

    async def emergency_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /emergency command - emergency risk management"""
        try:
            is_emergency = user_settings.is_emergency_mode()
            trading_enabled = user_settings.is_trading_enabled()

            # Emergency status
            status_text = (
                "🔴 EMERGENCY MODE ACTIVE" if is_emergency else "✅ Normal Operations"
            )
            trading_text = "📈 Enabled" if trading_enabled else "❌ Disabled"

            keyboard = [
                [
                    InlineKeyboardButton(
                        (
                            "🚨 EMERGENCY STOP"
                            if not is_emergency
                            else "✅ Disable Emergency"
                        ),
                        callback_data="emergency_toggle",
                    )
                ],
                [
                    InlineKeyboardButton(
                        "🛑 Close All Positions", callback_data="emergency_close_all"
                    ),
                    InlineKeyboardButton(
                        "⏸️ Pause Trading", callback_data="emergency_pause"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "💸 Set Max Daily Loss", callback_data="emergency_max_loss"
                    ),
                    InlineKeyboardButton(
                        "📊 Risk Status", callback_data="emergency_status"
                    ),
                ],
                [InlineKeyboardButton("❌ Close", callback_data="emergency_close")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            message = f"""🚨 **EMERGENCY CONTROL CENTER**

📊 **Current Status**:
• System Status: {status_text}
• Trading: {trading_text}
• Open Positions: {len(self.trading_system.active_trades) if self.trading_system else 0}

⚠️ **Emergency Actions Available**:
• Emergency Stop: Halt all trading immediately
• Close All: Close all open positions
• Pause Trading: Stop new trades, keep existing
• Set Loss Limits: Configure emergency thresholds

💡 Use these controls for immediate risk management."""

            await update.message.reply_text(
                message, parse_mode="Markdown", reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in emergency command: {str(e)}")
            await update.message.reply_text(
                "❌ Error opening emergency controls. Please try again later.",
                parse_mode="Markdown",
            )

    async def handle_settings_callback(self, query, data_parts):
        """Handle settings-related callbacks"""
        try:
            action = data_parts[1] if len(data_parts) > 1 else None

            if action == "strategy":
                keyboard = [
                    [
                        InlineKeyboardButton(
                            f"{'✅' if user_settings.get_strategy() == 'RSI_EMA' else '🔘'} RSI + EMA",
                            callback_data="strategy_set_RSI_EMA",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"{'✅' if user_settings.get_strategy() == 'MACD' else '🔘'} MACD",
                            callback_data="strategy_set_MACD",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"{'✅' if user_settings.get_strategy() == 'BOLLINGER' else '🔘'} Bollinger Bands",
                            callback_data="strategy_set_BOLLINGER",
                        )
                    ],
                    [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    "🔧 **SELECT TRADING STRATEGY**\n\nChoose your preferred strategy:",
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                )

            elif action == "risk":
                risk = user_settings.get_risk_settings()
                keyboard = [
                    [
                        InlineKeyboardButton(
                            "📊 Max Risk per Trade", callback_data="risk_max_trade"
                        ),
                        InlineKeyboardButton(
                            "🛑 Stop Loss", callback_data="risk_stop_loss"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "🎯 Take Profit", callback_data="risk_take_profit"
                        ),
                        InlineKeyboardButton(
                            "📈 Max Positions", callback_data="risk_max_positions"
                        ),
                    ],
                    [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    f"""💰 **RISK MANAGEMENT SETTINGS**

Current Settings:
• Max Risk per Trade: {risk['max_risk_per_trade']*100:.1f}%
• Stop Loss: {risk['custom_stop_loss'] or f"{risk['stop_loss_atr']}x ATR"}{'%' if risk['custom_stop_loss'] else ''}
• Take Profit: {risk['custom_take_profit'] or f"{risk['take_profit_atr']}x ATR"}{'%' if risk['custom_take_profit'] else ''}
• Max Open Positions: {risk['max_open_positions']}

Select setting to modify:""",
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                )

            elif action == "notifications":
                notifications = user_settings.settings["notifications"]
                keyboard = [
                    [
                        InlineKeyboardButton(
                            f"Signal Alerts: {'✅' if notifications['signal_alerts'] else '❌'}",
                            callback_data="notif_toggle_signal",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"Trade Execution: {'✅' if notifications['trade_execution'] else '❌'}",
                            callback_data="notif_toggle_execution",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"Risk Warnings: {'✅' if notifications['risk_warnings'] else '❌'}",
                            callback_data="notif_toggle_risk",
                        )
                    ],
                    [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    "📱 **NOTIFICATION SETTINGS**\n\nToggle notification types:",
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                )

            elif action == "view":
                settings_text = user_settings.get_settings_summary()
                keyboard = [
                    [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    settings_text, parse_mode="Markdown", reply_markup=reply_markup
                )

            elif action == "reset":
                keyboard = [
                    [
                        InlineKeyboardButton(
                            "✅ Yes, Reset", callback_data="settings_reset_confirm"
                        ),
                        InlineKeyboardButton(
                            "❌ Cancel", callback_data="settings_main"
                        ),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    "⚠️ **RESET SETTINGS**\n\nThis will reset ALL settings to defaults.\nAre you sure?",
                    parse_mode="Markdown",
                    reply_markup=reply_markup,
                )

            elif action == "close":
                await query.edit_message_text(
                    "⚙️ Settings closed. Use /settings to reopen.", parse_mode="Markdown"
                )

        except Exception as e:
            logger.error(f"Error handling settings callback: {str(e)}")
            await query.edit_message_text(
                "❌ Error in settings. Please try /settings again.",
                parse_mode="Markdown",
            )

    async def handle_emergency_callback(self, query, data_parts):
        """Handle emergency-related callbacks"""
        try:
            action = data_parts[1] if len(data_parts) > 1 else None

            if action == "toggle":
                if user_settings.is_emergency_mode():
                    user_settings.disable_emergency_stop()
                    await query.edit_message_text(
                        "✅ **Emergency Mode Disabled**\n\nNormal trading operations resumed.",
                        parse_mode="Markdown",
                    )
                else:
                    user_settings.enable_emergency_stop()
                    await query.edit_message_text(
                        "🚨 **EMERGENCY MODE ACTIVATED**\n\nAll trading has been stopped.\nExisting positions remain open.\nUse /emergency to manage.",
                        parse_mode="Markdown",
                    )

            elif action == "close_all":
                if self.trading_system:
                    # Close all positions
                    closed_count = 0
                    for symbol in list(self.trading_system.active_trades.keys()):
                        try:
                            ticker = await self.trading_system.mexc_client.get_ticker(
                                symbol
                            )
                            current_price = float(ticker["lastPrice"])
                            await self.trading_system.close_position(
                                symbol, "emergency_close", current_price
                            )
                            closed_count += 1
                        except Exception as e:
                            logger.error(f"Error closing position {symbol}: {str(e)}")

                    await query.edit_message_text(
                        f"🛑 **All Positions Closed**\n\nClosed {closed_count} position(s).\nEmergency action completed.",
                        parse_mode="Markdown",
                    )
                else:
                    await query.edit_message_text(
                        "ℹ️ No trading system active or no positions to close.",
                        parse_mode="Markdown",
                    )

            elif action == "pause":
                user_settings.set_trading_enabled(False)
                await query.edit_message_text(
                    "⏸️ **Trading Paused**\n\nNew trades disabled.\nExisting positions remain active.\nUse /emergency to resume.",
                    parse_mode="Markdown",
                )

            elif action == "status":
                risk = user_settings.get_risk_settings()
                emergency = user_settings.settings["emergency"]

                status_text = f"""📊 **RISK STATUS REPORT**

🚨 **Emergency Settings**:
• Emergency Mode: {'🔴 ACTIVE' if emergency['emergency_mode'] else '✅ Normal'}
• Trading Enabled: {'✅ Yes' if risk['trading_enabled'] else '❌ No'}
• Max Daily Loss: {emergency['max_daily_loss']*100:.1f}%

💰 **Current Risk**:
• Max Risk per Trade: {risk['max_risk_per_trade']*100:.1f}%
• Open Positions: {len(self.trading_system.active_trades) if self.trading_system else 0}/{risk['max_open_positions']}

⚡ **Quick Actions Available**:
Use /emergency for immediate risk controls"""

                keyboard = [
                    [
                        InlineKeyboardButton(
                            "🔄 Refresh", callback_data="emergency_status"
                        )
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await query.edit_message_text(
                    status_text, parse_mode="Markdown", reply_markup=reply_markup
                )

            elif action == "close":
                await query.edit_message_text(
                    "🚨 Emergency controls closed. Use /emergency to reopen.",
                    parse_mode="Markdown",
                )

        except Exception as e:
            logger.error(f"Error handling emergency callback: {str(e)}")
            await query.edit_message_text(
                "❌ Error in emergency controls. Please try /emergency again.",
                parse_mode="Markdown",
            )

    def run(self):
        """Start the bot"""
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(
            CommandHandler("performance", self.performance_command)
        )
        self.application.add_handler(
            CommandHandler("test_signal", self.test_signal_command)
        )
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(
            CommandHandler("emergency", self.emergency_command)
        )
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

        # Start polling
        logger.info("Bot started")
        self.application.run_polling()
