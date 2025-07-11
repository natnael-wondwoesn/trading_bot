#!/usr/bin/env python3
"""
Multi-User Scalable Trading Bot
Handles thousands of users simultaneously with per-user isolation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
from dataclasses import dataclass
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError, Forbidden, BadRequest

from services.user_service import user_service, User, UserSession
from db.multi_user_db import multi_user_db
from models.models import Signal
import queue
import threading
from exchange_factory import ExchangeFactory
import time

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    user: User
    session: UserSession
    settings: Dict
    active_trades: List
    last_interaction: datetime
    notification_queue: asyncio.Queue
    is_rate_limited: bool = False


@dataclass
class SystemMessage:
    message_type: str  # 'signal', 'alert', 'notification', 'system'
    user_id: int
    content: Dict
    priority: int = 1  # 1=high, 2=medium, 3=low
    created_at: datetime = None


class MultiUserTradingBot:
    """Scalable bot architecture for handling thousands of users"""

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.application = None

        # User management
        self.active_users: Dict[int, UserContext] = {}  # telegram_id -> UserContext
        self.user_locks: Dict[int, asyncio.Lock] = (
            {}
        )  # telegram_id -> Lock for thread safety

        # Message queuing system
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.notification_workers = []
        self.notification_worker_count = 5

        # Performance monitoring
        self.stats = {
            "total_users": 0,
            "active_users_24h": 0,
            "messages_sent_today": 0,
            "commands_processed_today": 0,
            "errors_today": 0,
            "json_parse_errors_today": 0,
            "errors_network_today": 0,
            "errors_timeout_today": 0,
            "errors_bad_request_today": 0,
            "errors_service_init_today": 0,
            "errors_json_parse_today": 0,
            "errors_unknown_today": 0,
            "last_reset": datetime.now().date(),
        }

        # System health
        self.system_healthy = True
        self.maintenance_mode = False

        # Rate limiting
        self.rate_limits: Dict[int, Dict] = {}  # telegram_id -> rate limit data
        self._network_retry_count = 0  # New attribute for network retry count

    async def initialize(self):
        """Initialize the multi-user bot system"""
        logger.info("Initializing multi-user trading bot...")

        # Set running flag
        self.running = True

        # Initialize services
        await user_service.initialize()

        # Start background workers (but not telegram application yet)
        await self._start_background_workers()

        logger.info("Multi-user bot initialized successfully")

    async def _setup_handlers(self):
        """Setup command and callback handlers"""

        # Command handlers with user context
        handlers = [
            CommandHandler("start", self._handle_start),
            CommandHandler("help", self._handle_help),
            CommandHandler("settings", self._handle_settings),
            CommandHandler("emergency", self._handle_emergency),
            CommandHandler("status", self._handle_status),
            CommandHandler("dashboard", self._handle_dashboard),
            CommandHandler("performance", self._handle_performance),
            CommandHandler("subscription", self._handle_subscription),
            CommandHandler("support", self._handle_support),
            CallbackQueryHandler(self._handle_callback),
            MessageHandler(filters.COMMAND, self._handle_unknown_command),
        ]

        for handler in handlers:
            self.application.add_handler(handler)

        # Error handler
        self.application.add_error_handler(self._error_handler)

    async def _start_background_workers(self):
        """Start background workers for message processing"""

        # Start notification workers
        for i in range(self.notification_worker_count):
            worker = asyncio.create_task(self._notification_worker(f"worker-{i}"))
            self.notification_workers.append(worker)

        # Start maintenance tasks
        asyncio.create_task(self._maintenance_task())
        asyncio.create_task(self._stats_update_task())
        asyncio.create_task(self._health_check_task())

        logger.info(f"Started {self.notification_worker_count} notification workers")

    # User Context Management
    async def _get_user_context(
        self, telegram_id: int, update: Update = None
    ) -> Optional[UserContext]:
        """Get or create user context"""

        # Get user lock for thread safety
        if telegram_id not in self.user_locks:
            self.user_locks[telegram_id] = asyncio.Lock()

        async with self.user_locks[telegram_id]:
            # Check if user context exists and is valid
            if telegram_id in self.active_users:
                context = self.active_users[telegram_id]
                context.last_interaction = datetime.now()
                return context

            # Create new user context
            if update and update.effective_user:
                user, is_new = await user_service.register_or_login_user(
                    update.effective_user
                )
                session = await user_service.create_user_session(user)
                settings = await user_service.get_user_settings(user.user_id)

                # Safely convert settings to dict to avoid JSON parsing issues
                settings_dict = {}
                if settings:
                    try:
                        settings_dict = {
                            "strategy": getattr(settings, "strategy", "RSI_EMA"),
                            "exchange": getattr(settings, "exchange", "MEXC"),
                            "risk_management": getattr(settings, "risk_management", {}),
                            "notifications": getattr(settings, "notifications", {}),
                            "emergency": getattr(settings, "emergency", {}),
                        }
                    except Exception as e:
                        logger.warning(
                            f"Error converting settings to dict for user {user.user_id}: {e}"
                        )
                        settings_dict = {
                            "strategy": "RSI_EMA",
                            "exchange": "MEXC",
                            "risk_management": {},
                            "notifications": {},
                            "emergency": {},
                        }

                context = UserContext(
                    user=user,
                    session=session,
                    settings=settings_dict,
                    active_trades=[],
                    last_interaction=datetime.now(),
                    notification_queue=asyncio.Queue(maxsize=100),
                )

                self.active_users[telegram_id] = context

                if is_new:
                    await self._send_welcome_sequence(telegram_id, user)

                return context

        return None

    async def _send_welcome_sequence(self, telegram_id: int, user: User):
        """Send welcome sequence to new users"""
        try:
            welcome_messages = [
                "Welcome to Professional Trading Bot!",
                "Let me quickly show you around...",
                f"Your account: {user.subscription_tier.title()} subscription",
                "Tip: Use /help to see all available commands",
                "Ready to start trading? Type /dashboard to begin!",
            ]

            for i, message in enumerate(welcome_messages):
                await asyncio.sleep(0.5)  # Slight delay between messages
                await self.application.bot.send_message(
                    chat_id=telegram_id, text=message, parse_mode="Markdown"
                )

        except Exception as e:
            logger.warning(f"Failed to send welcome sequence to {telegram_id}: {e}")

    async def _cleanup_inactive_users(self):
        """Remove inactive user contexts to save memory"""
        cutoff_time = datetime.now() - timedelta(hours=2)
        inactive_users = [
            telegram_id
            for telegram_id, context in self.active_users.items()
            if context.last_interaction < cutoff_time
        ]

        for telegram_id in inactive_users:
            if telegram_id in self.active_users:
                del self.active_users[telegram_id]
            if telegram_id in self.user_locks:
                del self.user_locks[telegram_id]

        if inactive_users:
            logger.info(f"Cleaned up {len(inactive_users)} inactive user contexts")

    # Command Handlers
    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Check if maintenance mode
        if self.maintenance_mode:
            await update.message.reply_text(
                "🔧 **System Maintenance**\n\n"
                "The trading bot is currently under maintenance.\n"
                "Please try again later.",
                parse_mode="Markdown",
            )
            return

        # Get subscription info
        limits = user_service.get_subscription_limits(user_context.user)

        welcome_message = f"""🤖 **Welcome to Professional Trading Bot**

👤 **Your Account:**
• Subscription: {user_context.user.subscription_tier.title()}
• Daily Trades: {limits.daily_trades}
• Max Positions: {limits.concurrent_positions}
• Member Since: {user_context.user.created_at.strftime('%B %Y')}

📊 **Available Features:**
{self._format_features(limits.features)}

🎯 **Quick Commands:**
• /dashboard - Your trading overview
• /settings - Configure strategies & risk
• /emergency - Emergency controls
• /performance - View your results
• /subscription - Upgrade your plan

Ready to start smart trading! 🚀"""

        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        help_message = """🤖 **Trading Bot Commands Help**

**📊 Core Commands:**
• `/start` - Welcome & account overview
• `/dashboard` - Your trading performance
• `/settings` - Configure strategies & risk
• `/emergency` - Emergency stop/controls

**📈 Analysis & Info:**
• `/status` - Current system status
• `/performance` - Detailed trading results
• `/subscription` - View/upgrade your plan

**🛠 Support:**
• `/support` - Contact customer support
• `/help` - This help message

**💡 Quick Tips:**
• Use `/emergency` to instantly stop all trading
• Check `/dashboard` for real-time performance
• Adjust risk settings in `/settings`

Need more help? Use `/support` to reach our team! 🎯"""

        await update.message.reply_text(help_message, parse_mode="Markdown")

    async def _handle_emergency(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /emergency command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Emergency keyboard
        keyboard = [
            [
                InlineKeyboardButton(
                    "🛑 STOP ALL TRADING",
                    callback_data=f"emergency_stop_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔒 DISABLE BOT",
                    callback_data=f"emergency_disable_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "💰 CLOSE POSITIONS",
                    callback_data=f"emergency_close_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📊 System Status",
                    callback_data=f"emergency_status_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data=f"emergency_cancel_{user_context.user.user_id}",
                ),
            ],
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🚨 **EMERGENCY CONTROLS**\n\n"
            "⚠️ Choose your emergency action:\n\n"
            "• **STOP ALL TRADING** - Halt new trades, keep positions\n"
            "• **DISABLE BOT** - Complete shutdown for this user\n"
            "• **CLOSE POSITIONS** - Close all open positions\n\n"
            "**Use with caution!**",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Get system status
        stats = self.get_system_stats()

        status_message = f"""📊 **System Status**

🟢 **Service Health:**
• Bot Status: {"🟢 Online" if not self.maintenance_mode else "🔧 Maintenance"}
• Active Users: {stats.get('active_users', 0)}
• Messages Today: {stats.get('messages_sent_today', 0)}

⚡ **Performance:**
• Uptime: {stats.get('uptime_hours', 0):.1f} hours
• Commands/Hour: {stats.get('commands_per_hour', 0):.1f}
• Response Time: {stats.get('avg_response_time', 0):.2f}s

💼 **Your Status:**
• Subscription: {user_context.user.subscription_tier.title()}
• Trading: {"🟢 Active" if user_context.settings.get('trading_enabled', True) else "🔴 Disabled"}
• Last Activity: {user_context.last_interaction.strftime('%H:%M:%S')}

🔄 Updated: {datetime.now().strftime('%H:%M:%S')}"""

        await update.message.reply_text(status_message, parse_mode="Markdown")

    async def _handle_performance(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /performance command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        try:
            # Check if user_service is properly initialized and has the method
            if not hasattr(user_service, "get_user_performance"):
                logger.error("user_service doesn't have get_user_performance method")
                await update.message.reply_text(
                    "⚠️ **Performance Service Unavailable**\n\n"
                    "The performance tracking service is currently initializing. "
                    "Please try again in a few moments.",
                    parse_mode="Markdown",
                )
                return

            # Get user performance data with comprehensive error handling
            try:
                performance = await user_service.get_user_performance(
                    user_context.user.user_id
                )
            except AttributeError as ae:
                logger.error(f"AttributeError calling get_user_performance: {ae}")
                await update.message.reply_text(
                    "⚠️ **Service Temporarily Unavailable**\n\n"
                    "The performance service is starting up. Please try again in a moment.",
                    parse_mode="Markdown",
                )
                return
            except Exception as pe:
                logger.error(f"Error getting user performance: {pe}")
                # Return default empty performance
                performance = {
                    "total_pnl": 0.0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "daily_pnl": 0.0,
                }

            if not performance:
                await update.message.reply_text(
                    "📊 **No Performance Data**\n\n"
                    "You haven't completed any trades yet.\n"
                    "Start trading to see your performance metrics!",
                    parse_mode="Markdown",
                )
                return

            # Format and send performance data
            total_pnl = performance.get("total_pnl", 0.0)
            total_trades = performance.get("total_trades", 0)
            win_rate = performance.get("win_rate", 0.0) * 100
            daily_pnl = performance.get("daily_pnl", 0.0)

            pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
            daily_emoji = "📈" if daily_pnl >= 0 else "📉"

            performance_text = (
                f"📊 **Your Trading Performance**\n\n"
                f"{pnl_emoji} **Total P&L:** ${total_pnl:,.2f}\n"
                f"📈 **Total Trades:** {total_trades}\n"
                f"🎯 **Win Rate:** {win_rate:.1f}%\n"
                f"{daily_emoji} **Today's P&L:** ${daily_pnl:,.2f}\n\n"
                f"_Last updated: {datetime.now().strftime('%H:%M:%S')}_"
            )

            keyboard = [
                [
                    InlineKeyboardButton(
                        "📊 Detailed Report",
                        callback_data=f"performance_detailed_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "🔄 Refresh",
                        callback_data=f"performance_refresh_{user_context.user.user_id}",
                    ),
                ]
            ]

            await update.message.reply_text(
                performance_text,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        except Exception as e:
            logger.error(f"Error in performance command: {e}")
            await update.message.reply_text(
                "❌ **Error**\n\n"
                "Sorry, I couldn't retrieve your performance data right now. "
                "Please try again later.",
                parse_mode="Markdown",
            )

    async def _handle_support(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /support command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Support options keyboard
        keyboard = [
            [
                InlineKeyboardButton(
                    "❓ FAQ",
                    callback_data=f"support_faq_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "💬 Live Chat",
                    callback_data=f"support_chat_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📧 Email Support",
                    callback_data=f"support_email_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "🐛 Report Bug",
                    callback_data=f"support_bug_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📚 Documentation",
                    callback_data=f"support_docs_{user_context.user.user_id}",
                ),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        support_message = f"""🛠 **Customer Support**

👤 **Your Account:** {user_context.user.subscription_tier.title()} Subscriber
🆔 **User ID:** {user_context.user.user_id}

🤝 **How can we help you today?**

• **FAQ** - Common questions & answers
• **Live Chat** - Instant support (Premium+)
• **Email Support** - Detailed assistance
• **Report Bug** - Technical issues
• **Documentation** - Complete guides

⏰ **Support Hours:**
• Live Chat: 24/7 (Premium/Enterprise)
• Email: 24-48h response time

📞 **Enterprise Support:**
Direct phone support available for Enterprise subscribers."""

        await update.message.reply_text(
            support_message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _handle_unknown_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle unknown commands"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        command = update.message.text

        unknown_message = f"""❓ **Unknown Command**

I don't recognize the command: `{command}`

**Available Commands:**
• `/start` - Get started
• `/help` - View all commands
• `/dashboard` - Trading overview
• `/settings` - Configure bot
• `/emergency` - Emergency controls

Type `/help` for a complete list of commands! 🤖"""

        await update.message.reply_text(unknown_message, parse_mode="Markdown")

    async def _handle_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /settings command with user isolation"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Check rate limits
        if not await self._check_user_rate_limit(update.effective_user.id):
            await update.message.reply_text(
                "⏰ You're sending commands too quickly. Please wait a moment.",
                parse_mode="Markdown",
            )
            return

        # Get current settings
        current_exchange = user_context.settings.get("exchange", "MEXC")
        current_strategy = user_context.settings.get("strategy", "RSI_EMA")

        # Create personalized settings menu
        keyboard = [
            [
                InlineKeyboardButton(
                    f"🏦 Exchange: {current_exchange}",
                    callback_data=f"settings_exchange_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "🔧 Strategy",
                    callback_data=f"settings_strategy_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "💰 Risk Mgmt",
                    callback_data=f"settings_risk_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "📱 Notifications",
                    callback_data=f"settings_notifications_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🚨 Emergency",
                    callback_data=f"settings_emergency_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "📊 View All",
                    callback_data=f"settings_view_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔄 Reset",
                    callback_data=f"settings_reset_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "❌ Close",
                    callback_data=f"settings_close_{user_context.user.user_id}",
                ),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        settings_text = f"""⚙️ **TRADING SETTINGS**
        
👤 **User:** {user_context.user.first_name or user_context.user.username}
🎯 **Plan:** {user_context.user.subscription_tier.title()}

**Current Configuration:**
🏦 **Exchange:** {current_exchange}
🔧 **Strategy:** {current_strategy}
⚡ **Status:** {'🟢 Active' if user_context.settings.get('trading_enabled', True) else '🔴 Paused'}

**Configure your trading preferences:**"""

        await update.message.reply_text(
            settings_text, parse_mode="Markdown", reply_markup=reply_markup
        )

    async def _handle_dashboard(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /dashboard command with exchange-specific data"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Get user's selected exchange
        selected_exchange = user_context.settings.get("exchange", "MEXC")

        # Send loading message
        loading_msg = await update.message.reply_text(
            f"📊 Loading {selected_exchange} dashboard...", parse_mode="Markdown"
        )

        try:
            # Get account data from the selected exchange
            account_data = await self._get_exchange_account_data(
                user_context.user.user_id, selected_exchange
            )

            if not account_data:
                await loading_msg.edit_text(
                    f"❌ Unable to load {selected_exchange} account data.\n\n"
                    "Please check your API configuration in settings."
                )
                return

            # Format comprehensive dashboard
            dashboard_text = await self._format_dashboard_message(
                user_context, selected_exchange, account_data
            )

            # Create action buttons
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔄 Refresh",
                        callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "📈 Positions",
                        callback_data=f"dashboard_positions_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "💼 Portfolio",
                        callback_data=f"dashboard_portfolio_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "📊 History",
                        callback_data=f"dashboard_history_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🏦 Switch Exchange",
                        callback_data=f"settings_exchange_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "⚙️ Settings",
                        callback_data=f"settings_back_{user_context.user.user_id}",
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await loading_msg.edit_text(
                dashboard_text, parse_mode="Markdown", reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error loading dashboard: {e}")
            await loading_msg.edit_text(
                f"❌ Error loading {selected_exchange} dashboard.\n\n"
                f"Please try again or check your settings."
            )

    async def _get_exchange_account_data(self, user_id: int, exchange: str) -> Dict:
        """Get account data from the specified exchange"""
        try:
            if exchange == "MEXC":
                return await self._get_mexc_account_data(user_id)
            elif exchange == "Bybit":
                return await self._get_bybit_account_data(user_id)
            else:
                logger.error(f"Unsupported exchange: {exchange}")
                return None
        except Exception as e:
            logger.error(f"Error getting {exchange} account data: {e}")
            return None

    async def _get_mexc_account_data(self, user_id: int) -> Dict:
        """Get MEXC account data"""
        try:
            # Import MEXC client
            from mexc.mexc_client import MEXCClient
            from config.config import Config

            # Create MEXC client (you might want to get user-specific API keys)
            mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)

            # Get account info
            account_info = await mexc_client.get_account()

            # Get balance data
            balance_data = await mexc_client.get_balance()

            # Get open orders
            open_orders = await mexc_client.get_open_orders()

            # Calculate total balance value
            total_balance = 0
            balances = []

            for asset, balance in balance_data.items():
                total_free = balance.get("free", 0)
                total_locked = balance.get("locked", 0)
                total_asset = total_free + total_locked

                if total_asset > 0:
                    balances.append(
                        {
                            "asset": asset,
                            "free": total_free,
                            "locked": total_locked,
                            "total": total_asset,
                        }
                    )

                    # For simplicity, count USDT as 1:1, others would need price conversion
                    if asset in ["USDT", "USDC", "BUSD"]:
                        total_balance += total_asset

            await mexc_client.close()

            return {
                "exchange": "MEXC",
                "account_info": account_info,
                "balances": balances,
                "total_balance_usdt": total_balance,
                "open_orders": len(open_orders),
                "api_status": "connected",
            }

        except Exception as e:
            logger.error(f"Error getting MEXC account data: {e}")
            return {"exchange": "MEXC", "api_status": "error", "error_message": str(e)}

    async def _get_bybit_account_data(self, user_id: int) -> Dict:
        """Get Bybit account data with support for both Unified and Classic account modes"""
        try:
            # Import Bybit client
            from bybit.bybit_client import BybitClient
            from config.config import Config

            # Create Bybit client (you might want to get user-specific API keys)
            async with BybitClient(
                Config.BYBIT_API_KEY, Config.BYBIT_API_SECRET
            ) as bybit_client:

                # Get account info to determine account mode
                account_info = await bybit_client.get_account_info()
                unified_status = account_info.get("result", {}).get(
                    "unifiedMarginStatus", 0
                )

                logger.info(f"Bybit unified margin status: {unified_status}")

                # Get open orders (this returns a list directly)
                open_orders_list = await bybit_client.get_open_orders()

                # Process balance data
                total_balance = 0
                balances = []

                # Method 1: Try UNIFIED account wallet (works if account is in Unified mode)
                if unified_status == 1:  # Account is in Unified Trading mode
                    logger.info(
                        "Account is in Unified Trading mode, checking unified wallet..."
                    )
                    try:
                        wallet_response = await bybit_client.get_wallet_balance(
                            account_type="UNIFIED"
                        )

                        if isinstance(wallet_response, dict):
                            wallet_list = wallet_response.get("result", {}).get(
                                "list", []
                            )
                            if wallet_list and len(wallet_list) > 0:
                                first_wallet = wallet_list[0]
                                if isinstance(first_wallet, dict):
                                    coins = first_wallet.get("coin", [])
                                    logger.info(
                                        f"Found {len(coins)} coins in unified wallet"
                                    )

                                    for coin_data in coins:
                                        if isinstance(coin_data, dict):
                                            wallet_balance = float(
                                                coin_data.get("walletBalance", "0")
                                            )
                                            locked_balance = float(
                                                coin_data.get("locked", "0")
                                            )
                                            available_balance = (
                                                wallet_balance - locked_balance
                                            )
                                            coin_symbol = coin_data.get("coin", "")

                                            if wallet_balance > 0:
                                                balances.append(
                                                    {
                                                        "asset": coin_symbol,
                                                        "free": available_balance,
                                                        "locked": locked_balance,
                                                        "total": wallet_balance,
                                                    }
                                                )

                                                # Count USDT value
                                                if coin_symbol in [
                                                    "USDT",
                                                    "USDC",
                                                    "BUSD",
                                                ]:
                                                    total_balance += wallet_balance

                    except Exception as unified_error:
                        logger.warning(f"Unified wallet access failed: {unified_error}")

                        # Method 2: Try Classic Account mode - check individual coins
                if not balances:  # If no balances found in unified account
                    logger.info(
                        "Trying Classic Account mode - checking individual coin balances..."
                    )

                    try:
                        # Check common coins individually
                        common_coins = [
                            "USDT",
                            "USDC",
                            "BUSD",
                            "BTC",
                            "ETH",
                            "BNB",
                            "SOL",
                            "ADA",
                            "DOGE",
                        ]

                        for coin in common_coins:
                            try:
                                # Check individual coin balance
                                coin_response = await bybit_client.get_wallet_balance(
                                    account_type="UNIFIED", coin=coin
                                )

                                if coin_response.get("result", {}).get("list"):
                                    wallets = coin_response["result"]["list"]
                                    for wallet in wallets:
                                        coins_data = wallet.get("coin", [])
                                        for coin_data in coins_data:
                                            if coin_data.get("coin") == coin:
                                                wallet_balance = float(
                                                    coin_data.get("walletBalance", "0")
                                                )
                                                if wallet_balance > 0:
                                                    locked_balance = float(
                                                        coin_data.get("locked", "0")
                                                    )
                                                    available_balance = (
                                                        wallet_balance - locked_balance
                                                    )

                                                    balances.append(
                                                        {
                                                            "asset": coin,
                                                            "free": available_balance,
                                                            "locked": locked_balance,
                                                            "total": wallet_balance,
                                                        }
                                                    )

                                                    if coin in ["USDT", "USDC", "BUSD"]:
                                                        total_balance += wallet_balance

                                                    logger.info(
                                                        f"Found {coin}: {wallet_balance}"
                                                    )

                            except Exception as coin_error:
                                # Skip coins that don't exist or have errors
                                continue

                    except Exception as classic_error:
                        logger.warning(
                            f"Classic account balance check failed: {classic_error}"
                        )

                # Method 3: Try the get_balance compatibility method as final fallback
                if not balances:
                    logger.info("Trying compatibility get_balance method...")
                    try:
                        balance_response = await bybit_client.get_balance()
                        if isinstance(balance_response, dict):
                            for asset, balance_info in balance_response.items():
                                if isinstance(balance_info, dict):
                                    free_balance = balance_info.get("free", 0)
                                    locked_balance = balance_info.get("locked", 0)
                                    total_asset_balance = free_balance + locked_balance

                                    if total_asset_balance > 0:
                                        balances.append(
                                            {
                                                "asset": asset,
                                                "free": free_balance,
                                                "locked": locked_balance,
                                                "total": total_asset_balance,
                                            }
                                        )

                                        if asset in ["USDT", "USDC", "BUSD"]:
                                            total_balance += total_asset_balance

                    except Exception as compat_error:
                        logger.warning(
                            f"Compatibility balance method failed: {compat_error}"
                        )

                # Log results
                logger.info(
                    f"Bybit balance check complete: {len(balances)} assets, ${total_balance} USDT total"
                )
                for balance in balances:
                    logger.info(
                        f"  {balance['asset']}: {balance['total']} (free: {balance['free']}, locked: {balance['locked']})"
                    )

                return {
                    "exchange": "Bybit",
                    "account_info": account_info,
                    "balances": balances,
                    "total_balance_usdt": total_balance,
                    "open_orders": (
                        len(open_orders_list)
                        if isinstance(open_orders_list, list)
                        else 0
                    ),
                    "api_status": "connected",
                    "account_mode": "unified" if unified_status == 1 else "classic",
                    "unified_status": unified_status,
                }

        except Exception as e:
            logger.error(f"Error getting Bybit account data: {e}")
            return {"exchange": "Bybit", "api_status": "error", "error_message": str(e)}

    async def _format_dashboard_message(
        self, user_context: UserContext, exchange: str, account_data: Dict
    ) -> str:
        """Format the dashboard message"""
        try:
            if account_data.get("api_status") == "error":
                return f"""📊 **{exchange.upper()} DASHBOARD**

❌ **API Connection Error**

{account_data.get('error_message', 'Unknown error')}

**Troubleshooting:**
• Check API key configuration
• Verify API permissions
• Ensure API keys are valid
• Check network connectivity

Use ⚙️ Settings to configure your API keys."""

            # Get user dashboard data
            dashboard_data = await user_service.get_user_dashboard_data(
                user_context.user.user_id
            )

            balances = account_data.get("balances", [])
            total_balance = account_data.get("total_balance_usdt", 0)
            open_orders = account_data.get("open_orders", 0)

            # Special handling for Bybit Classic account mode
            account_mode_info = ""
            if exchange == "Bybit":
                account_mode = account_data.get("account_mode", "unknown")
                unified_status = account_data.get("unified_status", 0)

                if account_mode == "classic":
                    account_mode_info = f"""
🔔 **Account Mode:** Classic (Status: {unified_status})
⚠️ **Note:** If you have balance but it shows $0, it might be in Spot Trading wallet (separate from Unified Account). Check your Bybit account settings or contact support."""

            dashboard_text = f"""📊 **{exchange.upper()} DASHBOARD**

👤 **Account Overview**
🏦 Exchange: {exchange}
💰 Total Balance: ${total_balance:,.2f} USDT
📈 Open Orders: {open_orders}
⚡ Status: {'🟢 Connected' if account_data.get('api_status') == 'connected' else '🔴 Disconnected'}{account_mode_info}

💼 **Portfolio Breakdown**"""

            # Show top balances
            if balances:
                sorted_balances = sorted(
                    balances, key=lambda x: x["total"], reverse=True
                )
                for balance in sorted_balances[:5]:  # Show top 5 assets
                    asset = balance["asset"]
                    total = balance["total"]
                    free = balance["free"]
                    locked = balance["locked"]

                    dashboard_text += f"""
• **{asset}**: {total:,.4f}
  └ Available: {free:,.4f} | Locked: {locked:,.4f}"""

                if len(balances) > 5:
                    dashboard_text += f"\n  └ ... and {len(balances) - 5} more assets"
            else:
                if (
                    exchange == "Bybit"
                    and account_data.get("account_mode") == "classic"
                ):
                    dashboard_text += """
• No assets found in Unified Account

🔍 **Troubleshooting:**
• Your balance might be in Classic Spot Trading
• Check API permissions (needs 'Read' access)
• Verify you're using the correct API keys
• Consider switching to Unified Trading mode in Bybit"""
                else:
                    dashboard_text += "\n• No assets found"

            # Add trading performance from database
            if dashboard_data:
                dashboard_text += f"""

📈 **Trading Performance**
• Today's Trades: {dashboard_data['daily_stats']['total_trades']}
• Win Rate: {dashboard_data['daily_stats']['win_rate']:.1f}%
• Today's P&L: ${dashboard_data['daily_stats']['total_pnl']:.2f}
• This Week: {dashboard_data['performance_summary']['total_trades_7d']} trades

📊 **Account Limits**
• Plan: {dashboard_data['user_info']['subscription_tier'].title()}
• Daily Usage: {dashboard_data['usage']['daily_trades_used']:.1f}%
• Position Usage: {dashboard_data['usage']['positions_used']:.1f}%"""

            dashboard_text += f"""

🕐 **Last Updated:** {datetime.now().strftime('%H:%M:%S')}
🔄 Use "Refresh" to update data"""

            return dashboard_text

        except Exception as e:
            logger.error(f"Error formatting dashboard: {e}")
            return f"""📊 **{exchange.upper()} DASHBOARD**

❌ **Error formatting dashboard data**

Please try refreshing or contact support if the issue persists."""

    async def _handle_subscription(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /subscription command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        current_limits = user_service.get_subscription_limits(user_context.user)

        subscription_text = f"""💎 **SUBSCRIPTION PLANS**

🆔 **Current Plan: {user_context.user.subscription_tier.title()}**

📊 **Plan Comparison:**

🆓 **Free Plan**
• Daily Trades: 5
• Max Positions: 2
• Basic Signals & Stats
• Community Support

💰 **Premium Plan** - $29/month
• Daily Trades: 25
• Max Positions: 5
• Advanced Signals & Analytics
• Custom Alerts
• Priority Support

🏢 **Enterprise Plan** - $99/month
• Daily Trades: 100
• Max Positions: 20
• All Premium Features
• Custom Strategies
• API Access
• Dedicated Support

Ready to upgrade your trading experience?"""

        keyboard = []
        if user_context.user.subscription_tier == "free":
            keyboard.extend(
                [
                    [
                        InlineKeyboardButton(
                            "⬆️ Upgrade to Premium",
                            callback_data=f"upgrade_premium_{user_context.user.user_id}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "🚀 Upgrade to Enterprise",
                            callback_data=f"upgrade_enterprise_{user_context.user.user_id}",
                        )
                    ],
                ]
            )
        elif user_context.user.subscription_tier == "premium":
            keyboard.append(
                [
                    InlineKeyboardButton(
                        "🚀 Upgrade to Enterprise",
                        callback_data=f"upgrade_enterprise_{user_context.user.user_id}",
                    )
                ]
            )

        keyboard.append(
            [
                InlineKeyboardButton(
                    "💳 Billing Info",
                    callback_data=f"billing_info_{user_context.user.user_id}",
                )
            ]
        )
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            subscription_text, parse_mode="Markdown", reply_markup=reply_markup
        )

    # Callback Handler
    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle button callbacks with user isolation"""
        query = update.callback_query
        await query.answer()

        user_context = await self._get_user_context(query.from_user.id)
        if not user_context:
            await query.edit_message_text(
                "Session expired. Please use /start to begin."
            )
            return

        data_parts = query.data.split("_")
        if len(data_parts) < 3:
            await query.edit_message_text("Invalid command.")
            return

        action_type = data_parts[0]
        action = data_parts[1]

        # Handle different callback data formats with safe integer parsing
        try:
            if action_type == "strategy" and action == "select":
                # Format: strategy_select_STRATEGY_NAME_user_id
                if len(data_parts) < 4:
                    await query.edit_message_text("Invalid strategy selection.")
                    return
                user_id = int(data_parts[3])
            elif action_type == "exchange" and action == "select":
                # Format: exchange_select_EXCHANGE_NAME_user_id
                if len(data_parts) < 4:
                    await query.edit_message_text("Invalid exchange selection.")
                    return
                user_id = int(data_parts[3])
            else:
                # Standard format: action_type_action_user_id
                user_id = int(data_parts[2])
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing callback data '{query.data}': {e}")
            await query.edit_message_text("Invalid request format. Please try again.")
            return

        # Verify user ID matches
        if user_id != user_context.user.user_id:
            await query.edit_message_text("Unauthorized action.")
            return

        # Route to appropriate handler
        if action_type == "settings":
            await self._handle_settings_callback(query, action, user_context)
        elif action_type == "dashboard":
            await self._handle_dashboard_callback(query, action, user_context)
        elif action_type == "exchange" and action == "select":
            # Handle exchange selection
            if len(data_parts) >= 3:
                exchange_name = data_parts[2]  # exchange_select_MEXC_user_id
                try:
                    # Update user exchange in database
                    await user_service.update_user_settings(
                        user_context.user.user_id, exchange=exchange_name
                    )

                    # Update local settings
                    user_context.settings["exchange"] = exchange_name

                    # Notify trading orchestrator of the change
                    try:
                        from services.trading_orchestrator import trading_orchestrator

                        await trading_orchestrator.update_user_settings(
                            user_context.user.user_id
                        )
                    except Exception as orchestrator_error:
                        logger.warning(
                            f"Could not notify trading orchestrator: {orchestrator_error}"
                        )

                    success_message = f"""✅ **Exchange Updated Successfully!**

🏦 **New Exchange:** {exchange_name}

**What changed:**
• All trading operations will use {exchange_name}
• Dashboard will show {exchange_name} account info
• Balance and portfolio data from {exchange_name}
• Order management through {exchange_name} API

Your trading settings have been updated. Use `/dashboard` to view your {exchange_name} account information.

**Next Steps:**
• Check your `/dashboard` for account details
• Verify your API keys are configured
• Review your trading settings"""

                    keyboard = [
                        [
                            InlineKeyboardButton(
                                "📊 View Dashboard",
                                callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                            ),
                            InlineKeyboardButton(
                                "⚙️ Settings",
                                callback_data=f"settings_back_{user_context.user.user_id}",
                            ),
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(
                        success_message,
                        reply_markup=reply_markup,
                        parse_mode="Markdown",
                    )

                except Exception as e:
                    logger.error(
                        f"Error updating exchange for user {user_context.user.user_id}: {e}"
                    )
                    await query.edit_message_text(
                        "❌ Failed to update exchange. Please try again."
                    )
        elif action_type == "strategy" and action == "select":
            # Handle strategy selection
            if len(data_parts) >= 4:
                strategy_name = data_parts[2]  # strategy_select_RSI_EMA_user_id
                try:
                    # Update user strategy in database
                    await user_service.update_user_settings(
                        user_context.user.user_id, strategy=strategy_name
                    )

                    # Update local settings
                    user_context.settings["strategy"] = strategy_name

                    # Notify trading orchestrator of the change
                    try:
                        from services.trading_orchestrator import trading_orchestrator

                        await trading_orchestrator.update_user_settings(
                            user_context.user.user_id
                        )
                    except Exception as orchestrator_error:
                        logger.warning(
                            f"Could not notify trading orchestrator: {orchestrator_error}"
                        )

                    await query.edit_message_text(
                        f"✅ **Strategy Updated Successfully!**\n\n"
                        f"**New Strategy:** {strategy_name}\n\n"
                        f"Your new strategy will be used for all future signals.\n\n"
                        f"**Note:** This change takes effect immediately for new trades.",
                        parse_mode="Markdown",
                    )

                except Exception as e:
                    logger.error(
                        f"Error updating strategy for user {user_context.user.user_id}: {e}"
                    )
                    await query.edit_message_text(
                        "❌ Failed to update strategy. Please try again."
                    )
        else:
            await query.edit_message_text("Unknown action.")

    async def _handle_settings_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle settings callback actions"""
        try:
            if action == "exchange":
                # Exchange selection interface
                current_exchange = user_context.settings.get("exchange", "MEXC")

                keyboard = [
                    [
                        InlineKeyboardButton(
                            f"{'[✓] MEXC' if current_exchange == 'MEXC' else 'MEXC'}",
                            callback_data=f"exchange_select_MEXC_{user_context.user.user_id}",
                        ),
                        InlineKeyboardButton(
                            f"{'[✓] Bybit' if current_exchange == 'Bybit' else 'Bybit'}",
                            callback_data=f"exchange_select_Bybit_{user_context.user.user_id}",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "← Back to Settings",
                            callback_data=f"settings_back_{user_context.user.user_id}",
                        )
                    ],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                exchange_text = f"""🏦 **EXCHANGE SELECTION**

**Current Exchange:** {current_exchange}

**Available Exchanges:**

📊 **MEXC**
• Spot Trading ✅
• Low Fees ✅
• Wide Selection ✅
• API v3 Support ✅

📊 **Bybit**  
• Spot Trading ✅
• Advanced Features ✅
• Professional Tools ✅
• API v5 Support ✅

**Select your preferred exchange:**

⚠️ **Note:** Changing exchange will affect:
• Balance display
• Trading operations
• Order management
• Portfolio tracking"""

                await query.edit_message_text(
                    exchange_text, reply_markup=reply_markup, parse_mode="Markdown"
                )

            elif action == "strategy":
                # Import here to avoid circular imports
                from services.trading_orchestrator import StrategyFactory

                # Get available strategies
                strategies = StrategyFactory.get_available_strategies()
                current_strategy = user_context.settings.get(
                    "trading_strategy", "RSI_EMA"
                )

                # Create strategy selection keyboard
                keyboard = []
                for strategy_name, description in strategies.items():
                    # Add checkmark if current strategy
                    display_name = (
                        f"[✓] {strategy_name}"
                        if strategy_name == current_strategy
                        else strategy_name
                    )
                    keyboard.append(
                        [
                            InlineKeyboardButton(
                                display_name,
                                callback_data=f"strategy_select_{strategy_name}_{user_context.user.user_id}",
                            )
                        ]
                    )

                # Add back button
                keyboard.append(
                    [
                        InlineKeyboardButton(
                            "← Back to Settings",
                            callback_data=f"settings_back_{user_context.user.user_id}",
                        )
                    ]
                )

                reply_markup = InlineKeyboardMarkup(keyboard)

                strategy_text = "**🔧 Strategy Selection**\n\n"
                strategy_text += f"**Current Strategy:** {current_strategy}\n\n"
                strategy_text += "**Available Strategies:**\n"

                for strategy_name, description in strategies.items():
                    icon = "◉" if strategy_name == current_strategy else "○"
                    strategy_text += f"{icon} **{strategy_name}**\n"
                    strategy_text += f"   {description}\n\n"

                strategy_text += (
                    "Select a strategy to change your signal detection method:"
                )

                await query.edit_message_text(
                    strategy_text, reply_markup=reply_markup, parse_mode="Markdown"
                )
            elif action == "risk":
                # Risk management settings
                risk_settings = user_context.settings.get("risk_management", {})

                keyboard = [
                    [
                        InlineKeyboardButton(
                            "🔧 Configure Risk",
                            callback_data=f"risk_configure_{user_context.user.user_id}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "← Back to Settings",
                            callback_data=f"settings_back_{user_context.user.user_id}",
                        )
                    ],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Format values safely to avoid Markdown issues
                max_risk = risk_settings.get("max_risk_per_trade", 0.02) * 100
                trading_enabled_icon = (
                    "✅" if risk_settings.get("trading_enabled", True) else "❌"
                )

                # Determine risk level safely
                risk_value = risk_settings.get("max_risk_per_trade", 0.02)
                if risk_value <= 0.02:
                    risk_level = "🟢 Conservative"
                elif risk_value <= 0.05:
                    risk_level = "🟡 Moderate"
                else:
                    risk_level = "🔴 Aggressive"

                risk_text = f"""💰 **RISK MANAGEMENT**

**Current Settings:**
• Max Risk per Trade: {max_risk:.1f}%
• Stop Loss (ATR): {risk_settings.get('stop_loss_atr', 2.0)}x
• Take Profit (ATR): {risk_settings.get('take_profit_atr', 3.0)}x
• Max Open Positions: {risk_settings.get('max_open_positions', 5)}
• Trading Enabled: {trading_enabled_icon}

**Risk Level:** {risk_level}

Configure your risk parameters to match your trading style."""

                await query.edit_message_text(
                    risk_text, reply_markup=reply_markup, parse_mode="Markdown"
                )

            elif action == "notifications":
                # Notification settings
                notifications = user_context.settings.get("notifications", {})

                keyboard = [
                    [
                        InlineKeyboardButton(
                            "🔧 Configure Alerts",
                            callback_data=f"notif_configure_{user_context.user.user_id}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "← Back to Settings",
                            callback_data=f"settings_back_{user_context.user.user_id}",
                        )
                    ],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Format notification status safely
                signal_alerts = (
                    "✅" if notifications.get("signal_alerts", True) else "❌"
                )
                trade_execution = (
                    "✅" if notifications.get("trade_execution", True) else "❌"
                )
                risk_warnings = (
                    "✅" if notifications.get("risk_warnings", True) else "❌"
                )

                notif_text = f"""📱 **NOTIFICATION SETTINGS**

**Current Settings:**
• Signal Alerts: {signal_alerts}
• Trade Execution: {trade_execution}
• Risk Warnings: {risk_warnings}

Stay informed about your trading activity with customizable notifications."""

                await query.edit_message_text(
                    notif_text, reply_markup=reply_markup, parse_mode="Markdown"
                )

            elif action == "view":
                # View all settings
                settings = user_context.settings
                risk_mgmt = settings.get("risk_management", {})
                notifications = settings.get("notifications", {})
                emergency = settings.get("emergency", {})

                keyboard = [
                    [
                        InlineKeyboardButton(
                            "← Back to Settings",
                            callback_data=f"settings_back_{user_context.user.user_id}",
                        )
                    ],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                # Format all values safely to avoid Markdown issues
                max_risk_percent = risk_mgmt.get("max_risk_per_trade", 0.02) * 100
                trading_status = (
                    "✅ Enabled"
                    if risk_mgmt.get("trading_enabled", True)
                    else "❌ Disabled"
                )
                signal_alerts_status = (
                    "✅" if notifications.get("signal_alerts", True) else "❌"
                )
                trade_updates_status = (
                    "✅" if notifications.get("trade_execution", True) else "❌"
                )
                risk_warnings_status = (
                    "✅" if notifications.get("risk_warnings", True) else "❌"
                )
                emergency_mode_status = (
                    "✅ Active"
                    if emergency.get("emergency_mode", False)
                    else "❌ Inactive"
                )
                auto_close_status = (
                    "✅" if emergency.get("auto_close_on_loss", False) else "❌"
                )
                max_daily_loss_percent = emergency.get("max_daily_loss", 0.05) * 100

                view_text = f"""📊 **ALL SETTINGS OVERVIEW**

**Exchange & Strategy:**
🏦 Exchange: {settings.get('exchange', 'MEXC')}
🔧 Strategy: {settings.get('strategy', 'RSI_EMA')}

**Risk Management:**
💰 Max Risk: {max_risk_percent:.1f}%
🛑 Stop Loss: {risk_mgmt.get('stop_loss_atr', 2.0)}x ATR
🎯 Take Profit: {risk_mgmt.get('take_profit_atr', 3.0)}x ATR
📊 Max Positions: {risk_mgmt.get('max_open_positions', 5)}
⚡ Trading: {trading_status}

**Notifications:**
🔔 Signal Alerts: {signal_alerts_status}
📈 Trade Updates: {trade_updates_status}
⚠️ Risk Warnings: {risk_warnings_status}

**Emergency:**
🚨 Emergency Mode: {emergency_mode_status}
🔒 Auto Close: {auto_close_status}
📉 Max Daily Loss: {max_daily_loss_percent:.1f}%"""

                await query.edit_message_text(
                    view_text, reply_markup=reply_markup, parse_mode="Markdown"
                )

            elif action == "back":
                # Return to main settings menu
                await self._show_main_settings_menu(query, user_context)
            elif action == "close":
                await query.edit_message_text("⚙️ **Settings closed.**")
            else:
                await query.edit_message_text("Settings action not implemented yet.")
        except Exception as e:
            logger.error(f"Error in settings callback: {e}")
            await query.edit_message_text("An error occurred. Please try again.")

    async def _handle_dashboard_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle dashboard callback actions"""
        try:
            if action == "refresh":
                # Get user's selected exchange and refresh data
                selected_exchange = user_context.settings.get("exchange", "MEXC")

                # Show loading
                await query.edit_message_text(
                    f"🔄 Refreshing {selected_exchange} dashboard...",
                    parse_mode="Markdown",
                )

                try:
                    # Get fresh account data
                    account_data = await self._get_exchange_account_data(
                        user_context.user.user_id, selected_exchange
                    )

                    if account_data:
                        # Format updated dashboard
                        dashboard_text = await self._format_dashboard_message(
                            user_context, selected_exchange, account_data
                        )

                        # Recreate action buttons
                        keyboard = [
                            [
                                InlineKeyboardButton(
                                    "🔄 Refresh",
                                    callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                                ),
                                InlineKeyboardButton(
                                    "📈 Positions",
                                    callback_data=f"dashboard_positions_{user_context.user.user_id}",
                                ),
                            ],
                            [
                                InlineKeyboardButton(
                                    "💼 Portfolio",
                                    callback_data=f"dashboard_portfolio_{user_context.user.user_id}",
                                ),
                                InlineKeyboardButton(
                                    "📊 History",
                                    callback_data=f"dashboard_history_{user_context.user.user_id}",
                                ),
                            ],
                            [
                                InlineKeyboardButton(
                                    "🏦 Switch Exchange",
                                    callback_data=f"settings_exchange_{user_context.user.user_id}",
                                ),
                                InlineKeyboardButton(
                                    "⚙️ Settings",
                                    callback_data=f"settings_back_{user_context.user.user_id}",
                                ),
                            ],
                        ]
                        reply_markup = InlineKeyboardMarkup(keyboard)

                        await query.edit_message_text(
                            dashboard_text,
                            reply_markup=reply_markup,
                            parse_mode="Markdown",
                        )
                    else:
                        await query.edit_message_text(
                            f"❌ Failed to refresh {selected_exchange} data.\n\n"
                            "Please check your API configuration.",
                            parse_mode="Markdown",
                        )

                except Exception as refresh_error:
                    logger.error(f"Error refreshing dashboard: {refresh_error}")
                    await query.edit_message_text(
                        "❌ **Refresh Failed**\n\n"
                        "Unable to refresh dashboard data. Please try again.",
                        parse_mode="Markdown",
                    )

            elif action == "positions":
                # Show open positions
                selected_exchange = user_context.settings.get("exchange", "MEXC")

                try:
                    account_data = await self._get_exchange_account_data(
                        user_context.user.user_id, selected_exchange
                    )

                    positions_text = f"""📈 **OPEN POSITIONS** ({selected_exchange})

"""

                    open_orders = account_data.get("open_orders", 0)
                    if open_orders == 0:
                        positions_text += "📊 No open positions at the moment.\n\n"
                        positions_text += "Start trading to see your positions here."
                    else:
                        positions_text += f"📊 **{open_orders} Active Orders**\n\n"
                        positions_text += "Use your exchange platform to view detailed position information."

                    keyboard = [
                        [
                            InlineKeyboardButton(
                                "← Back to Dashboard",
                                callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                            )
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(
                        positions_text, reply_markup=reply_markup, parse_mode="Markdown"
                    )

                except Exception as e:
                    logger.error(f"Error loading positions: {e}")
                    await query.edit_message_text(
                        "❌ Error loading positions data.", parse_mode="Markdown"
                    )

            elif action == "portfolio":
                # Show portfolio breakdown
                selected_exchange = user_context.settings.get("exchange", "MEXC")

                try:
                    account_data = await self._get_exchange_account_data(
                        user_context.user.user_id, selected_exchange
                    )

                    balances = account_data.get("balances", [])
                    total_balance = account_data.get("total_balance_usdt", 0)

                    portfolio_text = f"""💼 **PORTFOLIO DETAILS** ({selected_exchange})

💰 **Total Portfolio Value:** ${total_balance:,.2f} USDT

📊 **Asset Breakdown:**
"""

                    if balances:
                        sorted_balances = sorted(
                            balances, key=lambda x: x["total"], reverse=True
                        )
                        for i, balance in enumerate(
                            sorted_balances[:10], 1
                        ):  # Show top 10
                            asset = balance["asset"]
                            total = balance["total"]
                            free = balance["free"]
                            locked = balance["locked"]

                            # Calculate percentage of total portfolio
                            if total_balance > 0:
                                percentage = (
                                    (total / total_balance * 100)
                                    if asset in ["USDT", "USDC", "BUSD"]
                                    else 0
                                )
                            else:
                                percentage = 0

                            portfolio_text += f"""
{i}. **{asset}**: {total:,.6f}
   💰 Value: {percentage:.1f}% of portfolio
   🟢 Available: {free:,.6f}
   🔒 Locked: {locked:,.6f}"""

                        if len(balances) > 10:
                            portfolio_text += (
                                f"\n\n... and {len(balances) - 10} more assets"
                            )
                    else:
                        portfolio_text += "\n📊 No assets found in portfolio."

                    keyboard = [
                        [
                            InlineKeyboardButton(
                                "← Back to Dashboard",
                                callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                            )
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(
                        portfolio_text, reply_markup=reply_markup, parse_mode="Markdown"
                    )

                except Exception as e:
                    logger.error(f"Error loading portfolio: {e}")
                    await query.edit_message_text(
                        "❌ Error loading portfolio data.", parse_mode="Markdown"
                    )

            elif action == "history":
                # Show trading history from database
                try:
                    performance = await user_service.get_user_performance(
                        user_context.user.user_id, days=7
                    )

                    history_text = f"""📊 **TRADING HISTORY** (Last 7 Days)

📈 **Performance Summary:**
• Total P&L: ${performance.get('total_pnl', 0):.2f}
• Total Trades: {performance.get('total_trades', 0)}
• Win Rate: {performance.get('win_rate', 0):.1f}%
• Best Trade: ${performance.get('best_trade', 0):.2f}
• Worst Trade: ${performance.get('worst_trade', 0):.2f}

📊 **Recent Activity:**
• Daily P&L: ${performance.get('daily_pnl', 0):.2f}
• Weekly P&L: ${performance.get('weekly_pnl', 0):.2f}
• Monthly P&L: ${performance.get('monthly_pnl', 0):.2f}

📈 **Statistics:**
• Avg Trade Size: ${performance.get('avg_trade_size', 0):.2f}
• Winning Trades: {performance.get('winning_trades', 0)}
• Losing Trades: {performance.get('losing_trades', 0)}"""

                    keyboard = [
                        [
                            InlineKeyboardButton(
                                "← Back to Dashboard",
                                callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                            )
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(
                        history_text, reply_markup=reply_markup, parse_mode="Markdown"
                    )

                except Exception as e:
                    logger.error(f"Error loading history: {e}")
                    await query.edit_message_text(
                        "❌ Error loading trading history.", parse_mode="Markdown"
                    )

        except Exception as e:
            logger.error(f"Error in dashboard callback: {e}")
            await query.answer("Error processing request")

    async def _show_main_settings_menu(self, query, user_context: UserContext):
        """Show the main settings menu"""
        try:
            # Get current settings
            current_exchange = user_context.settings.get("exchange", "MEXC")
            current_strategy = user_context.settings.get("strategy", "RSI_EMA")

            # Create main settings menu
            keyboard = [
                [
                    InlineKeyboardButton(
                        f"🏦 Exchange: {current_exchange}",
                        callback_data=f"settings_exchange_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "🔧 Strategy",
                        callback_data=f"settings_strategy_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "💰 Risk Mgmt",
                        callback_data=f"settings_risk_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "📱 Notifications",
                        callback_data=f"settings_notifications_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🚨 Emergency",
                        callback_data=f"settings_emergency_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "📊 View All",
                        callback_data=f"settings_view_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🔄 Reset",
                        callback_data=f"settings_reset_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "❌ Close",
                        callback_data=f"settings_close_{user_context.user.user_id}",
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Format status safely to avoid Markdown issues
            trading_enabled = user_context.settings.get("trading_enabled", True)
            status_text = "🟢 Active" if trading_enabled else "🔴 Paused"

            settings_text = f"""⚙️ **TRADING SETTINGS**

👤 **User:** {user_context.user.first_name or user_context.user.username}
🎯 **Plan:** {user_context.user.subscription_tier.title()}

**Current Configuration:**
🏦 **Exchange:** {current_exchange}
🔧 **Strategy:** {current_strategy}
⚡ **Status:** {status_text}

**Configure your trading preferences:**"""

            await query.edit_message_text(
                settings_text, parse_mode="Markdown", reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error showing main settings menu: {e}")
            await query.edit_message_text("❌ Error loading settings menu.")

    # Notification System
    async def _notification_worker(self, worker_name: str):
        """Background worker for processing notifications"""
        logger.info(f"Notification worker {worker_name} started")

        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()

                # Process message
                await self._process_system_message(message)

                # Mark task as done
                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"Error in notification worker {worker_name}: {e}")
                await asyncio.sleep(1)  # Brief pause on error

    async def _process_system_message(self, message: SystemMessage):
        """Process system message (signal, alert, etc.)"""
        try:
            user_context = self.active_users.get(message.user_id)
            if not user_context:
                # User not active, skip
                return

            if message.message_type == "signal":
                await self._send_signal_notification(message.user_id, message.content)
            elif message.message_type == "alert":
                await self._send_alert_notification(message.user_id, message.content)
            elif message.message_type == "system":
                await self._send_system_notification(message.user_id, message.content)

            await self._increment_stat("messages_sent_today")

        except Exception as e:
            logger.error(f"Error processing system message: {e}")
            await self._increment_stat("errors_today")

    async def _send_signal_notification(self, user_id: int, content: Dict):
        """Send signal notification to user"""
        try:
            signal = content.get("signal", {})
            message = f"""**Trading Signal**

Symbol: {signal.get('symbol', 'N/A')}
Action: {signal.get('action', 'N/A')}
Price: ${signal.get('price', 0):.4f}
Confidence: {signal.get('confidence', 0):.1f}%

Time: {datetime.now().strftime('%H:%M:%S')}"""

            # Find user's Telegram ID
            for telegram_id, context in self.active_users.items():
                if context.user.user_id == user_id:
                    await self.application.bot.send_message(
                        chat_id=telegram_id, text=message, parse_mode="Markdown"
                    )
                    break
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}")

    async def _send_alert_notification(self, user_id: int, content: Dict):
        """Send alert notification to user"""
        try:
            title = content.get("title", "Alert")
            alert_content = content.get("content", "")
            alert_type = content.get("type", "info")

            icon = (
                "!" if alert_type == "warning" else "i" if alert_type == "info" else "X"
            )

            message = f"""**{icon} {title}**

{alert_content}

Time: {datetime.now().strftime('%H:%M:%S')}"""

            # Find user's Telegram ID
            for telegram_id, context in self.active_users.items():
                if context.user.user_id == user_id:
                    await self.application.bot.send_message(
                        chat_id=telegram_id, text=message, parse_mode="Markdown"
                    )
                    break
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")

    async def _send_system_notification(self, user_id: int, content: Dict):
        """Send system notification to user"""
        try:
            system_content = content.get("content", "")

            message = f"""**System Notification**

{system_content}

Time: {datetime.now().strftime('%H:%M:%S')}"""

            # Find user's Telegram ID
            for telegram_id, context in self.active_users.items():
                if context.user.user_id == user_id:
                    await self.application.bot.send_message(
                        chat_id=telegram_id, text=message, parse_mode="Markdown"
                    )
                    break
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")

    # Public API for sending notifications
    async def send_signal_to_user(self, user_id: int, signal: Signal):
        """Send trading signal to specific user"""
        message = SystemMessage(
            message_type="signal",
            user_id=user_id,
            content={"signal": signal.__dict__},
            priority=1,
            created_at=datetime.now(),
        )

        await self.message_queue.put(message)

    async def send_alert_to_user(
        self, user_id: int, title: str, content: str, alert_type: str = "info"
    ):
        """Send alert to specific user"""
        message = SystemMessage(
            message_type="alert",
            user_id=user_id,
            content={"title": title, "content": content, "type": alert_type},
            priority=2,
            created_at=datetime.now(),
        )

        await self.message_queue.put(message)

    async def broadcast_system_message(
        self, content: str, subscription_tiers: List[str] = None
    ):
        """Broadcast message to all users or specific subscription tiers"""
        target_users = []

        for telegram_id, context in self.active_users.items():
            if (
                subscription_tiers is None
                or context.user.subscription_tier in subscription_tiers
            ):
                target_users.append(context.user.user_id)

        for user_id in target_users:
            message = SystemMessage(
                message_type="system",
                user_id=user_id,
                content={"content": content},
                priority=3,
                created_at=datetime.now(),
            )
            await self.message_queue.put(message)

        logger.info(f"Broadcasted message to {len(target_users)} users")

    # Maintenance and Monitoring
    async def _maintenance_task(self):
        """Regular maintenance tasks"""
        while True:
            try:
                # Clean up inactive users every 30 minutes
                await self._cleanup_inactive_users()

                # Clean up expired sessions
                await user_service.cleanup_expired_sessions()

                # Reset daily stats if needed
                await self._reset_daily_stats_if_needed()

                await asyncio.sleep(1800)  # 30 minutes

            except Exception as e:
                logger.error(f"Error in maintenance task: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def _stats_update_task(self):
        """Update system statistics"""
        while True:
            try:
                # Update user counts
                self.stats["total_users"] = await multi_user_db.get_active_users_count()
                self.stats["active_users_24h"] = len(
                    [
                        ctx
                        for ctx in self.active_users.values()
                        if ctx.last_interaction > datetime.now() - timedelta(hours=24)
                    ]
                )

                # Log metrics
                await multi_user_db.log_system_metric(
                    "active_users", self.stats["active_users_24h"]
                )
                await multi_user_db.log_system_metric(
                    "total_users", self.stats["total_users"]
                )
                await multi_user_db.log_system_metric(
                    "messages_sent", self.stats["messages_sent_today"]
                )

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in stats update task: {e}")
                await asyncio.sleep(60)

    async def _health_check_task(self):
        """Monitor system health"""
        while True:
            try:
                # Check queue sizes
                queue_size = self.message_queue.qsize()
                if queue_size > 5000:
                    logger.warning(f"High message queue size: {queue_size}")
                    self.system_healthy = False
                else:
                    self.system_healthy = True

                # Check error rate
                error_rate = self.stats["errors_today"] / max(
                    self.stats["commands_processed_today"], 1
                )
                if error_rate > 0.1:  # 10% error rate
                    logger.warning(f"High error rate: {error_rate:.2%}")

                await asyncio.sleep(60)  # 1 minute

            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)

    # Helper Methods
    async def _check_user_rate_limit(self, telegram_id: int) -> bool:
        """Check if user is within rate limits"""
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d %H:%M")

        if telegram_id not in self.rate_limits:
            self.rate_limits[telegram_id] = {}

        user_limits = self.rate_limits[telegram_id]
        current_count = user_limits.get(minute_key, 0)

        if current_count >= 10:  # 10 commands per minute
            return False

        user_limits[minute_key] = current_count + 1

        # Clean old entries
        old_keys = [
            k
            for k in user_limits.keys()
            if k < (now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
        ]
        for key in old_keys:
            del user_limits[key]

        return True

    def _format_features(self, features: List[str]) -> str:
        """Format feature list for display"""
        feature_map = {
            "basic_signals": "• Basic Trading Signals",
            "advanced_signals": "• Advanced Signal Analysis",
            "basic_stats": "• Basic Performance Stats",
            "detailed_stats": "• Detailed Analytics",
            "custom_alerts": "• Custom Alert System",
            "custom_strategies": "• Custom Strategy Builder",
            "api_access": "• API Access",
            "priority_support": "• Priority Support",
        }

        return "\n".join([feature_map.get(f, f"• {f}") for f in features])

    async def _increment_stat(self, stat_name: str):
        """Thread-safe stat increment"""
        self.stats[stat_name] = self.stats.get(stat_name, 0) + 1

    async def _reset_daily_stats_if_needed(self):
        """Reset daily stats if new day"""
        today = datetime.now().date()
        if today > self.stats["last_reset"]:
            self.stats.update(
                {
                    "messages_sent_today": 0,
                    "commands_processed_today": 0,
                    "errors_today": 0,
                    "json_parse_errors_today": 0,
                    "errors_network_today": 0,
                    "errors_timeout_today": 0,
                    "errors_bad_request_today": 0,
                    "errors_service_init_today": 0,
                    "errors_json_parse_today": 0,
                    "errors_unknown_today": 0,
                    "last_reset": today,
                }
            )
            logger.info("Daily stats reset")

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler with improved error handling"""
        error = context.error
        error_message = str(error)

        # Log the error with more context
        if isinstance(update, Update):
            logger.error(f"Exception while handling update {update.update_id}: {error}")
        else:
            logger.error(f"Exception while handling update: {error}")

        await self._increment_stat("errors_today")

        # Handle specific error types with appropriate responses
        try:
            if "Extra data" in error_message and "line" in error_message:
                # JSON parsing error - likely malformed update
                logger.warning(
                    f"JSON parsing error in telegram update: {error_message}"
                )

                # Track JSON parsing errors
                await self._increment_stat("json_parse_errors_today")

                # Enhanced JSON error recovery
                try:
                    logger.info("Attempting enhanced JSON error recovery...")

                    # Clear any pending updates to prevent repeated errors
                    if hasattr(self, "application") and self.application:
                        try:
                            # Stop polling temporarily to clear buffer
                            if (
                                hasattr(self.application.updater, "_running")
                                and self.application.updater._running
                            ):
                                logger.info(
                                    "Temporarily stopping polling to clear buffer..."
                                )
                                await self.application.updater.stop()
                                await asyncio.sleep(2)  # Wait for cleanup

                            # Create a temporary bot instance to clear updates with better error handling
                            temp_bot = Bot(token=self.bot_token)
                            try:
                                # Get and discard pending updates with higher limit to clear buffer
                                logger.info("Clearing corrupted update buffer...")
                                updates = await temp_bot.get_updates(
                                    offset=-1,
                                    limit=100,
                                    timeout=5,
                                    allowed_updates=["message", "callback_query"],
                                )

                                if updates:
                                    # Get the last update ID and use it as offset to skip all previous
                                    last_update_id = max(
                                        update.update_id for update in updates
                                    )
                                    await temp_bot.get_updates(
                                        offset=last_update_id + 1, limit=1, timeout=1
                                    )
                                    logger.info(
                                        f"Cleared {len(updates)} corrupted updates, last ID: {last_update_id}"
                                    )
                                else:
                                    logger.info("No pending updates found to clear")

                            except Exception as temp_error:
                                logger.warning(
                                    f"Temp bot clear attempt failed: {temp_error}"
                                )
                            finally:
                                # Ensure temp bot session is closed
                                if hasattr(temp_bot, "_bot") and hasattr(
                                    temp_bot._bot, "_session"
                                ):
                                    await temp_bot._bot._session.close()
                                elif hasattr(temp_bot, "_request") and hasattr(
                                    temp_bot._request, "_session"
                                ):
                                    await temp_bot._request._session.close()

                            # Restart polling with clean state
                            if (
                                hasattr(self.application.updater, "_running")
                                and not self.application.updater._running
                            ):
                                logger.info("Restarting polling with clean state...")
                                await self.application.updater.start_polling(
                                    poll_interval=3.0,  # Slightly slower to reduce errors
                                    timeout=15,  # Reduced timeout
                                    drop_pending_updates=True,  # Force drop any remaining pending
                                    allowed_updates=["message", "callback_query"],
                                )
                                logger.info("Polling restarted successfully")

                        except Exception as recovery_error:
                            logger.error(
                                f"Enhanced update recovery failed: {recovery_error}"
                            )

                    # Check if we're getting too many JSON errors and implement circuit breaker
                    json_error_count = self.stats.get("json_parse_errors_today", 0)
                    if json_error_count > 5:  # Lower threshold for faster response
                        logger.warning(
                            f"High JSON error count ({json_error_count}), implementing progressive delay"
                        )
                        # Progressive delay based on error count
                        delay = min(json_error_count * 2, 30)  # Max 30 seconds
                        await asyncio.sleep(delay)

                        # If errors are very high, restart the entire bot application
                        if json_error_count > 20:
                            logger.error(
                                "Critical JSON error count reached, attempting bot restart..."
                            )
                            try:
                                await self._restart_bot_application()
                            except Exception as restart_error:
                                logger.error(f"Bot restart failed: {restart_error}")

                except Exception as json_recovery_error:
                    logger.error(
                        f"Error in enhanced JSON error recovery: {json_recovery_error}"
                    )

                # Don't respond to user for JSON errors, just log and recover
                return

            elif (
                "NetworkError" in error_message or "getaddrinfo failed" in error_message
            ):
                # Network connectivity issues
                logger.warning(f"Network connectivity issue: {error_message}")

                # Enhanced exponential backoff for network errors
                retry_count = getattr(self, "_network_retry_count", 0)
                if retry_count < 5:  # Max 5 retries
                    self._network_retry_count = retry_count + 1
                    wait_time = min(2**retry_count, 60)  # Max 60 seconds
                    logger.info(
                        f"Network error retry {retry_count + 1}/5, waiting {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)

                    # Try to restart connection after network error
                    try:
                        if (
                            hasattr(self.application, "updater")
                            and self.application.updater
                        ):
                            await self.application.updater.stop()
                            await asyncio.sleep(2)
                            await self.application.updater.start_polling(
                                poll_interval=3.0,
                                timeout=20,
                                drop_pending_updates=True,
                                allowed_updates=["message", "callback_query"],
                            )
                            logger.info("Network connection reestablished")
                    except Exception as network_restart_error:
                        logger.error(f"Network restart failed: {network_restart_error}")
                else:
                    # Reset retry count after max retries
                    self._network_retry_count = 0
                    logger.error(
                        "Max network retries exceeded, continuing without response"
                    )

                # Don't try to respond to user during network issues
                return

            elif "ConnectTimeout" in error_message or "ReadTimeout" in error_message:
                # Telegram API timeout - implement better timeout handling
                logger.warning(f"Telegram API timeout: {error_message}")

                # Track timeout errors
                await self._increment_stat("errors_timeout_today")

                # If we're getting too many timeouts, adjust polling settings
                timeout_count = self.stats.get("errors_timeout_today", 0)
                if timeout_count > 10:
                    logger.warning(
                        f"High timeout count ({timeout_count}), adjusting polling settings"
                    )
                    try:
                        if (
                            hasattr(self.application, "updater")
                            and self.application.updater
                        ):
                            await self.application.updater.stop()
                            await asyncio.sleep(3)
                            # Restart with more conservative settings
                            await self.application.updater.start_polling(
                                poll_interval=5.0,  # Slower polling
                                timeout=30,  # Longer timeout
                                drop_pending_updates=True,
                                allowed_updates=["message", "callback_query"],
                            )
                            logger.info(
                                "Adjusted polling settings for timeout resilience"
                            )
                    except Exception as timeout_adjust_error:
                        logger.error(
                            f"Timeout adjustment failed: {timeout_adjust_error}"
                        )

                return

            elif "Conflict" in error_message and "webhook" in error_message.lower():
                # Webhook conflict (multiple bot instances)
                logger.error(f"Webhook conflict detected: {error_message}")
                # This is a critical configuration issue - try to resolve
                try:
                    temp_bot = Bot(token=self.bot_token)
                    await temp_bot.delete_webhook()
                    logger.info("Webhook deleted to resolve conflict")
                except Exception as webhook_error:
                    logger.error(f"Failed to delete webhook: {webhook_error}")
                return

            elif "BadRequest" in error_message:
                # Invalid request to Telegram API
                logger.warning(f"Bad request to Telegram API: {error_message}")
                await self._increment_stat("errors_bad_request_today")

                # Try to respond to user if we have an update
                if isinstance(update, Update) and update.effective_chat:
                    try:
                        await update.effective_chat.send_message(
                            "⚠️ Sorry, I couldn't process that request. "
                            "Please try again with a different command.",
                            timeout=10,
                        )
                    except Exception as send_error:
                        logger.error(f"Failed to send error response: {send_error}")
                return

            elif "Unauthorized" in error_message:
                # Bot token issue
                logger.critical(f"Bot authorization error: {error_message}")
                # This is critical - bot token is invalid
                return

            elif (
                "UserService" in error_message
                and "get_user_performance" in error_message
            ):
                # Service initialization issue
                logger.error(f"Service initialization error: {error_message}")
                await self._increment_stat("errors_service_init_today")

                if isinstance(update, Update) and update.effective_chat:
                    try:
                        await update.effective_chat.send_message(
                            "⏳ **System Initializing**\\n\\n"
                            "Some services are still starting up. "
                            "Please try again in a few moments.",
                            parse_mode="Markdown",
                            timeout=10,
                        )
                    except Exception as send_error:
                        logger.error(
                            f"Failed to send initialization error response: {send_error}"
                        )
                return

            else:
                # Generic error handling
                logger.error(f"Unhandled error in telegram bot: {error_message}")
                await self._increment_stat("errors_unknown_today")

                # Try to send a generic error message to user
                if isinstance(update, Update) and update.effective_chat:
                    try:
                        await update.effective_chat.send_message(
                            "❌ **Temporary Error**\\n\\n"
                            "I encountered an unexpected error. "
                            "Please try again in a moment.",
                            parse_mode="Markdown",
                            timeout=10,
                        )
                    except Exception as send_error:
                        logger.error(
                            f"Failed to send generic error response: {send_error}"
                        )

        except Exception as handler_error:
            # Error in error handler - log and continue
            logger.error(f"Error in error handler: {handler_error}")

        # Reset network retry count on successful error handling
        if "NetworkError" not in error_message:
            self._network_retry_count = 0

        # Update error statistics for monitoring
        try:
            await self._track_error_type(error_message)
        except Exception as stats_error:
            logger.error(f"Error updating error statistics: {stats_error}")

    async def _restart_bot_application(self):
        """Restart the bot application to recover from critical errors"""
        try:
            logger.info("Attempting to restart bot application...")

            # Stop current application
            if hasattr(self, "application") and self.application:
                await self.application.stop()
                await self.application.shutdown()

            # Wait for cleanup
            await asyncio.sleep(5)

            # Recreate application with enhanced settings
            from telegram.ext import Application

            self.application = (
                Application.builder()
                .token(self.bot_token)
                .concurrent_updates(128)  # Reduced from 256
                .pool_timeout(45)
                .connect_timeout(45)
                .read_timeout(45)
                .write_timeout(45)
                .build()
            )

            # Re-setup handlers
            await self._setup_handlers()

            # Initialize and start
            await self.application.initialize()
            await self.application.start()

            # Start polling with conservative settings
            await self.application.updater.start_polling(
                poll_interval=4.0,
                timeout=25,
                drop_pending_updates=True,
                allowed_updates=["message", "callback_query"],
            )

            logger.info("Bot application restarted successfully")

            # Reset error counts after successful restart
            self.stats["json_parse_errors_today"] = 0
            self.stats["errors_timeout_today"] = 0

        except Exception as restart_error:
            logger.error(f"Failed to restart bot application: {restart_error}")
            raise

    async def _track_error_type(self, error_message: str):
        """Track different types of errors for monitoring"""
        try:
            error_type = "unknown"

            if "NetworkError" in error_message or "getaddrinfo" in error_message:
                error_type = "network"
            elif "Timeout" in error_message:
                error_type = "timeout"
            elif "BadRequest" in error_message:
                error_type = "bad_request"
            elif "UserService" in error_message:
                error_type = "service_init"
            elif "JSON" in error_message:
                error_type = "json_parse"

            # Update error type counter
            current_count = self.stats.get(f"errors_{error_type}_today", 0)
            self.stats[f"errors_{error_type}_today"] = current_count + 1

            # Log warning if specific error type is high
            if current_count > 20:  # More than 20 of same error type
                logger.warning(f"High {error_type} error count: {current_count + 1}")

        except Exception as e:
            logger.error(f"Error tracking error type: {e}")

    # Bot Control
    async def start(self) -> Application:
        """Start the bot with enhanced instance conflict prevention and JSON error resilience"""
        logger.info("Initializing Multi-User Trading Bot...")

        # Enhanced bot instance conflict prevention
        logger.info("Checking for existing bot instances...")
        try:
            # Try to get bot info to detect if another instance is running
            temp_bot = Bot(token=self.bot_token)
            try:
                # Test if we can connect without conflicts
                await temp_bot.get_me()
                logger.info("Bot token verified successfully")

                # Clear any pending updates to prevent conflicts and JSON errors
                logger.info("Clearing pending updates...")
                try:
                    # Use multiple calls to ensure all updates are cleared
                    for i in range(3):  # Try up to 3 times
                        updates = await temp_bot.get_updates(
                            offset=-1,
                            limit=100,
                            timeout=5,
                            allowed_updates=["message", "callback_query"],
                        )
                        if not updates:
                            break

                        last_update_id = max(update.update_id for update in updates)
                        await temp_bot.get_updates(
                            offset=last_update_id + 1, limit=1, timeout=1
                        )
                        logger.info(
                            f"Cleared {len(updates)} pending updates (attempt {i+1})"
                        )

                    logger.info("All pending updates cleared successfully")
                except Exception as clear_error:
                    logger.warning(f"Error clearing updates: {clear_error}")
                    # Continue anyway

            except Exception as token_error:
                logger.error(f"Bot token verification failed: {token_error}")
                raise ValueError(f"Invalid bot token or network issue: {token_error}")
            finally:
                # Close temporary bot session properly
                try:
                    if hasattr(temp_bot, "_bot") and hasattr(temp_bot._bot, "_session"):
                        await temp_bot._bot._session.close()
                    elif hasattr(temp_bot, "_request") and hasattr(
                        temp_bot._request, "_session"
                    ):
                        await temp_bot._request._session.close()
                except Exception as close_error:
                    logger.warning(f"Error closing temp bot session: {close_error}")

        except Exception as e:
            logger.error(f"Error during bot instance check: {e}")
            # Continue anyway - the error might be due to network issues

        # Initialize database and ensure it's ready
        await multi_user_db.initialize()

        # Initialize rate limiter and other services
        await self.initialize()

        # Build application with optimized settings for high-load production and JSON error resilience
        application = (
            Application.builder()
            .token(self.bot_token)
            .concurrent_updates(128)  # Reduced from 256 to prevent overload
            .pool_timeout(45)  # Increased timeouts for stability
            .connect_timeout(45)
            .read_timeout(45)
            .write_timeout(45)
            .build()
        )

        # Set the application instance before setting up handlers
        self.application = application

        # Add all handlers (now that self.application is set)
        await self._setup_handlers()

        # Add error handler
        application.add_error_handler(self._error_handler)

        # Optimized polling configuration for production
        await application.initialize()

        # Start polling with optimized settings
        logger.info("Starting bot polling with enhanced error resilience...")
        await application.start()

        # Use more conservative polling settings to prevent JSON errors
        await application.updater.start_polling(
            poll_interval=3.0,  # Increased from 2.0 to reduce load and errors
            timeout=25,  # Reduced from 20 for faster error detection
            drop_pending_updates=True,  # Clear pending updates to avoid backlog
            allowed_updates=[
                "message",
                "callback_query",
            ],  # Removed inline_query to reduce complexity
        )

        # Start background cleanup task for memory optimization and error prevention
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._periodic_health_check())

        logger.info(
            "Multi-User Trading Bot started successfully with enhanced error resilience!"
        )
        logger.info(f"Bot username: @{application.bot.username}")

        return application

    async def _periodic_cleanup(self):
        """Periodic cleanup task to optimize memory usage"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Clean up old rate limit data
                current_time = time.time()
                expired_users = []

                for telegram_id, rate_data in self.rate_limits.items():
                    if (
                        current_time - rate_data.get("last_request", 0) > 3600
                    ):  # 1 hour old
                        expired_users.append(telegram_id)

                for telegram_id in expired_users:
                    del self.rate_limits[telegram_id]

                if expired_users:
                    logger.debug(
                        f"Cleaned up rate limit data for {len(expired_users)} inactive users"
                    )

                # Clean up old user contexts
                expired_contexts = []
                for telegram_id, context in self.active_users.items():
                    if hasattr(context, "last_activity"):
                        if (
                            current_time - context.last_activity.timestamp() > 1800
                        ):  # 30 minutes old
                            expired_contexts.append(telegram_id)

                for telegram_id in expired_contexts:
                    del self.active_users[telegram_id]

                if expired_contexts:
                    logger.debug(
                        f"Cleaned up contexts for {len(expired_contexts)} inactive users"
                    )

                # Force garbage collection if too many objects
                import gc

                if len(gc.get_objects()) > 50000:  # High object count
                    gc.collect()
                    logger.debug("Performed garbage collection for memory optimization")

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def stop(self):
        """Stop the bot gracefully"""
        logger.info("Stopping multi-user bot...")

        try:
            # Set maintenance mode to stop accepting new requests
            self.maintenance_mode = True

            # Wait a moment for current operations to complete
            await asyncio.sleep(1.0)

            # Stop notification workers
            logger.info("Stopping notification workers...")
            for worker in self.notification_workers:
                worker.cancel()

            # Wait for workers to stop
            if self.notification_workers:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self.notification_workers, return_exceptions=True
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Some notification workers didn't stop within timeout"
                    )

            # Cancel periodic cleanup task if running
            logger.info("Stopping background tasks...")
            for task in asyncio.all_tasks():
                if hasattr(task, "get_name") and "periodic_cleanup" in task.get_name():
                    task.cancel()

            # Stop the telegram application gracefully
            if self.application:
                logger.info("Stopping Telegram application...")
                try:
                    # Stop polling first
                    if self.application.updater.running:
                        await self.application.updater.stop()

                    # Then stop the application
                    await self.application.stop()

                    # Finally shutdown the application
                    await self.application.shutdown()

                except Exception as e:
                    logger.error(f"Error stopping Telegram application: {e}")

            # Clear remaining queues and caches
            logger.info("Clearing remaining data...")
            try:
                # Clear message queue
                while not self.message_queue.empty():
                    try:
                        self.message_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                # Clear user contexts
                self.active_users.clear()
                self.rate_limits.clear()

            except Exception as e:
                logger.error(f"Error clearing bot data: {e}")

            logger.info("Multi-user bot stopped gracefully")

        except Exception as e:
            logger.error(f"Error during bot shutdown: {e}")
            # Force stop if graceful shutdown fails
            if self.application:
                try:
                    await self.application.stop()
                except Exception as force_error:
                    logger.error(f"Error in force stop: {force_error}")

    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            **self.stats,
            "active_contexts": len(self.active_users),
            "queue_size": self.message_queue.qsize(),
            "system_healthy": self.system_healthy,
            "maintenance_mode": self.maintenance_mode,
        }

    async def _periodic_health_check(self):
        """Periodic health check to prevent JSON errors and maintain bot stability"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if not self.running:
                    break

                # Check JSON error rate
                json_errors = self.stats.get("json_parse_errors_today", 0)
                timeout_errors = self.stats.get("errors_timeout_today", 0)
                total_errors = self.stats.get("errors_today", 0)

                logger.info(
                    f"Health check: JSON errors: {json_errors}, Timeout errors: {timeout_errors}, Total errors: {total_errors}"
                )

                # If error rates are high, proactively restart polling
                if json_errors > 3 or timeout_errors > 15 or total_errors > 50:
                    logger.warning(
                        f"High error rates detected, performing preventive restart..."
                    )
                    try:
                        if (
                            hasattr(self.application, "updater")
                            and self.application.updater
                        ):
                            await self.application.updater.stop()
                            await asyncio.sleep(3)

                            # Clear any pending updates
                            temp_bot = Bot(token=self.bot_token)
                            try:
                                updates = await temp_bot.get_updates(
                                    offset=-1, limit=50, timeout=3
                                )
                                if updates:
                                    last_id = max(
                                        update.update_id for update in updates
                                    )
                                    await temp_bot.get_updates(
                                        offset=last_id + 1, limit=1, timeout=1
                                    )
                                    logger.info(
                                        f"Cleared {len(updates)} updates during health check"
                                    )
                            except Exception as clear_error:
                                logger.warning(
                                    f"Update clear during health check failed: {clear_error}"
                                )
                            finally:
                                if hasattr(temp_bot, "_bot") and hasattr(
                                    temp_bot._bot, "_session"
                                ):
                                    await temp_bot._bot._session.close()
                                elif hasattr(temp_bot, "_request") and hasattr(
                                    temp_bot._request, "_session"
                                ):
                                    await temp_bot._request._session.close()

                            # Restart with clean settings
                            await self.application.updater.start_polling(
                                poll_interval=3.0,
                                timeout=25,
                                drop_pending_updates=True,
                                allowed_updates=["message", "callback_query"],
                            )

                            logger.info("Preventive restart completed successfully")

                            # Reset some error counters after successful restart
                            self.stats["json_parse_errors_today"] = max(
                                0, json_errors - 2
                            )
                            self.stats["errors_timeout_today"] = max(
                                0, timeout_errors - 5
                            )

                    except Exception as restart_error:
                        logger.error(f"Preventive restart failed: {restart_error}")

                # Check if bot is responsive
                try:
                    if hasattr(self.application, "bot"):
                        await self.application.bot.get_me()
                        logger.debug("Bot responsiveness check passed")
                except Exception as responsiveness_error:
                    logger.warning(
                        f"Bot responsiveness check failed: {responsiveness_error}"
                    )

            except Exception as health_check_error:
                logger.error(f"Health check error: {health_check_error}")
                await asyncio.sleep(60)  # Wait before next check if error occurred

    async def shutdown(self):
        """Shutdown the bot gracefully"""
        logger.info("Shutting down Multi-User Trading Bot...")

        self.running = False

        try:
            # Stop background workers
            for worker in self.notification_workers:
                if not worker.done():
                    worker.cancel()

            # Wait for workers to finish
            if self.notification_workers:
                await asyncio.gather(*self.notification_workers, return_exceptions=True)

            # Stop the application
            if hasattr(self, "application") and self.application:
                if hasattr(self.application, "updater") and self.application.updater:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()

            # Close user service
            if hasattr(user_service, "close"):
                await user_service.close()

            # Close database connections
            if hasattr(multi_user_db, "close"):
                await multi_user_db.close()

            logger.info("Bot shutdown completed")

        except Exception as shutdown_error:
            logger.error(f"Error during shutdown: {shutdown_error}")


# Global bot instance
multi_user_bot = None  # Will be set by production_main.py after instantiation
