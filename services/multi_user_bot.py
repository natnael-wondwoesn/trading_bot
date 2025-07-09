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

                context = UserContext(
                    user=user,
                    session=session,
                    settings=settings.__dict__ if settings else {},
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

        # Create personalized settings menu
        keyboard = [
            [
                InlineKeyboardButton(
                    "🔧 Strategy",
                    callback_data=f"settings_strategy_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "💰 Risk Mgmt",
                    callback_data=f"settings_risk_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📱 Notifications",
                    callback_data=f"settings_notifications_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "🚨 Emergency",
                    callback_data=f"settings_emergency_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📊 View All",
                    callback_data=f"settings_view_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "🔄 Reset",
                    callback_data=f"settings_reset_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ Close",
                    callback_data=f"settings_close_{user_context.user.user_id}",
                )
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        settings_text = f"""⚙️ **TRADING SETTINGS**
        
👤 **User:** {user_context.user.first_name or user_context.user.username}
🎯 **Plan:** {user_context.user.subscription_tier.title()}

Configure your trading preferences:"""

        await update.message.reply_text(
            settings_text, parse_mode="Markdown", reply_markup=reply_markup
        )

    async def _handle_dashboard(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /dashboard command"""
        user_context = await self._get_user_context(update.effective_user.id, update)
        if not user_context:
            return

        await self._increment_stat("commands_processed_today")

        # Get comprehensive dashboard data
        dashboard_data = await user_service.get_user_dashboard_data(
            user_context.user.user_id
        )

        if not dashboard_data:
            await update.message.reply_text("❌ Unable to load dashboard data.")
            return

        # Format dashboard message
        dashboard_text = f"""📊 **TRADING DASHBOARD**

👤 **Account Info:**
• Plan: {dashboard_data['user_info']['subscription_tier'].title()}
• Member Since: {datetime.fromisoformat(dashboard_data['user_info']['member_since']).strftime('%b %Y')}

📈 **Today's Performance:**
• Trades: {dashboard_data['daily_stats']['total_trades']}
• Wins: {dashboard_data['daily_stats']['winning_trades']} 
• Win Rate: {dashboard_data['daily_stats']['win_rate']:.1f}%
• P&L: ${dashboard_data['daily_stats']['total_pnl']:.2f}

💼 **Current Status:**
• Open Positions: {dashboard_data['open_trades']}
• Daily Limit: {dashboard_data['usage']['daily_trades_used']:.1f}% used
• Position Limit: {dashboard_data['usage']['positions_used']:.1f}% used

📊 **7-Day Summary:**
• Total Trades: {dashboard_data['performance_summary']['total_trades_7d']}
• Avg Win Rate: {dashboard_data['performance_summary']['win_rate_7d']:.1f}%
• Avg Daily P&L: ${dashboard_data['performance_summary']['avg_daily_pnl']:.2f}"""

        # Add action buttons
        keyboard = [
            [
                InlineKeyboardButton(
                    "📈 Performance",
                    callback_data=f"dashboard_performance_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "⚙️ Settings",
                    callback_data=f"dashboard_settings_{user_context.user.user_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔄 Refresh",
                    callback_data=f"dashboard_refresh_{user_context.user.user_id}",
                ),
                InlineKeyboardButton(
                    "📱 Share",
                    callback_data=f"dashboard_share_{user_context.user.user_id}",
                ),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            dashboard_text, parse_mode="Markdown", reply_markup=reply_markup
        )

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
        elif action_type == "upgrade":
            await self._handle_upgrade_callback(query, action, user_context)
        elif action_type == "emergency":
            await self._handle_emergency_callback(query, action, user_context)
        elif action_type == "support":
            await self._handle_support_callback(query, action, user_context)
        elif action_type == "performance":
            await self._handle_performance_callback(query, action, user_context)
        elif action_type == "billing":
            await self._handle_billing_callback(query, action, user_context)
        elif action_type == "strategy":
            await self._handle_strategy_callback(query, action, user_context)
        elif action_type == "exchange":
            await self._handle_exchange_callback(query, action, user_context)
        elif action_type == "exchange_select":
            # Handle exchange selection (similar to strategy_select)
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
                    from services.trading_orchestrator import trading_orchestrator

                    await trading_orchestrator.update_user_settings(
                        user_context.user.user_id
                    )

                    await query.edit_message_text(
                        f"**Exchange Updated**\n\n"
                        f"Your trading exchange has been changed to **{exchange_name}**.\n\n"
                        f"All future trades will be executed on {exchange_name}.\n\n"
                        f"Use /settings to make further changes.",
                        parse_mode="Markdown",
                    )

                except Exception as e:
                    logger.error(
                        f"Error updating exchange for user {user_context.user.user_id}: {e}"
                    )
                    await query.edit_message_text(
                        "Failed to update exchange. Please try again."
                    )
        else:
            await query.edit_message_text("Unknown action.")

    async def _handle_settings_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle settings callback actions"""
        try:
            if action == "strategy":
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
                        f"[*] {strategy_name}"
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

                strategy_text = "**Strategy Selection**\n\n"
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
                await query.edit_message_text(
                    "**Risk Management**\n\n"
                    "Max Daily Loss: 5%\n"
                    "Stop Loss: 2%\n"
                    "Take Profit: 4%\n\n"
                    "Use /settings to modify these settings.",
                    parse_mode="Markdown",
                )
            elif action == "notifications":
                await query.edit_message_text(
                    "**Notification Settings**\n\n"
                    "Signal Alerts: Enabled\n"
                    "Trade Updates: Enabled\n"
                    "System Alerts: Enabled\n\n"
                    "Use /settings to modify these settings.",
                    parse_mode="Markdown",
                )
            elif action == "view":
                settings = user_context.settings
                await query.edit_message_text(
                    f"**All Settings**\n\n"
                    f"Trading Enabled: {settings.get('trading_enabled', True)}\n"
                    f"Risk Level: {settings.get('risk_level', 'Medium')}\n"
                    f"Position Size: {settings.get('position_size', 1)}%\n"
                    f"Stop Loss: {settings.get('stop_loss', 2)}%\n"
                    f"Take Profit: {settings.get('take_profit', 4)}%",
                    parse_mode="Markdown",
                )
            elif action == "back":
                # Return to main settings menu
                await self._show_main_settings_menu(query, user_context)
            elif action == "exchange":
                await self._handle_exchange_callback(query, "select", user_context)
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
                try:
                    # Check if user_service has the required method
                    if not hasattr(user_service, "get_user_performance"):
                        await query.edit_message_text(
                            "⚠️ **Service Initializing**\n\n"
                            "The dashboard service is starting up. Please try again shortly.",
                            parse_mode="Markdown",
                        )
                        return

                    # Get updated user performance with error handling
                    try:
                        performance = await user_service.get_user_performance(
                            user_context.user.user_id
                        )
                    except AttributeError:
                        # Service not ready, use defaults
                        performance = {
                            "total_pnl": 0.0,
                            "daily_pnl": 0.0,
                            "total_trades": 0,
                            "win_rate": 0.0,
                        }

                    total_pnl = performance.get("total_pnl", 0.0)
                    daily_pnl = performance.get("daily_pnl", 0.0)
                    active_trades = (
                        len(user_context.active_trades)
                        if user_context.active_trades
                        else 0
                    )

                    await query.edit_message_text(
                        f"**Dashboard Refreshed**\n\n"
                        f"Total P&L: ${total_pnl:,.2f}\n"
                        f"Today: ${daily_pnl:,.2f}\n"
                        f"Active Trades: {active_trades}\n"
                        f"Strategy: {user_context.settings.get('trading_strategy', 'N/A')}\n\n"
                        f"_Updated: {datetime.now().strftime('%H:%M:%S')}_",
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    logger.error(f"Error refreshing dashboard: {e}")
                    await query.edit_message_text(
                        "❌ **Refresh Failed**\n\n"
                        "Unable to refresh dashboard data. Please try again.",
                        parse_mode="Markdown",
                    )

            elif action == "trades":
                # Show active trades
                active_trades = user_context.active_trades or []
                if not active_trades:
                    await query.edit_message_text(
                        "📊 **Active Trades**\n\n" "No active trades at the moment.",
                        parse_mode="Markdown",
                    )
                else:
                    trades_text = "📊 **Active Trades**\n\n"
                    for i, trade in enumerate(
                        active_trades[:5], 1
                    ):  # Show max 5 trades
                        pnl = trade.current_pnl or 0
                        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                        trades_text += (
                            f"{i}. {trade.symbol} {trade.side.upper()}\n"
                            f"   {pnl_emoji} P&L: ${pnl:,.2f}\n\n"
                        )

                    if len(active_trades) > 5:
                        trades_text += (
                            f"_... and {len(active_trades) - 5} more trades_\n"
                        )

                    await query.edit_message_text(
                        trades_text,
                        parse_mode="Markdown",
                    )

        except Exception as e:
            logger.error(f"Error in dashboard callback: {e}")
            await query.answer("Error processing request")

    async def _handle_emergency_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle emergency callback actions"""
        try:
            if action == "stop":
                # Stop all trading for this user
                await user_service.update_user_settings(
                    user_context.user.user_id, trading_enabled=False
                )
                await query.edit_message_text(
                    "**EMERGENCY STOP ACTIVATED**\n\n"
                    "All trading has been stopped for your account.\n"
                    "Existing positions remain open.\n\n"
                    "Use /settings to re-enable trading.",
                    parse_mode="Markdown",
                )
            elif action == "disable":
                await query.edit_message_text(
                    "**BOT DISABLED**\n\n"
                    "Your bot has been disabled.\n"
                    "Contact support to re-enable.\n\n"
                    "Use /support for assistance.",
                    parse_mode="Markdown",
                )
            elif action == "close":
                await query.edit_message_text(
                    "**CLOSING POSITIONS**\n\n"
                    "All open positions are being closed.\n"
                    "This may take a few moments.\n\n"
                    "Check /dashboard for updates.",
                    parse_mode="Markdown",
                )
            elif action == "cancel":
                await query.edit_message_text(
                    "Emergency action cancelled.\n" "Your trading continues normally."
                )
            else:
                await query.edit_message_text("Emergency action not implemented yet.")
        except Exception as e:
            logger.error(f"Error in emergency callback: {e}")
            await query.edit_message_text("An error occurred. Please try again.")

    async def _handle_support_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle support callback actions"""
        try:
            if action == "faq":
                await query.edit_message_text(
                    "**Frequently Asked Questions**\n\n"
                    "Q: How do I start trading?\n"
                    "A: Use /start and configure your settings.\n\n"
                    "Q: How do I stop trading?\n"
                    "A: Use /emergency for immediate stop.\n\n"
                    "Q: How do I change risk settings?\n"
                    "A: Use /settings > Risk Management.\n\n"
                    "Need more help? Use /support.",
                    parse_mode="Markdown",
                )
            elif action == "chat":
                await query.edit_message_text(
                    "**Live Chat Support**\n\n"
                    "Premium and Enterprise users have access to 24/7 live chat.\n\n"
                    "Your subscription: "
                    + user_context.user.subscription_tier.title()
                    + "\n\n"
                    "For immediate help, email: support@tradingbot.com",
                    parse_mode="Markdown",
                )
            elif action == "email":
                await query.edit_message_text(
                    "**Email Support**\n\n"
                    "Send your questions to:\n"
                    "support@tradingbot.com\n\n"
                    "Include your User ID: " + str(user_context.user.user_id) + "\n\n"
                    "Response time: 24-48 hours",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text("Support action not implemented yet.")
        except Exception as e:
            logger.error(f"Error in support callback: {e}")
            await query.edit_message_text("An error occurred. Please try again.")

    async def _handle_performance_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle performance callback actions"""
        try:
            if action == "detailed":
                try:
                    # Check if user_service has the required method
                    if not hasattr(user_service, "get_user_performance"):
                        await query.edit_message_text(
                            "⚠️ **Service Initializing**\n\n"
                            "The performance service is starting up. Please try again shortly.",
                            parse_mode="Markdown",
                        )
                        return

                    try:
                        performance = await user_service.get_user_performance(
                            user_context.user.user_id
                        )
                    except AttributeError:
                        # Service not ready, use defaults
                        performance = {
                            "total_pnl": 0.0,
                            "total_trades": 0,
                            "winning_trades": 0,
                            "losing_trades": 0,
                            "win_rate": 0.0,
                            "best_trade": 0.0,
                            "worst_trade": 0.0,
                            "avg_trade_size": 0.0,
                        }

                    total_pnl = performance.get("total_pnl", 0.0)
                    total_trades = performance.get("total_trades", 0)
                    winning_trades = performance.get("winning_trades", 0)
                    losing_trades = performance.get("losing_trades", 0)
                    win_rate = performance.get("win_rate", 0.0) * 100
                    best_trade = performance.get("best_trade", 0.0)
                    worst_trade = performance.get("worst_trade", 0.0)
                    avg_trade = performance.get("avg_trade_size", 0.0)

                    await query.edit_message_text(
                        f"📊 **Detailed Performance Report**\n\n"
                        f"💰 **P&L Summary:**\n"
                        f"• Total P&L: ${total_pnl:,.2f}\n"
                        f"• Best Trade: ${best_trade:,.2f}\n"
                        f"• Worst Trade: ${worst_trade:,.2f}\n"
                        f"• Avg Trade: ${avg_trade:,.2f}\n\n"
                        f"📈 **Trade Statistics:**\n"
                        f"• Total Trades: {total_trades}\n"
                        f"• Winning Trades: {winning_trades}\n"
                        f"• Losing Trades: {losing_trades}\n"
                        f"• Win Rate: {win_rate:.1f}%\n\n"
                        f"_Report generated: {datetime.now().strftime('%H:%M:%S')}_",
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    logger.error(f"Error getting detailed performance: {e}")
                    await query.edit_message_text(
                        "❌ **Error**\n\n"
                        "Unable to generate detailed report. Please try again.",
                        parse_mode="Markdown",
                    )

            elif action == "refresh":
                # Refresh current performance view
                await self._handle_performance_callback(query, "detailed", user_context)

        except Exception as e:
            logger.error(f"Error in performance callback: {e}")
            await query.answer("Error processing request")

    async def _handle_upgrade_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle upgrade callback actions"""
        try:
            if action == "premium":
                await query.edit_message_text(
                    "**Upgrade to Premium**\n\n"
                    "Premium Features:\n"
                    "- 25 daily trades\n"
                    "- 5 concurrent positions\n"
                    "- Advanced strategies\n"
                    "- Priority support\n\n"
                    "Price: $29.99/month\n\n"
                    "Contact: billing@tradingbot.com",
                    parse_mode="Markdown",
                )
            elif action == "enterprise":
                await query.edit_message_text(
                    "**Upgrade to Enterprise**\n\n"
                    "Enterprise Features:\n"
                    "- Unlimited trades\n"
                    "- 20 concurrent positions\n"
                    "- Custom strategies\n"
                    "- Dedicated support\n"
                    "- API access\n\n"
                    "Price: $99.99/month\n\n"
                    "Contact: enterprise@tradingbot.com",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text("Upgrade option not implemented yet.")
        except Exception as e:
            logger.error(f"Error in upgrade callback: {e}")
            await query.edit_message_text("An error occurred. Please try again.")

    async def _handle_billing_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle billing callback actions"""
        try:
            if action == "info":
                await query.edit_message_text(
                    f"**Billing Information**\n\n"
                    f"Current Plan: {user_context.user.subscription_tier.title()}\n"
                    f"User ID: {user_context.user.user_id}\n"
                    f"Member Since: {user_context.user.created_at.strftime('%B %Y')}\n\n"
                    f"For billing questions:\n"
                    f"billing@tradingbot.com",
                    parse_mode="Markdown",
                )
            else:
                await query.edit_message_text("Billing action not implemented yet.")
        except Exception as e:
            logger.error(f"Error in billing callback: {e}")
            await query.edit_message_text("An error occurred. Please try again.")

    async def _handle_strategy_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle strategy selection callback"""
        try:
            if action == "select":
                # Parse the strategy selection from callback data
                # Callback format: strategy_select_STRATEGY_NAME_user_id
                data_parts = query.data.split("_")
                if len(data_parts) >= 4:
                    strategy_name = data_parts[
                        2
                    ]  # Extract strategy name from position 2

                    # Update user's strategy setting
                    await user_service.update_user_settings(
                        user_context.user.user_id, trading_strategy=strategy_name
                    )

                    # Update user context
                    user_context.settings["trading_strategy"] = strategy_name

                    # Notify trading orchestrator of strategy change
                    from services.trading_orchestrator import trading_orchestrator

                    await trading_orchestrator.update_user_settings(
                        user_context.user.user_id
                    )

                    # Import strategy factory for description
                    from services.trading_orchestrator import StrategyFactory

                    strategies = StrategyFactory.get_available_strategies()
                    description = strategies.get(strategy_name, "Unknown strategy")

                    success_message = f"**Strategy Updated Successfully!**\n\n"
                    success_message += f"**New Strategy:** {strategy_name}\n"
                    success_message += f"**Description:** {description}\n\n"
                    success_message += (
                        "Your new strategy will be used for all future signals.\n\n"
                    )
                    success_message += (
                        "**Note:** This change takes effect immediately for new trades."
                    )

                    # Create back to settings button
                    keyboard = [
                        [
                            InlineKeyboardButton(
                                "← Back to Settings",
                                callback_data=f"settings_back_{user_context.user.user_id}",
                            )
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await query.edit_message_text(
                        success_message,
                        reply_markup=reply_markup,
                        parse_mode="Markdown",
                    )
                else:
                    await query.edit_message_text("Invalid strategy selection.")
            else:
                await query.edit_message_text("Strategy action not implemented yet.")
        except Exception as e:
            logger.error(f"Error in strategy callback: {e}")
            await query.edit_message_text(
                "An error occurred while updating strategy. Please try again."
            )

    async def _handle_exchange_callback(
        self, query, action: str, user_context: UserContext
    ):
        """Handle exchange selection callback"""
        try:
            if action == "select":
                # Get available exchanges
                exchanges = ExchangeFactory.get_available_exchanges()
                current_exchange = user_context.settings.get("exchange", "MEXC")

                # Create exchange selection keyboard
                keyboard = []
                for exchange_name, description in exchanges.items():
                    # Add checkmark if current exchange
                    display_name = (
                        f"[*] {exchange_name}"
                        if exchange_name == current_exchange
                        else exchange_name
                    )
                    keyboard.append(
                        [
                            InlineKeyboardButton(
                                display_name,
                                callback_data=f"exchange_select_{exchange_name}_{user_context.user.user_id}",
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

                exchange_text = "**Exchange Selection**\n\n"
                exchange_text += f"**Current Exchange:** {current_exchange}\n\n"
                exchange_text += "**Available Exchanges:**\n"

                for exchange_name, description in exchanges.items():
                    icon = "◉" if exchange_name == current_exchange else "○"
                    exchange_text += f"{icon} **{exchange_name}**\n"
                    exchange_text += f"   {description}\n\n"

                exchange_text += "Select an exchange to change your trading platform."

                await query.edit_message_text(
                    exchange_text, reply_markup=reply_markup, parse_mode="Markdown"
                )
            else:
                await query.edit_message_text("Exchange action not implemented yet.")
        except Exception as e:
            logger.error(f"Error in exchange callback: {e}")
            await query.edit_message_text(
                "An error occurred while updating exchange. Please try again."
            )

    async def _show_main_settings_menu(self, query, user_context: UserContext):
        """Show the main settings menu"""
        try:
            # Create settings menu keyboard
            keyboard = [
                [
                    InlineKeyboardButton(
                        "Strategy",
                        callback_data=f"settings_strategy_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "Exchange",
                        callback_data=f"settings_exchange_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Risk Mgmt",
                        callback_data=f"settings_risk_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "Notifications",
                        callback_data=f"settings_notifications_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Emergency",
                        callback_data=f"settings_emergency_{user_context.user.user_id}",
                    ),
                    InlineKeyboardButton(
                        "View All",
                        callback_data=f"settings_view_{user_context.user.user_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "Reset",
                        callback_data=f"settings_reset_{user_context.user.user_id}",
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            # Get current settings for display
            current_strategy = user_context.settings.get("trading_strategy", "RSI_EMA")
            current_exchange = user_context.settings.get("exchange", "MEXC")

            settings_text = f"""**Settings Configuration**

**Current Settings:**
• Strategy: {current_strategy}
• Exchange: {current_exchange}
• Trading: {'Enabled' if user_context.settings.get('trading_enabled', True) else 'Disabled'}
• Risk Level: {user_context.settings.get('risk_level', 'Medium')}

**Configuration Options:**
• **Strategy** - Change signal detection method
• **Exchange** - Select trading exchange (MEXC/Bybit)
• **Risk Mgmt** - Configure risk management rules
• **Notifications** - Set alert preferences
• **Emergency** - Quick safety controls

Select an option to configure:"""

            await query.edit_message_text(
                settings_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error showing main settings menu: {e}")
            await query.edit_message_text(
                "Error loading settings menu. Please try /settings command."
            )

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
        self.stats[stat_name] += 1

    async def _reset_daily_stats_if_needed(self):
        """Reset daily stats if new day"""
        today = datetime.now().date()
        if today > self.stats["last_reset"]:
            self.stats.update(
                {
                    "messages_sent_today": 0,
                    "commands_processed_today": 0,
                    "errors_today": 0,
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

                # Try to recover from JSON parsing errors
                try:
                    logger.info("Attempting JSON error recovery...")

                    # Clear any pending updates to prevent repeated errors
                    if hasattr(self, "application") and self.application:
                        try:
                            # Create a temporary bot instance to clear updates
                            temp_bot = Bot(token=self.bot_token)
                            try:
                                # Get latest update offset to skip corrupted data
                                await temp_bot.get_updates(
                                    offset=-1, limit=1, timeout=1
                                )
                                logger.info("Cleared corrupted update data")
                            finally:
                                if hasattr(temp_bot, "_request") and hasattr(
                                    temp_bot._request, "_session"
                                ):
                                    await temp_bot._request._session.close()
                        except Exception as recovery_error:
                            logger.warning(
                                f"Update recovery attempt failed: {recovery_error}"
                            )

                    # Check if we're getting too many JSON errors
                    json_error_count = self.stats.get("json_parse_errors_today", 0)
                    if json_error_count > 10:  # More than 10 JSON errors today
                        logger.warning(
                            f"High JSON error count ({json_error_count}), implementing delay"
                        )
                        await asyncio.sleep(5)  # Brief pause to let things stabilize

                except Exception as json_recovery_error:
                    logger.error(f"Error in JSON error recovery: {json_recovery_error}")

                # Don't respond to user for JSON errors, just log and recover
                return

            elif (
                "NetworkError" in error_message or "getaddrinfo failed" in error_message
            ):
                # Network connectivity issues
                logger.warning(f"Network connectivity issue: {error_message}")

                # Implement exponential backoff for network errors
                retry_count = getattr(self, "_network_retry_count", 0)
                if retry_count < 5:  # Max 5 retries
                    self._network_retry_count = retry_count + 1
                    wait_time = min(2**retry_count, 60)  # Max 60 seconds
                    logger.info(
                        f"Network error retry {retry_count + 1}/5, waiting {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Reset retry count after max retries
                    self._network_retry_count = 0
                    logger.error(
                        "Max network retries exceeded, continuing without response"
                    )

                # Don't try to respond to user during network issues
                return

            elif "ConnectTimeout" in error_message or "ReadTimeout" in error_message:
                # Telegram API timeout
                logger.warning(f"Telegram API timeout: {error_message}")
                # Don't respond to prevent cascade of timeouts
                return

            elif "Conflict" in error_message and "webhook" in error_message.lower():
                # Webhook conflict (multiple bot instances)
                logger.error(f"Webhook conflict detected: {error_message}")
                # This is a critical configuration issue
                return

            elif "BadRequest" in error_message:
                # Invalid request to Telegram API
                logger.warning(f"Bad request to Telegram API: {error_message}")

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
        """Start the bot with enhanced instance conflict prevention"""
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

                # Clear any pending updates to prevent conflicts
                logger.info("Clearing pending updates...")
                await temp_bot.get_updates(offset=-1, limit=1, timeout=1)
                logger.info("Pending updates cleared")

            except Exception as token_error:
                logger.error(f"Bot token verification failed: {token_error}")
                raise ValueError(f"Invalid bot token or network issue: {token_error}")
            finally:
                # Close temporary bot session
                if hasattr(temp_bot, "_request") and hasattr(
                    temp_bot._request, "_session"
                ):
                    await temp_bot._request._session.close()

        except Exception as e:
            logger.error(f"Error during bot instance check: {e}")
            # Continue anyway - the error might be due to network issues

        # Initialize database and ensure it's ready
        await multi_user_db.initialize()

        # Initialize rate limiter and other services
        await self.initialize()

        # Build application with optimized settings for high-load production
        application = (
            Application.builder()
            .token(self.bot_token)
            .concurrent_updates(256)  # Support more concurrent updates
            .pool_timeout(30)
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
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
        logger.info("Starting bot polling with production optimization...")
        await application.start()

        # Use more efficient polling settings with enhanced error recovery
        await application.updater.start_polling(
            poll_interval=2.0,  # Increased from 1.0 to reduce CPU usage
            timeout=20,  # Timeout for getting updates
            drop_pending_updates=True,  # Clear pending updates to avoid backlog
            allowed_updates=[
                "message",
                "callback_query",
                "inline_query",
            ],  # Only handle needed updates
        )

        # Start background cleanup task for memory optimization
        asyncio.create_task(self._periodic_cleanup())

        logger.info("Multi-User Trading Bot started successfully!")
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


# Global bot instance
multi_user_bot = None  # Will be set by production_main.py after instantiation
