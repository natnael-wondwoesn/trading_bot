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

    async def initialize(self):
        """Initialize the multi-user bot system"""
        logger.info("Initializing multi-user trading bot...")

        # Initialize services
        await user_service.initialize()

        # Create Telegram application
        self.application = Application.builder().token(self.bot_token).build()

        # Add handlers
        await self._setup_handlers()

        # Start background workers
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
                "❌ Session expired. Please use /start to begin."
            )
            return

        data_parts = query.data.split("_")
        if len(data_parts) < 3:
            await query.edit_message_text("❌ Invalid command.")
            return

        action_type = data_parts[0]
        action = data_parts[1]
        user_id = int(data_parts[2])

        # Verify user ID matches
        if user_id != user_context.user.user_id:
            await query.edit_message_text("❌ Unauthorized action.")
            return

        # Route to appropriate handler
        if action_type == "settings":
            await self._handle_settings_callback(query, action, user_context)
        elif action_type == "dashboard":
            await self._handle_dashboard_callback(query, action, user_context)
        elif action_type == "upgrade":
            await self._handle_upgrade_callback(query, action, user_context)
        else:
            await query.edit_message_text("❌ Unknown action.")

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
        """Global error handler"""
        logger.error(f"Exception while handling update: {context.error}")
        await self._increment_stat("errors_today")

        # Try to notify user of error
        if isinstance(update, Update) and update.effective_user:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_user.id,
                    text="❌ An error occurred. Our team has been notified. Please try again later.",
                )
            except Exception:
                pass  # Don't let error handler errors crash the system

    # Bot Control
    async def start(self):
        """Start the multi-user bot"""
        logger.info("Starting multi-user trading bot...")

        await self.initialize()

        # Start the application
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        logger.info("Multi-user bot is now running!")

        return self.application

    async def stop(self):
        """Stop the bot gracefully"""
        logger.info("Stopping multi-user bot...")

        # Stop notification workers
        for worker in self.notification_workers:
            worker.cancel()

        # Stop the application
        if self.application:
            await self.application.stop()

        logger.info("Multi-user bot stopped")

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
multi_user_bot = MultiUserTradingBot
