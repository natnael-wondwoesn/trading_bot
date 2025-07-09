#!/usr/bin/env python3
"""
User Management Service for Multi-User Trading Platform
Handles user registration, authentication, subscription management, and rate limiting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import secrets
import jwt
from telegram import Update, User as TelegramUser
from db.multi_user_db import multi_user_db, User, UserSettings

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    session_id: str
    user_id: int
    telegram_id: int
    expires_at: datetime
    permissions: List[str]


@dataclass
class SubscriptionLimits:
    daily_trades: int
    monthly_trades: int
    concurrent_positions: int
    api_calls_per_minute: int
    features: List[str]


class UserService:
    """Comprehensive user management service"""

    # Subscription tiers and their limits
    SUBSCRIPTION_LIMITS = {
        "free": SubscriptionLimits(
            daily_trades=5,
            monthly_trades=50,
            concurrent_positions=2,
            api_calls_per_minute=10,
            features=["basic_signals", "basic_stats"],
        ),
        "premium": SubscriptionLimits(
            daily_trades=25,
            monthly_trades=500,
            concurrent_positions=5,
            api_calls_per_minute=30,
            features=[
                "basic_signals",
                "advanced_signals",
                "detailed_stats",
                "custom_alerts",
            ],
        ),
        "enterprise": SubscriptionLimits(
            daily_trades=100,
            monthly_trades=2000,
            concurrent_positions=20,
            api_calls_per_minute=100,
            features=[
                "all_signals",
                "advanced_analytics",
                "priority_support",
                "custom_strategies",
            ],
        ),
    }

    def __init__(self, jwt_secret: str = None):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.active_sessions: Dict[str, UserSession] = {}
        self.rate_limits: Dict[int, Dict] = {}  # user_id -> rate limit data

    async def initialize(self):
        """Initialize the user service"""
        await multi_user_db.initialize()
        logger.info("User service initialized")

    # User Registration and Authentication
    async def register_or_login_user(
        self, telegram_user: TelegramUser
    ) -> Tuple[User, bool]:
        """Register new user or login existing user"""
        user = await multi_user_db.get_user_by_telegram_id(telegram_user.id)
        is_new_user = False

        if not user:
            # Register new user
            user = await multi_user_db.create_user(
                telegram_id=telegram_user.id,
                username=telegram_user.username,
                first_name=telegram_user.first_name,
                last_name=telegram_user.last_name,
            )
            is_new_user = True
            logger.info(f"New user registered: {user.telegram_id} ({user.username})")

            # Send welcome analytics
            await multi_user_db.log_system_metric("new_user_registration", 1)
        else:
            # Update last active
            await multi_user_db.update_user_activity(user.user_id)

        return user, is_new_user

    async def create_user_session(
        self, user: User, duration_hours: int = 24
    ) -> UserSession:
        """Create authenticated session for user"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)

        # Get user permissions based on subscription
        permissions = self._get_user_permissions(user.subscription_tier)

        session = UserSession(
            session_id=session_id,
            user_id=user.user_id,
            telegram_id=user.telegram_id,
            expires_at=expires_at,
            permissions=permissions,
        )

        # Store session
        self.active_sessions[session_id] = session

        logger.info(f"Session created for user {user.user_id}: {session_id}")
        return session

    async def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate and return session if valid"""
        session = self.active_sessions.get(session_id)

        if not session:
            return None

        if datetime.now() > session.expires_at:
            # Session expired
            del self.active_sessions[session_id]
            return None

        return session

    async def get_user_by_session(self, session_id: str) -> Optional[User]:
        """Get user by session ID"""
        session = await self.validate_session(session_id)
        if session:
            return await multi_user_db.get_user_by_id(session.user_id)
        return None

    # Subscription Management
    async def upgrade_subscription(self, user_id: int, new_tier: str) -> bool:
        """Upgrade user subscription"""
        if new_tier not in self.SUBSCRIPTION_LIMITS:
            return False

        async with multi_user_db.get_connection() as db:
            await db.execute(
                """
                UPDATE users SET subscription_tier = ? WHERE user_id = ?
            """,
                (new_tier, user_id),
            )
            await db.commit()

        logger.info(f"User {user_id} upgraded to {new_tier}")
        await multi_user_db.log_system_metric(f"subscription_upgrade_{new_tier}", 1)
        return True

    def get_subscription_limits(self, user: User) -> SubscriptionLimits:
        """Get subscription limits for user"""
        return self.SUBSCRIPTION_LIMITS.get(
            user.subscription_tier, self.SUBSCRIPTION_LIMITS["free"]
        )

    async def check_user_limits(self, user: User, action: str) -> Tuple[bool, str]:
        """Check if user can perform action within limits"""
        limits = self.get_subscription_limits(user)

        if action == "daily_trade":
            # Check daily trade limit
            daily_stats = await multi_user_db.get_daily_stats(user.user_id)
            if daily_stats["total_trades"] >= limits.daily_trades:
                return False, f"Daily trade limit reached ({limits.daily_trades})"

        elif action == "open_position":
            # Check concurrent positions
            open_trades = await multi_user_db.get_user_open_trades(user.user_id)
            if len(open_trades) >= limits.concurrent_positions:
                return (
                    False,
                    f"Maximum concurrent positions reached ({limits.concurrent_positions})",
                )

        elif action == "api_call":
            # Check API rate limit
            if not await self._check_rate_limit(
                user.user_id, limits.api_calls_per_minute
            ):
                return (
                    False,
                    f"API rate limit exceeded ({limits.api_calls_per_minute}/min)",
                )

        return True, "OK"

    async def _check_rate_limit(self, user_id: int, max_calls_per_minute: int) -> bool:
        """Check API rate limit for user"""
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d %H:%M")

        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {}

        user_limits = self.rate_limits[user_id]

        # Clean old entries
        old_keys = [
            k
            for k in user_limits.keys()
            if k < (now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
        ]
        for key in old_keys:
            del user_limits[key]

        # Check current minute
        current_calls = user_limits.get(minute_key, 0)
        if current_calls >= max_calls_per_minute:
            return False

        # Increment counter
        user_limits[minute_key] = current_calls + 1
        return True

    def _get_user_permissions(self, subscription_tier: str) -> List[str]:
        """Get permissions based on subscription tier"""
        base_permissions = ["view_signals", "basic_trading", "view_stats"]

        if subscription_tier == "premium":
            return base_permissions + [
                "advanced_signals",
                "custom_alerts",
                "detailed_analytics",
            ]
        elif subscription_tier == "enterprise":
            return base_permissions + [
                "advanced_signals",
                "custom_alerts",
                "detailed_analytics",
                "custom_strategies",
                "priority_support",
                "api_access",
            ]

        return base_permissions

    # User Settings Management
    async def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        """Get user's trading settings"""
        return await multi_user_db.get_user_settings(user_id)

    async def update_user_settings(self, user_id: int, **settings) -> bool:
        """Update user's trading settings"""
        try:
            await multi_user_db.update_user_settings(user_id, **settings)
            return True
        except Exception as e:
            logger.error(f"Error updating settings for user {user_id}: {e}")
            return False

    async def reset_user_settings(self, user_id: int) -> bool:
        """Reset user settings to defaults"""
        default_settings = {
            "strategy": "RSI_EMA",
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "stop_loss_atr": 2.0,
                "take_profit_atr": 3.0,
                "max_open_positions": 5,
                "emergency_stop": False,
                "trading_enabled": True,
                "custom_stop_loss": None,
                "custom_take_profit": None,
            },
            "notifications": {
                "signal_alerts": True,
                "trade_execution": True,
                "risk_warnings": True,
            },
            "emergency": {
                "emergency_mode": False,
                "auto_close_on_loss": False,
                "max_daily_loss": 0.05,
            },
        }

        return await self.update_user_settings(user_id, **default_settings)

    # User Analytics and Monitoring
    async def get_user_dashboard_data(self, user_id: int) -> Dict:
        """Get comprehensive dashboard data for user"""
        user = await multi_user_db.get_user_by_id(user_id)
        if not user:
            return {}

        # Get daily stats
        daily_stats = await multi_user_db.get_daily_stats(user_id)

        # Get open trades
        open_trades = await multi_user_db.get_user_open_trades(user_id)

        # Get recent performance
        performance = await multi_user_db.get_user_performance(user_id, days=7)

        # Get subscription info
        limits = self.get_subscription_limits(user)

        # Calculate usage percentages
        daily_usage = (daily_stats["total_trades"] / limits.daily_trades) * 100
        position_usage = (len(open_trades) / limits.concurrent_positions) * 100

        return {
            "user_info": {
                "user_id": user.user_id,
                "telegram_id": user.telegram_id,
                "username": user.username,
                "subscription_tier": user.subscription_tier,
                "member_since": user.created_at.isoformat(),
                "last_active": user.last_active.isoformat(),
            },
            "daily_stats": daily_stats,
            "open_trades": len(open_trades),
            "subscription_limits": {
                "daily_trades": limits.daily_trades,
                "concurrent_positions": limits.concurrent_positions,
                "features": limits.features,
            },
            "usage": {
                "daily_trades_used": daily_usage,
                "positions_used": position_usage,
            },
            "performance_summary": {
                "total_days": len(performance),
                "avg_daily_pnl": (
                    sum(p.total_pnl for p in performance) / len(performance)
                    if performance
                    else 0
                ),
                "total_trades_7d": sum(p.total_trades for p in performance),
                "win_rate_7d": (
                    sum(p.win_rate for p in performance) / len(performance)
                    if performance
                    else 0
                ),
            },
        }

    async def get_user_activity_log(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user activity log"""
        trades = await multi_user_db.get_user_trade_history(user_id, limit)

        activity = []
        for trade in trades:
            activity.append(
                {
                    "type": "trade",
                    "action": f"{trade.side} {trade.symbol}",
                    "status": trade.status,
                    "pnl": trade.pnl,
                    "timestamp": trade.created_at.isoformat(),
                    "details": {
                        "trade_id": trade.trade_id,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "quantity": trade.quantity,
                    },
                }
            )

        return activity

    # Admin Functions
    async def get_system_overview(self) -> Dict:
        """Get system-wide overview (admin function)"""
        health = await multi_user_db.get_system_health()

        # Get subscription distribution
        async with multi_user_db.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT subscription_tier, COUNT(*) 
                FROM users WHERE is_active = TRUE
                GROUP BY subscription_tier
            """
            )
            subscription_dist = dict(await cursor.fetchall())

        return {
            "system_health": health,
            "subscription_distribution": subscription_dist,
            "active_sessions": len(self.active_sessions),
            "timestamp": datetime.now().isoformat(),
        }

    async def get_user_list(
        self, limit: int = 100, subscription_tier: str = None
    ) -> List[Dict]:
        """Get list of users (admin function)"""
        async with multi_user_db.get_connection() as db:
            query = """
                SELECT user_id, telegram_id, username, first_name, subscription_tier, 
                       created_at, last_active, is_active
                FROM users 
                WHERE 1=1
            """
            params = []

            if subscription_tier:
                query += " AND subscription_tier = ?"
                params.append(subscription_tier)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            return [
                {
                    "user_id": row[0],
                    "telegram_id": row[1],
                    "username": row[2],
                    "first_name": row[3],
                    "subscription_tier": row[4],
                    "created_at": row[5],
                    "last_active": row[6],
                    "is_active": row[7],
                }
                for row in rows
            ]

    async def deactivate_user(self, user_id: int, reason: str = "admin_action") -> bool:
        """Deactivate user account"""
        async with multi_user_db.get_connection() as db:
            await db.execute(
                """
                UPDATE users SET is_active = FALSE WHERE user_id = ?
            """,
                (user_id,),
            )
            await db.commit()

        # Remove active sessions
        sessions_to_remove = [
            session_id
            for session_id, session in self.active_sessions.items()
            if session.user_id == user_id
        ]
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]

        logger.info(f"User {user_id} deactivated: {reason}")
        return True

    # Cleanup and Maintenance
    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_sessions = [
            session_id
            for session_id, session in self.active_sessions.items()
            if session.expires_at < now
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def reset_monthly_counters(self):
        """Reset monthly trade counters (run monthly)"""
        async with multi_user_db.get_connection() as db:
            await db.execute(
                """
                UPDATE users SET 
                    monthly_trade_count = 0,
                    monthly_reset_date = CURRENT_DATE
                WHERE monthly_reset_date < date('now', 'start of month')
            """
            )
            await db.commit()

        logger.info("Monthly counters reset")


# Global user service instance
user_service = UserService()
