#!/usr/bin/env python3
"""
Multi-user database system for 24/7 trading service
Supports thousands of concurrent users with individual settings and trades
"""

import sqlite3
import asyncio
import aiosqlite
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import hashlib
import secrets

logger = logging.getLogger(__name__)


@dataclass
class User:
    user_id: int
    telegram_id: int
    username: str
    first_name: str
    last_name: str
    is_active: bool
    subscription_tier: str  # 'free', 'premium', 'enterprise'
    created_at: datetime
    last_active: datetime
    api_key: str
    daily_trade_limit: int
    monthly_trade_count: int


@dataclass
class UserSettings:
    user_id: int
    strategy: str
    exchange: str
    risk_management: Dict
    notifications: Dict
    emergency: Dict
    last_updated: datetime


@dataclass
class UserTrade:
    trade_id: str
    user_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    stop_loss: float
    take_profit: float
    status: str  # 'open', 'closed', 'cancelled'
    pnl: Optional[float]
    commission: float
    created_at: datetime
    closed_at: Optional[datetime]
    signal_confidence: float


@dataclass
class UserPerformance:
    user_id: int
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_commission: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float


class MultiUserDatabase:
    """High-performance multi-user database with connection pooling"""

    def __init__(self, db_path: str = "trading_service.db", max_connections: int = 20):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connection_pool = asyncio.Queue(maxsize=max_connections)
        self._initialized = False

    async def initialize(self):
        """Initialize database with all required tables"""
        if self._initialized:
            return

        logger.info("Initializing multi-user database...")

        # Create initial connection for setup
        async with aiosqlite.connect(self.db_path) as db:
            await self._create_tables(db)
            await self._create_indexes(db)
            await db.commit()

        # Initialize connection pool
        for _ in range(self.max_connections):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            await conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
            await self._connection_pool.put(conn)

        self._initialized = True
        logger.info(f"Database initialized with {self.max_connections} connections")

        # Repair corrupted JSON data on startup (after initialization)
        try:
            repair_results = await self.repair_corrupted_json_data()
            if sum(repair_results.values()) - repair_results["total_users_checked"] > 0:
                logger.info("Database JSON repair completed successfully")
        except Exception as e:
            logger.warning(f"JSON repair failed, but continuing: {e}")

    async def shutdown(self):
        """Properly shutdown database and close all connections"""
        if not self._initialized:
            return

        logger.info("Shutting down database connections...")

        try:
            # Close all connections in the pool
            closed_connections = 0
            while not self._connection_pool.empty():
                try:
                    conn = await asyncio.wait_for(
                        self._connection_pool.get(), timeout=1.0
                    )
                    await conn.close()
                    closed_connections += 1
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for connection from pool during shutdown"
                    )
                    break
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")

            logger.info(f"Closed {closed_connections} database connections")

        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
        finally:
            self._initialized = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self._initialized:
            await self.initialize()

        conn = await self._connection_pool.get()
        try:
            yield conn
        finally:
            await self._connection_pool.put(conn)

    async def _create_tables(self, db):
        """Create all required tables"""

        # Users table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER UNIQUE NOT NULL,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                subscription_tier TEXT DEFAULT 'free',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                api_key TEXT UNIQUE,
                daily_trade_limit INTEGER DEFAULT 10,
                monthly_trade_count INTEGER DEFAULT 0,
                monthly_reset_date DATE DEFAULT CURRENT_DATE
            )
        """
        )

        # User settings table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                strategy TEXT DEFAULT 'RSI_EMA',
                exchange TEXT DEFAULT 'MEXC',
                risk_management TEXT DEFAULT '{}',
                notifications TEXT DEFAULT '{}',
                emergency TEXT DEFAULT '{}',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # User trades table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_trades (
                trade_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                status TEXT DEFAULT 'open',
                pnl REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                signal_confidence REAL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # User performance table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_performance (
                user_id INTEGER NOT NULL,
                date DATE NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                total_commission REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, date),
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # User sessions table (for API authentication)
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # System metrics table (for monitoring)
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # User API usage tracking
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_api_usage (
                user_id INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                request_count INTEGER DEFAULT 1,
                date DATE DEFAULT CURRENT_DATE,
                last_request TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, endpoint, date),
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

    async def _create_indexes(self, db):
        """Create database indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_id)",
            "CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)",
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_trades_user_id ON user_trades(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON user_trades(status)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON user_trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_created_at ON user_trades(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_performance_user_date ON user_performance(user_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_user_date ON user_api_usage(user_id, date)",
        ]

        for index in indexes:
            await db.execute(index)

    # User Management Methods
    async def create_user(
        self,
        telegram_id: int,
        username: str = None,
        first_name: str = None,
        last_name: str = None,
    ) -> User:
        """Create a new user"""
        api_key = self._generate_api_key()

        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                INSERT INTO users (telegram_id, username, first_name, last_name, api_key)
                VALUES (?, ?, ?, ?, ?)
            """,
                (telegram_id, username, first_name, last_name, api_key),
            )

            user_id = cursor.lastrowid
            await db.commit()

            # Create default settings
            await self._create_default_settings(db, user_id)
            await db.commit()

            return await self.get_user_by_id(user_id)

    async def get_user_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """Get user by Telegram ID"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM users WHERE telegram_id = ?
            """,
                (telegram_id,),
            )

            row = await cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by user ID"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM users WHERE user_id = ?
            """,
                (user_id,),
            )

            row = await cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None

    async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM users WHERE api_key = ?
            """,
                (api_key,),
            )

            row = await cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None

    async def update_user_activity(self, user_id: int):
        """Update user's last active timestamp"""
        async with self.get_connection() as db:
            await db.execute(
                """
                UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?
            """,
                (user_id,),
            )
            await db.commit()

    async def get_active_users_count(self) -> int:
        """Get count of active users"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM users WHERE is_active = TRUE
                AND last_active > datetime('now', '-7 days')
            """
            )
            return (await cursor.fetchone())[0]

    # Settings Management
    def _safe_json_loads(self, json_str: str, default_value: dict = None) -> dict:
        """Safely parse JSON string with fallback to default value"""
        if default_value is None:
            default_value = {}

        try:
            if not json_str or json_str.strip() == "":
                return default_value

            # Clean the JSON string - remove extra whitespace and potential trailing chars
            cleaned_json = json_str.strip()

            # Try to find the end of valid JSON by looking for balanced braces
            if cleaned_json.startswith("{"):
                brace_count = 0
                valid_end = 0
                for i, char in enumerate(cleaned_json):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            valid_end = i + 1
                            break

                if valid_end > 0:
                    cleaned_json = cleaned_json[:valid_end]

            # Parse the cleaned JSON
            result = json.loads(cleaned_json)

            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.warning(
                    f"JSON parsing resulted in non-dict type: {type(result)}, using default"
                )
                return default_value

            return result

        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON decode error: {e}. Using default value. Original string: {repr(json_str[:100])}"
            )
            return default_value
        except Exception as e:
            logger.error(
                f"Unexpected error parsing JSON: {e}. Using default value. Original string: {repr(json_str[:100])}"
            )
            return default_value

    def _safe_json_dumps(self, data: dict, field_name: str = "unknown") -> str:
        """Safely serialize dictionary to JSON string"""
        try:
            if not isinstance(data, dict):
                logger.warning(
                    f"Attempting to serialize non-dict data for {field_name}: {type(data)}"
                )
                return "{}"

            # Remove any non-serializable values
            clean_data = {}
            for key, value in data.items():
                try:
                    # Test if the value is JSON serializable
                    json.dumps(value)
                    clean_data[key] = value
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping non-serializable value for {field_name}.{key}: {e}"
                    )

            result = json.dumps(clean_data, ensure_ascii=True, separators=(",", ":"))

            # Validate that we can parse it back
            json.loads(result)

            return result

        except Exception as e:
            logger.error(f"Error serializing JSON for {field_name}: {e}")
            return "{}"

    async def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        """Get user settings with safe JSON parsing"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM user_settings WHERE user_id = ?
            """,
                (user_id,),
            )

            row = await cursor.fetchone()
            if row:
                # Default values for settings in case of JSON parsing errors
                default_risk_management = {
                    "max_risk_per_trade": 0.02,
                    "stop_loss_atr": 2.0,
                    "take_profit_atr": 3.0,
                    "max_open_positions": 5,
                    "emergency_stop": False,
                    "trading_enabled": True,
                }

                default_notifications = {
                    "signal_alerts": True,
                    "trade_execution": True,
                    "risk_warnings": True,
                }

                default_emergency = {
                    "emergency_mode": False,
                    "auto_close_on_loss": False,
                    "max_daily_loss": 0.05,
                }

                try:
                    return UserSettings(
                        user_id=row[0],
                        strategy=row[1] or "RSI_EMA",
                        exchange=row[2] or "MEXC",
                        risk_management=self._safe_json_loads(
                            row[3], default_risk_management
                        ),
                        notifications=self._safe_json_loads(
                            row[4], default_notifications
                        ),
                        emergency=self._safe_json_loads(row[5], default_emergency),
                        last_updated=(
                            datetime.fromisoformat(row[6]) if row[6] else datetime.now()
                        ),
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating UserSettings object for user {user_id}: {e}"
                    )
                    # Return settings with all defaults if object creation fails
                    return UserSettings(
                        user_id=user_id,
                        strategy="RSI_EMA",
                        exchange="MEXC",
                        risk_management=default_risk_management,
                        notifications=default_notifications,
                        emergency=default_emergency,
                        last_updated=datetime.now(),
                    )
            return None

    async def update_user_settings(self, user_id: int, **kwargs):
        """Update user settings with safe JSON handling"""
        async with self.get_connection() as db:
            try:
                settings = await self.get_user_settings(user_id)
                if not settings:
                    await self._create_default_settings(db, user_id)
                    settings = await self.get_user_settings(user_id)

                updates = []
                params = []

                if "strategy" in kwargs:
                    strategy = kwargs["strategy"]
                    # Validate strategy is a string
                    if isinstance(strategy, str) and strategy.strip():
                        updates.append("strategy = ?")
                        params.append(strategy.strip())

                if "exchange" in kwargs:
                    exchange = kwargs["exchange"]
                    # Validate exchange is a string
                    if isinstance(exchange, str) and exchange.strip():
                        updates.append("exchange = ?")
                        params.append(exchange.strip())

                if "risk_management" in kwargs:
                    risk_data = kwargs["risk_management"]
                    if isinstance(risk_data, dict):
                        updates.append("risk_management = ?")
                        params.append(
                            self._safe_json_dumps(risk_data, "risk_management")
                        )
                    else:
                        logger.warning(
                            f"Invalid risk_management data type for user {user_id}: {type(risk_data)}"
                        )

                if "notifications" in kwargs:
                    notif_data = kwargs["notifications"]
                    if isinstance(notif_data, dict):
                        updates.append("notifications = ?")
                        params.append(
                            self._safe_json_dumps(notif_data, "notifications")
                        )
                    else:
                        logger.warning(
                            f"Invalid notifications data type for user {user_id}: {type(notif_data)}"
                        )

                if "emergency" in kwargs:
                    emergency_data = kwargs["emergency"]
                    if isinstance(emergency_data, dict):
                        updates.append("emergency = ?")
                        params.append(
                            self._safe_json_dumps(emergency_data, "emergency")
                        )
                    else:
                        logger.warning(
                            f"Invalid emergency data type for user {user_id}: {type(emergency_data)}"
                        )

                if updates:
                    updates.append("last_updated = CURRENT_TIMESTAMP")
                    params.append(user_id)

                    await db.execute(
                        f"""
                        UPDATE user_settings SET {', '.join(updates)} WHERE user_id = ?
                    """,
                        params,
                    )
                    await db.commit()
                    logger.debug(f"Successfully updated settings for user {user_id}")

            except Exception as e:
                logger.error(f"Error updating user settings for user {user_id}: {e}")
                # Try to rollback the transaction
                try:
                    await db.rollback()
                except:
                    pass
                raise

    # Trade Management
    async def create_trade(
        self,
        user_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        signal_confidence: float = 0.0,
    ) -> str:
        """Create a new trade"""
        trade_id = self._generate_trade_id(user_id, symbol)

        async with self.get_connection() as db:
            await db.execute(
                """
                INSERT INTO user_trades 
                (trade_id, user_id, symbol, side, entry_price, quantity, 
                 stop_loss, take_profit, signal_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade_id,
                    user_id,
                    symbol,
                    side,
                    entry_price,
                    quantity,
                    stop_loss,
                    take_profit,
                    signal_confidence,
                ),
            )
            await db.commit()

            return trade_id

    async def close_trade(
        self, trade_id: str, exit_price: float, commission: float = 0.0
    ):
        """Close a trade"""
        async with self.get_connection() as db:
            # Get trade details
            cursor = await db.execute(
                """
                SELECT user_id, side, entry_price, quantity FROM user_trades 
                WHERE trade_id = ? AND status = 'open'
            """,
                (trade_id,),
            )

            trade_row = await cursor.fetchone()
            if not trade_row:
                return False

            user_id, side, entry_price, quantity = trade_row

            # Calculate PnL
            if side.upper() == "BUY":
                pnl = (exit_price - entry_price) * quantity - commission
            else:
                pnl = (entry_price - exit_price) * quantity - commission

            # Update trade
            await db.execute(
                """
                UPDATE user_trades 
                SET status = 'closed', exit_price = ?, pnl = ?, commission = ?, 
                    closed_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            """,
                (exit_price, pnl, commission, trade_id),
            )

            await db.commit()

            # Update daily performance
            await self._update_daily_performance(db, user_id)
            await db.commit()

            return True

    async def get_user_open_trades(self, user_id: int) -> List[UserTrade]:
        """Get user's open trades"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM user_trades WHERE user_id = ? AND status = 'open'
                ORDER BY created_at DESC
            """,
                (user_id,),
            )

            rows = await cursor.fetchall()
            return [self._row_to_trade(row) for row in rows]

    async def get_user_trade_history(
        self, user_id: int, limit: int = 100
    ) -> List[UserTrade]:
        """Get user's trade history"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM user_trades WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
            """,
                (user_id, limit),
            )

            rows = await cursor.fetchall()
            return [self._row_to_trade(row) for row in rows]

    # Performance Analytics
    async def get_user_performance(
        self, user_id: int, days: int = 30
    ) -> List[UserPerformance]:
        """Get user performance for specified days"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM user_performance 
                WHERE user_id = ? AND date >= date('now', '-{} days')
                ORDER BY date DESC
            """.format(
                    days
                ),
                (user_id,),
            )

            rows = await cursor.fetchall()
            return [self._row_to_performance(row) for row in rows]

    async def get_daily_stats(self, user_id: int) -> Dict:
        """Get today's trading statistics"""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(SUM(commission), 0) as total_commission
                FROM user_trades 
                WHERE user_id = ? AND date(created_at) = date('now') AND status = 'closed'
            """,
                (user_id,),
            )

            row = await cursor.fetchone()
            if row:
                total_trades, winning, losing, pnl, commission = row
                win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

                return {
                    "total_trades": total_trades,
                    "winning_trades": winning,
                    "losing_trades": losing,
                    "win_rate": win_rate,
                    "total_pnl": pnl,
                    "total_commission": commission,
                }

            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_commission": 0,
            }

    # System Monitoring
    async def log_system_metric(self, metric_name: str, value: float):
        """Log system metric for monitoring"""
        async with self.get_connection() as db:
            await db.execute(
                """
                INSERT INTO system_metrics (metric_name, metric_value)
                VALUES (?, ?)
            """,
                (metric_name, value),
            )
            await db.commit()

    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        async with self.get_connection() as db:
            # Active users in last 24 hours
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM users 
                WHERE last_active > datetime('now', '-1 day') AND is_active = TRUE
            """
            )
            active_users_24h = (await cursor.fetchone())[0]

            # Total trades today
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM user_trades 
                WHERE date(created_at) = date('now')
            """
            )
            trades_today = (await cursor.fetchone())[0]

            # Open trades count
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM user_trades WHERE status = 'open'
            """
            )
            open_trades = (await cursor.fetchone())[0]

            return {
                "active_users_24h": active_users_24h,
                "trades_today": trades_today,
                "open_trades": open_trades,
                "timestamp": datetime.now().isoformat(),
            }

    # Helper Methods
    def _generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)

    def _generate_trade_id(self, user_id: int, symbol: str) -> str:
        """Generate unique trade ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = secrets.token_hex(4)
        return f"{user_id}_{symbol}_{timestamp}_{random_part}"

    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row[0],
            telegram_id=row[1],
            username=row[2],
            first_name=row[3],
            last_name=row[4],
            is_active=row[5],
            subscription_tier=row[6],
            created_at=datetime.fromisoformat(row[7]),
            last_active=datetime.fromisoformat(row[8]),
            api_key=row[9],
            daily_trade_limit=row[10],
            monthly_trade_count=row[11],
        )

    def _row_to_trade(self, row) -> UserTrade:
        """Convert database row to UserTrade object"""
        return UserTrade(
            trade_id=row[0],
            user_id=row[1],
            symbol=row[2],
            side=row[3],
            entry_price=row[4],
            exit_price=row[5],
            quantity=row[6],
            stop_loss=row[7],
            take_profit=row[8],
            status=row[9],
            pnl=row[10],
            commission=row[11],
            created_at=datetime.fromisoformat(row[12]),
            closed_at=datetime.fromisoformat(row[13]) if row[13] else None,
            signal_confidence=row[14],
        )

    def _row_to_performance(self, row) -> UserPerformance:
        """Convert database row to UserPerformance object"""
        return UserPerformance(
            user_id=row[0],
            date=row[1],
            total_trades=row[2],
            winning_trades=row[3],
            losing_trades=row[4],
            total_pnl=row[5],
            total_commission=row[6],
            win_rate=row[7],
            avg_win=row[8],
            avg_loss=row[9],
            max_drawdown=row[10],
            sharpe_ratio=row[11],
        )

    async def _create_default_settings(self, db, user_id: int):
        """Create default settings for new user with safe JSON"""
        default_risk = {
            "max_risk_per_trade": 0.02,
            "stop_loss_atr": 2.0,
            "take_profit_atr": 3.0,
            "max_open_positions": 5,
            "emergency_stop": False,
            "trading_enabled": True,
            "custom_stop_loss": None,
            "custom_take_profit": None,
        }

        default_notifications = {
            "signal_alerts": True,
            "trade_execution": True,
            "risk_warnings": True,
        }

        default_emergency = {
            "emergency_mode": False,
            "auto_close_on_loss": False,
            "max_daily_loss": 0.05,
        }

        await db.execute(
            """
            INSERT INTO user_settings (user_id, exchange, risk_management, notifications, emergency)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                user_id,
                "MEXC",  # Default exchange
                self._safe_json_dumps(default_risk, "default_risk"),
                self._safe_json_dumps(default_notifications, "default_notifications"),
                self._safe_json_dumps(default_emergency, "default_emergency"),
            ),
        )

    async def repair_corrupted_json_data(self) -> Dict[str, int]:
        """Repair any corrupted JSON data in user_settings table"""
        logger.info("Starting JSON data repair process...")

        repaired_counts = {
            "risk_management": 0,
            "notifications": 0,
            "emergency": 0,
            "total_users_checked": 0,
        }

        async with self.get_connection() as db:
            # Get all user settings
            cursor = await db.execute(
                "SELECT user_id, risk_management, notifications, emergency FROM user_settings"
            )
            all_settings = await cursor.fetchall()

            for row in all_settings:
                user_id, risk_mgmt_json, notifications_json, emergency_json = row
                repaired_counts["total_users_checked"] += 1
                needs_update = False

                # Test and repair risk_management
                risk_data = self._safe_json_loads(risk_mgmt_json)
                if not risk_data or risk_mgmt_json != self._safe_json_dumps(
                    risk_data, "risk_management"
                ):
                    default_risk = {
                        "max_risk_per_trade": 0.02,
                        "stop_loss_atr": 2.0,
                        "take_profit_atr": 3.0,
                        "max_open_positions": 5,
                        "emergency_stop": False,
                        "trading_enabled": True,
                    }
                    await db.execute(
                        "UPDATE user_settings SET risk_management = ? WHERE user_id = ?",
                        (
                            self._safe_json_dumps(default_risk, "risk_management"),
                            user_id,
                        ),
                    )
                    repaired_counts["risk_management"] += 1
                    needs_update = True

                # Test and repair notifications
                notif_data = self._safe_json_loads(notifications_json)
                if not notif_data or notifications_json != self._safe_json_dumps(
                    notif_data, "notifications"
                ):
                    default_notifications = {
                        "signal_alerts": True,
                        "trade_execution": True,
                        "risk_warnings": True,
                    }
                    await db.execute(
                        "UPDATE user_settings SET notifications = ? WHERE user_id = ?",
                        (
                            self._safe_json_dumps(
                                default_notifications, "notifications"
                            ),
                            user_id,
                        ),
                    )
                    repaired_counts["notifications"] += 1
                    needs_update = True

                # Test and repair emergency
                emergency_data = self._safe_json_loads(emergency_json)
                if not emergency_data or emergency_json != self._safe_json_dumps(
                    emergency_data, "emergency"
                ):
                    default_emergency = {
                        "emergency_mode": False,
                        "auto_close_on_loss": False,
                        "max_daily_loss": 0.05,
                    }
                    await db.execute(
                        "UPDATE user_settings SET emergency = ? WHERE user_id = ?",
                        (
                            self._safe_json_dumps(default_emergency, "emergency"),
                            user_id,
                        ),
                    )
                    repaired_counts["emergency"] += 1
                    needs_update = True

                if needs_update:
                    await db.execute(
                        "UPDATE user_settings SET last_updated = CURRENT_TIMESTAMP WHERE user_id = ?",
                        (user_id,),
                    )

            await db.commit()

        total_repairs = (
            repaired_counts["risk_management"]
            + repaired_counts["notifications"]
            + repaired_counts["emergency"]
        )
        logger.info(
            f"JSON repair completed: {total_repairs} fields repaired across {repaired_counts['total_users_checked']} users"
        )

        return repaired_counts

    async def _update_daily_performance(self, db, user_id: int):
        """Update daily performance metrics"""
        today = datetime.now().date().isoformat()

        # Calculate daily stats
        cursor = await db.execute(
            """
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(commission), 0) as total_commission,
                COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN pnl < 0 THEN pnl END), 0) as avg_loss
            FROM user_trades 
            WHERE user_id = ? AND date(created_at) = ? AND status = 'closed'
        """,
            (user_id, today),
        )

        row = await cursor.fetchone()
        if row:
            total, winning, losing, pnl, commission, avg_win, avg_loss = row
            win_rate = (winning / total * 100) if total > 0 else 0

            # Upsert performance record
            await db.execute(
                """
                INSERT OR REPLACE INTO user_performance 
                (user_id, date, total_trades, winning_trades, losing_trades, 
                 total_pnl, total_commission, win_rate, avg_win, avg_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    today,
                    total,
                    winning,
                    losing,
                    pnl,
                    commission,
                    win_rate,
                    avg_win,
                    avg_loss,
                ),
            )


# Global database instance
multi_user_db = MultiUserDatabase()
