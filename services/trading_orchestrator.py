#!/usr/bin/env python3
"""
Trading Orchestrator - Manages isolated trading sessions for thousands of users
Each user gets their own isolated trading environment with individual settings and risk management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
from concurrent.futures import ThreadPoolExecutor
import threading

from metaapi_integration.strategies.forex_strategy import ForexStrategy
from services.user_service import user_service, User
from services.multi_user_bot import multi_user_bot
from db.multi_user_db import multi_user_db, UserTrade
from models.models import Signal
from strategy.strategies.bollinger_strategy import BollingerStrategy
from strategy.strategies.macd_strategy import MACDStrategy
from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
from strategy.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass
class UserTradingSession:
    """Isolated trading session for each user"""

    user_id: int
    telegram_id: int
    strategy: Strategy
    settings: Dict
    risk_manager: "UserRiskManager"
    active_trades: Dict[str, UserTrade] = field(default_factory=dict)
    daily_pnl: float = 0.0
    session_start: datetime = field(default_factory=datetime.now)
    last_signal: Optional[Signal] = None
    last_trade: Optional[datetime] = None
    is_active: bool = True
    emergency_mode: bool = False
    trading_enabled: bool = True
    performance_metrics: Dict = field(default_factory=dict)


class UserRiskManager:
    """Per-user risk management system"""

    def __init__(self, user_id: int, settings: Dict):
        self.user_id = user_id
        self.settings = settings
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.last_reset = datetime.now().date()

    async def check_trade_allowed(self, signal: Signal) -> Tuple[bool, str]:
        """Check if trade is allowed based on risk rules"""

        # Check daily reset
        await self._check_daily_reset()

        # Get user limits
        user = await multi_user_db.get_user_by_id(self.user_id)
        if not user:
            return False, "User not found"

        limits = user_service.get_subscription_limits(user)

        # Check emergency mode
        if self.settings.get("emergency", {}).get("emergency_mode", False):
            return False, "Emergency mode active"

        # Check if trading is enabled
        if not self.settings.get("risk_management", {}).get("trading_enabled", True):
            return False, "Trading disabled in settings"

        # Check daily trade limit
        if self.daily_trades >= limits.daily_trades:
            return False, f"Daily trade limit reached ({limits.daily_trades})"

        # Check max open positions
        open_trades = await multi_user_db.get_user_open_trades(self.user_id)
        max_positions = self.settings.get("risk_management", {}).get(
            "max_open_positions", 5
        )

        if len(open_trades) >= min(max_positions, limits.concurrent_positions):
            return False, f"Maximum positions reached ({max_positions})"

        # Check daily loss limit
        max_daily_loss = self.settings.get("emergency", {}).get("max_daily_loss", 0.05)
        if abs(self.daily_pnl) > max_daily_loss:
            return False, f"Daily loss limit exceeded ({max_daily_loss*100}%)"

        # Check consecutive losses
        if self.consecutive_losses >= 5:
            return False, "Too many consecutive losses (5). Please review strategy."

        # Check signal confidence
        min_confidence = self.settings.get("risk_management", {}).get(
            "min_signal_confidence", 0.6
        )
        if signal.confidence < min_confidence:
            return (
                False,
                f"Signal confidence too low ({signal.confidence:.2f} < {min_confidence:.2f})",
            )

        return True, "Trade allowed"

    async def calculate_position_size(
        self, signal: Signal, account_balance: float
    ) -> float:
        """Calculate position size based on risk management rules"""

        max_risk_per_trade = self.settings.get("risk_management", {}).get(
            "max_risk_per_trade", 0.02
        )

        # Calculate risk amount
        risk_amount = account_balance * max_risk_per_trade

        # Calculate stop loss distance
        stop_loss_atr = self.settings.get("risk_management", {}).get(
            "stop_loss_atr", 2.0
        )
        stop_distance = signal.atr * stop_loss_atr

        # Position size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance

        # Apply minimum and maximum position limits
        min_position = 0.01  # Minimum position size
        max_position = account_balance * 0.1  # Maximum 10% of balance

        position_size = max(min_position, min(position_size, max_position))

        return round(position_size, 2)

    async def update_after_trade(self, trade: UserTrade):
        """Update risk metrics after trade completion"""
        if trade.pnl:
            self.daily_pnl += trade.pnl

            if trade.pnl < 0:
                self.consecutive_losses += 1
                # Update max drawdown
                if abs(self.daily_pnl) > self.max_drawdown:
                    self.max_drawdown = abs(self.daily_pnl)
            else:
                self.consecutive_losses = 0

        self.daily_trades += 1

    async def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_reset = today


class StrategyFactory:
    """Factory for creating strategy instances"""

    STRATEGIES = {
        "RSI_EMA": RSIEMAStrategy,
        "MACD": "MACDStrategy",
        "BOLLINGER": "BollingerStrategy",
        "FOREX": "ForexStrategy",
    }

    STRATEGY_DESCRIPTIONS = {
        "RSI_EMA": "RSI + EMA - Combines RSI oversold/overbought levels with EMA trend confirmation",
        "MACD": "MACD Strategy - Uses MACD crossovers and momentum for signal generation",
        "BOLLINGER": "Bollinger Bands - Mean reversion and breakout strategy using Bollinger Bands",
        "FOREX": "Forex Strategy - Specialized strategy for forex pairs with spread filtering",
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, settings: Dict) -> Strategy:
        """Create strategy instance based on name"""
        strategy_class = cls.STRATEGIES.get(strategy_name)
        if not strategy_class:
            # Default to RSI_EMA
            strategy_class = RSIEMAStrategy
        elif isinstance(strategy_class, str):
            # Import strategy classes dynamically
            if strategy_class == "MACDStrategy":
                from strategy.strategies.macd_strategy import MACDStrategy

                strategy_class = MACDStrategy
            elif strategy_class == "BollingerStrategy":
                from strategy.strategies.bollinger_strategy import BollingerStrategy

                strategy_class = BollingerStrategy
            elif strategy_class == "ForexStrategy":
                from metaapi_integration.strategies.forex_strategy import ForexStrategy

                strategy_class = ForexStrategy

        return strategy_class()

    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """Get list of available strategies with descriptions"""
        return cls.STRATEGY_DESCRIPTIONS


class TradingOrchestrator:
    """Main orchestrator managing all user trading sessions"""

    def __init__(self, max_concurrent_users: int = 1000):
        self.max_concurrent_users = max_concurrent_users
        self.user_sessions: Dict[int, UserTradingSession] = {}
        self.session_locks: Dict[int, asyncio.Lock] = {}

        # Performance monitoring
        self.performance_stats = {
            "total_signals_processed": 0,
            "total_trades_executed": 0,
            "total_users_active": 0,
            "avg_processing_time": 0.0,
            "error_count": 0,
        }

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)

        # Signal distribution
        self.signal_queue = asyncio.Queue(maxsize=10000)
        self.signal_workers = []
        self.signal_worker_count = 8

        # Active monitoring
        self.is_running = False

    async def start(self):
        """Start the trading orchestrator"""
        await self.initialize()

    async def initialize(self):
        """Initialize the trading orchestrator"""
        logger.info("Initializing Trading Orchestrator...")

        # Start signal processing workers
        for i in range(self.signal_worker_count):
            worker = asyncio.create_task(self._signal_worker(f"signal-worker-{i}"))
            self.signal_workers.append(worker)

        # Start monitoring tasks
        asyncio.create_task(self._session_cleanup_task())
        asyncio.create_task(self._performance_monitoring_task())
        asyncio.create_task(self._risk_monitoring_task())

        self.is_running = True
        logger.info(
            f"Trading Orchestrator initialized with {self.signal_worker_count} workers"
        )

    async def get_or_create_session(
        self, user_id: int, telegram_id: int
    ) -> UserTradingSession:
        """Get existing or create new trading session for user"""

        if user_id not in self.session_locks:
            self.session_locks[user_id] = asyncio.Lock()

        async with self.session_locks[user_id]:
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                session.is_active = True
                return session

            # Create new session
            user = await multi_user_db.get_user_by_id(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")

            settings = await user_service.get_user_settings(user_id)
            if not settings:
                # Create default settings
                await user_service.reset_user_settings(user_id)
                settings = await user_service.get_user_settings(user_id)

            # Create strategy instance
            strategy_name = getattr(settings, "trading_strategy", "RSI_EMA")
            strategy = StrategyFactory.create_strategy(strategy_name, settings.__dict__)

            # Create risk manager
            risk_manager = UserRiskManager(user_id, settings.__dict__)

            # Load active trades
            active_trades = await multi_user_db.get_user_open_trades(user_id)
            trades_dict = {trade.trade_id: trade for trade in active_trades}

            session = UserTradingSession(
                user_id=user_id,
                telegram_id=telegram_id,
                strategy=strategy,
                settings=settings.__dict__,
                risk_manager=risk_manager,
                active_trades=trades_dict,
            )

            self.user_sessions[user_id] = session
            logger.info(f"Created trading session for user {user_id}")

            return session

    async def process_market_signal(
        self, symbol: str, market_data: Dict
    ) -> List[Signal]:
        """Process market data and generate signals for all active users"""
        start_time = datetime.now()
        signals_generated = []

        try:
            # Get all active user sessions
            active_sessions = [
                session
                for session in self.user_sessions.values()
                if session.is_active
                and session.trading_enabled
                and not session.emergency_mode
            ]

            # Process signals for each user in parallel
            tasks = []
            for session in active_sessions:
                task = asyncio.create_task(
                    self._process_user_signal(session, symbol, market_data)
                )
                tasks.append(task)

            # Wait for all signals to be processed
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful signals
            for result in results:
                if isinstance(result, Signal):
                    signals_generated.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error processing user signal: {result}")
                    self.performance_stats["error_count"] += 1

            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["total_signals_processed"] += len(signals_generated)
            self.performance_stats["avg_processing_time"] = (
                self.performance_stats["avg_processing_time"] * 0.9
            ) + (processing_time * 0.1)

            logger.info(
                f"Processed {len(signals_generated)} signals for {symbol} in {processing_time:.3f}s"
            )

        except Exception as e:
            logger.error(f"Error in market signal processing: {e}")
            self.performance_stats["error_count"] += 1

        return signals_generated

    async def _process_user_signal(
        self, session: UserTradingSession, symbol: str, market_data: Dict
    ) -> Optional[Signal]:
        """Process signal for individual user"""
        try:
            # Check if user can trade this symbol
            allowed, reason = await session.risk_manager.check_trade_allowed(
                Signal(
                    symbol=symbol, action="", confidence=1.0, timestamp=datetime.now()
                )  # Dummy signal for checking
            )

            if not allowed:
                return None

            # Generate signal using user's strategy
            signal = await self._run_strategy_analysis(
                session.strategy, symbol, market_data
            )

            if signal and signal.action != "HOLD":
                # Add to signal queue for execution
                await self.signal_queue.put({"session": session, "signal": signal})

                return signal

        except Exception as e:
            logger.error(f"Error processing signal for user {session.user_id}: {e}")

        return None

    async def _run_strategy_analysis(
        self, strategy: Strategy, symbol: str, market_data: Dict
    ) -> Optional[Signal]:
        """Run strategy analysis in thread pool to avoid blocking"""
        loop = asyncio.get_event_loop()

        try:
            # Run CPU-intensive strategy analysis in thread pool
            signal = await loop.run_in_executor(
                self.thread_pool,
                self._analyze_with_strategy,
                strategy,
                symbol,
                market_data,
            )
            return signal

        except Exception as e:
            logger.error(f"Error in strategy analysis: {e}")
            return None

    def _analyze_with_strategy(
        self, strategy: Strategy, symbol: str, market_data: Dict
    ) -> Optional[Signal]:
        """CPU-intensive strategy analysis (runs in thread pool)"""
        try:
            # Convert market data to format expected by strategy
            # This would depend on your specific strategy implementation
            signal = strategy.analyze(market_data)
            return signal
        except Exception as e:
            logger.error(f"Strategy analysis error: {e}")
            return None

    async def _signal_worker(self, worker_name: str):
        """Background worker for processing trading signals"""
        logger.info(f"Signal worker {worker_name} started")

        while self.is_running:
            try:
                # Get signal from queue
                signal_data = await self.signal_queue.get()

                # Process the signal
                await self._execute_signal(
                    signal_data["session"], signal_data["signal"]
                )

                # Mark task as done
                self.signal_queue.task_done()

            except Exception as e:
                logger.error(f"Error in signal worker {worker_name}: {e}")
                await asyncio.sleep(1)

    async def _execute_signal(self, session: UserTradingSession, signal: Signal):
        """Execute trading signal for user"""
        try:
            # Final risk check
            allowed, reason = await session.risk_manager.check_trade_allowed(signal)

            if not allowed:
                logger.info(f"Trade blocked for user {session.user_id}: {reason}")
                return

            # Calculate position size
            # Note: You'd need to implement account balance tracking
            account_balance = 10000.0  # Placeholder - implement proper balance tracking
            position_size = await session.risk_manager.calculate_position_size(
                signal, account_balance
            )

            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_profit_levels(
                signal, session.settings
            )

            # Create trade record
            trade_id = await multi_user_db.create_trade(
                user_id=session.user_id,
                symbol=signal.symbol,
                side=signal.action,
                entry_price=signal.price,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_confidence=signal.confidence,
            )

            # Update session
            session.last_signal = signal
            session.last_trade = datetime.now()

            # Send notification to user
            await multi_user_bot.send_signal_to_user(session.telegram_id, signal)

            # Update performance stats
            self.performance_stats["total_trades_executed"] += 1

            logger.info(
                f"Executed trade {trade_id} for user {session.user_id}: {signal.action} {signal.symbol}"
            )

        except Exception as e:
            logger.error(f"Error executing signal for user {session.user_id}: {e}")

    def _calculate_stop_profit_levels(
        self, signal: Signal, settings: Dict
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        risk_settings = settings.get("risk_management", {})

        # Custom levels if set
        if risk_settings.get("custom_stop_loss"):
            stop_loss = signal.price * (1 - risk_settings["custom_stop_loss"] / 100)
        else:
            # ATR-based stop loss
            stop_loss_atr = risk_settings.get("stop_loss_atr", 2.0)
            if signal.action.upper() == "BUY":
                stop_loss = signal.price - (signal.atr * stop_loss_atr)
            else:
                stop_loss = signal.price + (signal.atr * stop_loss_atr)

        if risk_settings.get("custom_take_profit"):
            take_profit = signal.price * (1 + risk_settings["custom_take_profit"] / 100)
        else:
            # ATR-based take profit
            take_profit_atr = risk_settings.get("take_profit_atr", 3.0)
            if signal.action.upper() == "BUY":
                take_profit = signal.price + (signal.atr * take_profit_atr)
            else:
                take_profit = signal.price - (signal.atr * take_profit_atr)

        return round(stop_loss, 5), round(take_profit, 5)

    # User Management
    async def update_user_settings(self, user_id: int):
        """Update user session when settings change"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]

            # Reload settings
            settings = await user_service.get_user_settings(user_id)
            if settings:
                session.settings = settings.__dict__

                # Update strategy if changed
                current_strategy = getattr(settings, "trading_strategy", "RSI_EMA")
                if current_strategy != session.strategy.__class__.__name__:
                    session.strategy = StrategyFactory.create_strategy(
                        current_strategy, settings.__dict__
                    )

                # Update risk manager
                session.risk_manager.settings = settings.__dict__

                logger.info(f"Updated settings for user {user_id}")

    async def activate_emergency_mode(self, user_id: int):
        """Activate emergency mode for user"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            session.emergency_mode = True
            session.trading_enabled = False

            # Close all open positions (implement based on your exchange integration)
            await self._close_all_user_positions(user_id)

            logger.warning(f"Emergency mode activated for user {user_id}")

    async def _close_all_user_positions(self, user_id: int):
        """Close all open positions for user (emergency)"""
        open_trades = await multi_user_db.get_user_open_trades(user_id)

        for trade in open_trades:
            # Implement position closing logic based on your exchange
            # For now, just mark as closed in database
            await multi_user_db.close_trade(
                trade_id=trade.trade_id,
                exit_price=trade.entry_price,  # Emergency close at current price
                commission=0.0,
            )

        logger.info(f"Closed {len(open_trades)} positions for user {user_id}")

    # Monitoring and Maintenance
    async def _session_cleanup_task(self):
        """Clean up inactive sessions"""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=4)
                inactive_users = []

                for user_id, session in self.user_sessions.items():
                    if (
                        session.session_start < cutoff_time
                        and not session.active_trades
                    ):
                        inactive_users.append(user_id)

                for user_id in inactive_users:
                    if user_id in self.user_sessions:
                        del self.user_sessions[user_id]
                    if user_id in self.session_locks:
                        del self.session_locks[user_id]

                if inactive_users:
                    logger.info(f"Cleaned up {len(inactive_users)} inactive sessions")

                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(300)

    async def _performance_monitoring_task(self):
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                self.performance_stats["total_users_active"] = len(self.user_sessions)

                # Log metrics to database
                await multi_user_db.log_system_metric(
                    "orchestrator_active_users",
                    self.performance_stats["total_users_active"],
                )
                await multi_user_db.log_system_metric(
                    "orchestrator_signals_processed",
                    self.performance_stats["total_signals_processed"],
                )
                await multi_user_db.log_system_metric(
                    "orchestrator_trades_executed",
                    self.performance_stats["total_trades_executed"],
                )
                await multi_user_db.log_system_metric(
                    "orchestrator_avg_processing_time",
                    self.performance_stats["avg_processing_time"],
                )

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

    async def _risk_monitoring_task(self):
        """Monitor system-wide risk metrics"""
        while self.is_running:
            try:
                # Monitor for users with high losses
                high_risk_users = []

                for user_id, session in self.user_sessions.items():
                    if session.risk_manager.daily_pnl < -500:  # $500 daily loss
                        high_risk_users.append(user_id)

                # Alert admins about high-risk users
                if high_risk_users:
                    logger.warning(f"High-risk users detected: {high_risk_users}")
                    await multi_user_db.log_system_metric(
                        "high_risk_users", len(high_risk_users)
                    )

                await asyncio.sleep(600)  # 10 minutes

            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(120)

    def get_system_stats(self) -> Dict:
        """Get orchestrator performance statistics"""
        return {
            **self.performance_stats,
            "active_sessions": len(self.user_sessions),
            "signal_queue_size": self.signal_queue.qsize(),
            "max_concurrent_users": self.max_concurrent_users,
            "is_running": self.is_running,
        }

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Trading Orchestrator...")

        self.is_running = False

        # Cancel all workers
        for worker in self.signal_workers:
            worker.cancel()

        # Wait for signal queue to empty
        await self.signal_queue.join()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Trading Orchestrator shutdown complete")


# Global orchestrator instance
trading_orchestrator = TradingOrchestrator()
