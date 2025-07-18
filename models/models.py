from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class Signal:
    pair: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    timestamp: datetime
    indicators: Dict[str, any]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None


@dataclass
class TradeSetup:
    pair: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward: float
    confidence: float


@dataclass
class PerformanceStats:
    date: str
    total_trades: int
    win_rate: float
    pnl: float
    pnl_percent: float
    best_trade: Dict
    worst_trade: Dict
    start_balance: float
    current_balance: float
    total_return: float


@dataclass
class UserSettings:
    """User settings model"""
    user_id: int
    strategy: str = "ENHANCED_RSI_EMA"
    exchange: str = "MEXC"
    risk_management: dict = None
    notifications: dict = None
    emergency: dict = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.risk_management is None:
            self.risk_management = {
                "trading_enabled": True,
                "max_risk_per_trade": 0.02,
                "max_open_positions": 3,
                "daily_loss_limit": 0.05
            }
        
        if self.notifications is None:
            self.notifications = {
                "signal_alerts": True,
                "trade_confirmations": True,
                "daily_summary": True,
                "emergency_alerts": True
            }
        
        if self.emergency is None:
            self.emergency = {
                "stop_all_trading": False,
                "close_all_positions": False,
                "notify_admin": False
            }
