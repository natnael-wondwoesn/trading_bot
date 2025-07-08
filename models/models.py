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
