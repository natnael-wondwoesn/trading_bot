from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict
from models.models import Signal


class Strategy(ABC):
    """Base class for all trading strategies"""

    def __init__(self, name: str, timeframe: str = "1h"):
        self.name = name
        self.timeframe = timeframe
        self.indicators = {}

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate all indicators needed for the strategy"""
        pass

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on strategy logic"""
        pass

    def calculate_position_size(
        self, account_balance: float, risk_percent: float = 0.02
    ) -> float:
        """Calculate position size based on risk management"""
        return account_balance * risk_percent

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)

    def get_market_regime(self, data: pd.DataFrame) -> str:
        """Identify current market regime"""
        from indicators import calculate_atr, calculate_sma

        close_prices = data["close"]
        sma_50 = calculate_sma(close_prices, 50)
        sma_200 = calculate_sma(close_prices, 200)
        atr = calculate_atr(data["high"], data["low"], data["close"])

        current_price = close_prices.iloc[-1]
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()

        if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
            regime = "trending_up"
        elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
            regime = "trending_down"
        else:
            regime = "ranging"

        if current_atr > avg_atr * 1.5:
            regime += "_volatile"

        return regime
