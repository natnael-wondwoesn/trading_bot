import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging

from strategy.strategies.strategy import Strategy
from models.models import Signal
from indicators import calculate_rsi, calculate_ema, calculate_atr, calculate_macd

logger = logging.getLogger(__name__)


class ForexStrategy(Strategy):
    """Forex trading strategy for MetaTrader"""

    def __init__(self, timeframe: str = "1H"):
        super().__init__("Forex Momentum Strategy", timeframe)
        self.rsi_period = 14
        self.ema_fast = 20
        self.ema_slow = 50
        self.atr_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate forex-specific indicators"""
        close_prices = data["close"]

        # Calculate indicators
        rsi = calculate_rsi(close_prices, self.rsi_period)
        ema_fast = calculate_ema(close_prices, self.ema_fast)
        ema_slow = calculate_ema(close_prices, self.ema_slow)
        atr = calculate_atr(data["high"], data["low"], data["close"], self.atr_period)
        macd_data = calculate_macd(
            close_prices, self.macd_fast, self.macd_slow, self.macd_signal
        )

        # Volume analysis (tick volume for forex)
        volume_sma = data["volume"].rolling(window=20).mean()

        # Market structure
        higher_highs = data["high"].iloc[-1] > data["high"].iloc[-10:-1].max()
        lower_lows = data["low"].iloc[-1] < data["low"].iloc[-10:-1].min()

        self.indicators = {
            "rsi": rsi.iloc[-1],
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "atr": atr.iloc[-1],
            "macd": macd_data["macd"].iloc[-1],
            "macd_signal": macd_data["signal"].iloc[-1],
            "macd_histogram": macd_data["histogram"].iloc[-1],
            "current_price": close_prices.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "avg_volume": volume_sma.iloc[-1],
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "spread": data.get("spread", pd.Series([0])).iloc[-1],
        }

        return self.indicators

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal for forex"""
        indicators = self.calculate_indicators(data)

        # Trend determination
        trend_up = indicators["ema_fast"] > indicators["ema_slow"]
        trend_down = indicators["ema_fast"] < indicators["ema_slow"]

        # RSI conditions
        rsi_bullish = 40 < indicators["rsi"] < 60  # Forex uses different levels
        rsi_bearish = 40 < indicators["rsi"] < 60

        # MACD conditions
        macd_bullish = (
            indicators["macd"] > indicators["macd_signal"]
            and indicators["macd_histogram"] > 0
        )
        macd_bearish = (
            indicators["macd"] < indicators["macd_signal"]
            and indicators["macd_histogram"] < 0
        )

        # Market structure
        uptrend_structure = indicators["higher_highs"]
        downtrend_structure = indicators["lower_lows"]

        # Volume confirmation
        volume_ok = indicators["volume"] > indicators["avg_volume"] * 0.8

        # Spread filter (important for forex)
        max_spread_pips = 3  # Maximum acceptable spread
        spread_ok = indicators["spread"] <= max_spread_pips

        # Signal generation
        buy_signal = (
            trend_up
            and macd_bullish
            and rsi_bullish
            and uptrend_structure
            and volume_ok
            and spread_ok
        )

        sell_signal = (
            trend_down
            and macd_bearish
            and rsi_bearish
            and downtrend_structure
            and volume_ok
            and spread_ok
        )

        # Determine action and confidence
        if buy_signal:
            action = "BUY"
            confidence = self._calculate_confidence(indicators, "BUY")
        elif sell_signal:
            action = "SELL"
            confidence = self._calculate_confidence(indicators, "SELL")
        else:
            action = "HOLD"
            confidence = 0

        # Calculate stop loss and take profit in pips
        atr_pips = self._price_to_pips(data.attrs.get("pair", ""), indicators["atr"])

        stop_loss_pips = atr_pips * 1.5  # 1.5x ATR for stop loss
        take_profit_pips = atr_pips * 2.5  # 2.5x ATR for take profit
        risk_reward = take_profit_pips / stop_loss_pips

        # Convert to price levels
        pip_value = self._get_pip_value(data.attrs.get("pair", ""))

        if action == "BUY":
            stop_loss = indicators["current_price"] - (stop_loss_pips * pip_value)
            take_profit = indicators["current_price"] + (take_profit_pips * pip_value)
        elif action == "SELL":
            stop_loss = indicators["current_price"] + (stop_loss_pips * pip_value)
            take_profit = indicators["current_price"] - (take_profit_pips * pip_value)
        else:
            stop_loss = None
            take_profit = None

        return Signal(
            pair=data.attrs.get("pair", "UNKNOWN"),
            action=action,
            confidence=confidence,
            current_price=indicators["current_price"],
            timestamp=datetime.now(),
            indicators={
                "trend": "bullish" if trend_up else "bearish",
                "rsi": indicators["rsi"],
                "macd_histogram": indicators["macd_histogram"],
                "spread": indicators["spread"],
                "atr_pips": atr_pips,
                "stop_loss_pips": stop_loss_pips,
                "take_profit_pips": take_profit_pips,
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )

    def _calculate_confidence(self, indicators: Dict, direction: str) -> float:
        """Calculate signal confidence"""
        confidence_score = 0

        if direction == "BUY":
            # Trend alignment
            if indicators["ema_fast"] > indicators["ema_slow"]:
                confidence_score += 0.3

            # MACD strength
            if indicators["macd_histogram"] > 0:
                confidence_score += 0.2

            # RSI not overbought
            if indicators["rsi"] < 70:
                confidence_score += 0.2

            # Market structure
            if indicators["higher_highs"]:
                confidence_score += 0.2

            # Low spread
            if indicators["spread"] < 2:
                confidence_score += 0.1

        else:  # SELL
            if indicators["ema_fast"] < indicators["ema_slow"]:
                confidence_score += 0.3

            if indicators["macd_histogram"] < 0:
                confidence_score += 0.2

            if indicators["rsi"] > 30:
                confidence_score += 0.2

            if indicators["lower_lows"]:
                confidence_score += 0.2

            if indicators["spread"] < 2:
                confidence_score += 0.1

        return min(confidence_score, 1.0)

    def _price_to_pips(self, symbol: str, price_difference: float) -> float:
        """Convert price difference to pips"""
        if "JPY" in symbol:
            return price_difference * 100
        else:
            return price_difference * 10000

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        if "JPY" in symbol:
            return 0.01
        else:
            return 0.0001
