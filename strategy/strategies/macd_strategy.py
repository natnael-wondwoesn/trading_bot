import pandas as pd
from datetime import datetime
from typing import Dict
from strategy.strategies.strategy import Strategy
from models.models import Signal
from indicators import calculate_macd, calculate_ema, calculate_atr


class MACDStrategy(Strategy):
    """Strategy based on MACD crossovers and momentum signals"""

    def __init__(self, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        super().__init__("MACD Strategy")
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_trend = 50  # For trend confirmation

        # Use config values for thresholds
        from config.config import Config

        self.min_volume_factor = 0.8

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate MACD and supporting indicators"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        close_prices = data["close"]

        # Calculate MACD
        macd_data = calculate_macd(
            close_prices, self.macd_fast, self.macd_slow, self.macd_signal
        )

        # Calculate trend EMA
        ema_trend = calculate_ema(close_prices, self.ema_trend)

        # Volume analysis
        volume_sma = data["volume"].rolling(window=20).mean()

        # MACD analysis
        macd_line = macd_data["macd"]
        signal_line = macd_data["signal"]
        histogram = macd_data["histogram"]

        # Trend and momentum
        macd_above_signal = macd_line.iloc[-1] > signal_line.iloc[-1]
        macd_above_zero = macd_line.iloc[-1] > 0
        histogram_increasing = histogram.iloc[-1] > histogram.iloc[-2]

        # Previous values for crossover detection
        prev_macd_above_signal = macd_line.iloc[-2] > signal_line.iloc[-2]

        self.indicators = {
            "macd": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": histogram.iloc[-1],
            "macd_above_signal": macd_above_signal,
            "macd_above_zero": macd_above_zero,
            "histogram_increasing": histogram_increasing,
            "bullish_crossover": macd_above_signal and not prev_macd_above_signal,
            "bearish_crossover": not macd_above_signal and prev_macd_above_signal,
            "ema_trend": ema_trend.iloc[-1],
            "current_price": close_prices.iloc[-1],
            "price_above_trend": close_prices.iloc[-1] > ema_trend.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "avg_volume": volume_sma.iloc[-1],
        }

        return self.indicators

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on MACD crossovers"""
        indicators = self.calculate_indicators(data)

        # MACD signal conditions
        bullish_crossover = indicators["bullish_crossover"]
        bearish_crossover = indicators["bearish_crossover"]

        # Trend confirmation
        uptrend = indicators["price_above_trend"]

        # MACD momentum
        strong_bullish = (
            indicators["macd_above_signal"]
            and indicators["macd_above_zero"]
            and indicators["histogram_increasing"]
        )

        strong_bearish = (
            not indicators["macd_above_signal"]
            and indicators["macd"] < 0
            and not indicators["histogram_increasing"]
        )

        # Volume confirmation
        volume_ok = (
            indicators["volume"] > indicators["avg_volume"] * self.min_volume_factor
        )

        # Signal generation with confidence scoring
        buy_confidence = 0
        sell_confidence = 0

        # Buy conditions
        if bullish_crossover and uptrend and volume_ok:
            buy_confidence = 0.8
        elif strong_bullish and uptrend and volume_ok:
            buy_confidence = 0.6
        elif bullish_crossover and volume_ok:
            buy_confidence = 0.5

        # Sell conditions
        if bearish_crossover and not uptrend and volume_ok:
            sell_confidence = 0.8
        elif strong_bearish and not uptrend and volume_ok:
            sell_confidence = 0.6
        elif bearish_crossover and volume_ok:
            sell_confidence = 0.5

        # Determine action
        if buy_confidence >= 0.5:
            action = "BUY"
            confidence = buy_confidence
        elif sell_confidence >= 0.5:
            action = "SELL"
            confidence = sell_confidence
        else:
            action = "HOLD"
            confidence = 0

        # Calculate stop loss and take profit using ATR
        atr = calculate_atr(data["high"], data["low"], data["close"]).iloc[-1]

        stop_loss = None
        take_profit = None
        risk_reward = None

        if action == "BUY":
            stop_loss = indicators["current_price"] - (2 * atr)
            take_profit = indicators["current_price"] + (3 * atr)
            risk_reward = 1.5
        elif action == "SELL":
            stop_loss = indicators["current_price"] + (2 * atr)
            take_profit = indicators["current_price"] - (3 * atr)
            risk_reward = 1.5

        return Signal(
            pair=data.attrs.get("pair", "UNKNOWN"),
            action=action,
            confidence=confidence,
            current_price=indicators["current_price"],
            timestamp=datetime.now(),
            indicators={
                "macd": indicators["macd"],
                "signal": indicators["signal"],
                "histogram": indicators["histogram"],
                "crossover_type": (
                    "bullish"
                    if bullish_crossover
                    else "bearish" if bearish_crossover else "none"
                ),
                "trend": "bullish" if uptrend else "bearish",
                "momentum": (
                    "strong_bullish"
                    if strong_bullish
                    else "strong_bearish" if strong_bearish else "neutral"
                ),
                "volume_confirmation": volume_ok,
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )
