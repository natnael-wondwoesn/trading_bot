import pandas as pd
from datetime import datetime
from typing import Dict
from strategy.strategies.strategy import Strategy
from models.models import Signal
from indicators import calculate_rsi, calculate_ema, calculate_atr


class RSIEMAStrategy(Strategy):
    """Strategy combining RSI and EMA crossover signals"""

    def __init__(self, rsi_period: int = 14, ema_fast: int = 9, ema_slow: int = 21):
        super().__init__("RSI + EMA Strategy")
        self.rsi_period = rsi_period
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow
        # Use config values for thresholds
        from config.config import Config

        self.rsi_oversold = Config.RSI_OVERSOLD
        self.rsi_overbought = Config.RSI_OVERBOUGHT

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate RSI and EMA indicators"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        close_prices = data["close"]

        # Calculate indicators
        rsi = calculate_rsi(close_prices, self.rsi_period)
        ema_fast = calculate_ema(close_prices, self.ema_fast_period)
        ema_slow = calculate_ema(close_prices, self.ema_slow_period)

        # Volume analysis
        volume_sma = data["volume"].rolling(window=20).mean()

        self.indicators = {
            "rsi": rsi.iloc[-1],
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "current_price": close_prices.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "avg_volume": volume_sma.iloc[-1],
        }

        return self.indicators

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on RSI and EMA"""
        indicators = self.calculate_indicators(data)

        # RSI signals
        rsi_buy_signal = indicators["rsi"] < self.rsi_oversold
        rsi_sell_signal = indicators["rsi"] > self.rsi_overbought
        rsi_strength = abs(50 - indicators["rsi"]) / 50

        # EMA signals
        ema_bullish = indicators["ema_fast"] > indicators["ema_slow"]
        ema_bearish = indicators["ema_fast"] < indicators["ema_slow"]
        price_above_ema = indicators["current_price"] > indicators["ema_fast"]
        price_below_ema = indicators["current_price"] < indicators["ema_fast"]

        ema_buy_signal = ema_bullish and price_above_ema
        ema_sell_signal = ema_bearish and price_below_ema

        # Volume confirmation
        volume_ok = indicators["volume"] > indicators["avg_volume"] * 0.8

        # Combined signals with confidence
        buy_confidence = 0
        sell_confidence = 0

        if rsi_buy_signal and ema_buy_signal and volume_ok:
            buy_confidence = (rsi_strength + 0.8) / 2

        if rsi_sell_signal and ema_sell_signal and volume_ok:
            sell_confidence = (rsi_strength + 0.8) / 2

        # Determine action
        if buy_confidence > 0.7:
            action = "BUY"
            confidence = buy_confidence
        elif sell_confidence > 0.7:
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
                "rsi": indicators["rsi"],
                "ema_trend": "bullish" if ema_bullish else "bearish",
                "volume_confirmation": volume_ok,
                "volatility": "normal" if atr < data["close"].std() else "high",
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )
