import pandas as pd
from datetime import datetime
from typing import Dict
from strategy.strategies.strategy import Strategy
from models.models import Signal
from indicators import calculate_rsi, calculate_ema, calculate_atr


class EnhancedRSIEMAStrategy(Strategy):
    """Enhanced RSI + EMA Strategy with more practical signal generation"""

    def __init__(self, rsi_period: int = 14, ema_fast: int = 9, ema_slow: int = 21):
        super().__init__("Enhanced RSI + EMA Strategy")
        self.rsi_period = rsi_period
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow

        # More practical thresholds
        self.rsi_oversold = 40  # Less extreme than 35
        self.rsi_overbought = 60  # Less extreme than 65
        self.rsi_strong_oversold = 30  # For high confidence signals
        self.rsi_strong_overbought = 70  # For high confidence signals

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

        # Previous values for trend analysis
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else rsi.iloc[-1]
        prev_ema_fast = ema_fast.iloc[-2] if len(ema_fast) > 1 else ema_fast.iloc[-1]
        prev_ema_slow = ema_slow.iloc[-2] if len(ema_slow) > 1 else ema_slow.iloc[-1]

        self.indicators = {
            "rsi": rsi.iloc[-1],
            "prev_rsi": prev_rsi,
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "prev_ema_fast": prev_ema_fast,
            "prev_ema_slow": prev_ema_slow,
            "current_price": close_prices.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "avg_volume": volume_sma.iloc[-1],
        }

        return self.indicators

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal with multiple signal types"""
        indicators = self.calculate_indicators(data)

        # Signal strength levels
        signal_strength = {"buy": 0, "sell": 0, "reasons": []}

        # 1. RSI Signals (Multiple levels)
        if indicators["rsi"] <= self.rsi_strong_oversold:
            signal_strength["buy"] += 0.4
            signal_strength["reasons"].append("RSI strongly oversold")
        elif indicators["rsi"] <= self.rsi_oversold:
            signal_strength["buy"] += 0.2
            signal_strength["reasons"].append("RSI oversold")

        if indicators["rsi"] >= self.rsi_strong_overbought:
            signal_strength["sell"] += 0.4
            signal_strength["reasons"].append("RSI strongly overbought")
        elif indicators["rsi"] >= self.rsi_overbought:
            signal_strength["sell"] += 0.2
            signal_strength["reasons"].append("RSI overbought")

        # 2. EMA Trend Signals
        ema_bullish = indicators["ema_fast"] > indicators["ema_slow"]
        ema_bearish = indicators["ema_fast"] < indicators["ema_slow"]

        # EMA crossover detection
        prev_ema_bullish = indicators["prev_ema_fast"] > indicators["prev_ema_slow"]
        ema_bullish_crossover = ema_bullish and not prev_ema_bullish
        ema_bearish_crossover = ema_bearish and not prev_ema_bullish

        if ema_bullish_crossover:
            signal_strength["buy"] += 0.3
            signal_strength["reasons"].append("EMA bullish crossover")
        elif ema_bullish:
            signal_strength["buy"] += 0.15
            signal_strength["reasons"].append("EMA bullish trend")

        if ema_bearish_crossover:
            signal_strength["sell"] += 0.3
            signal_strength["reasons"].append("EMA bearish crossover")
        elif ema_bearish:
            signal_strength["sell"] += 0.15
            signal_strength["reasons"].append("EMA bearish trend")

        # 3. Price position relative to EMAs
        price_above_fast_ema = indicators["current_price"] > indicators["ema_fast"]
        price_below_fast_ema = indicators["current_price"] < indicators["ema_fast"]

        if price_above_fast_ema and ema_bullish:
            signal_strength["buy"] += 0.1
            signal_strength["reasons"].append("Price above bullish EMA")

        if price_below_fast_ema and ema_bearish:
            signal_strength["sell"] += 0.1
            signal_strength["reasons"].append("Price below bearish EMA")

        # 4. RSI momentum (improving conditions)
        rsi_improving_for_buy = (
            indicators["rsi"] > indicators["prev_rsi"] and indicators["rsi"] < 50
        )
        rsi_improving_for_sell = (
            indicators["rsi"] < indicators["prev_rsi"] and indicators["rsi"] > 50
        )

        if rsi_improving_for_buy:
            signal_strength["buy"] += 0.1
            signal_strength["reasons"].append("RSI recovering from oversold")

        if rsi_improving_for_sell:
            signal_strength["sell"] += 0.1
            signal_strength["reasons"].append("RSI declining from overbought")

        # 5. Volume confirmation (less strict)
        volume_ok = (
            indicators["volume"] > indicators["avg_volume"] * 0.6
        )  # Reduced from 0.8

        if volume_ok:
            signal_strength["buy"] += 0.05
            signal_strength["sell"] += 0.05
            signal_strength["reasons"].append("Volume confirmation")

        # Determine final action and confidence
        buy_confidence = min(signal_strength["buy"], 1.0)
        sell_confidence = min(signal_strength["sell"], 1.0)

        # Lower confidence threshold for more signals
        min_confidence = 0.4  # Reduced from 0.7

        if buy_confidence >= min_confidence and buy_confidence > sell_confidence:
            action = "BUY"
            confidence = buy_confidence
        elif sell_confidence >= min_confidence and sell_confidence > buy_confidence:
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
            stop_loss = indicators["current_price"] - (1.5 * atr)  # Tighter stop
            take_profit = indicators["current_price"] + (2.5 * atr)  # Better R:R
            risk_reward = 1.67
        elif action == "SELL":
            stop_loss = indicators["current_price"] + (1.5 * atr)
            take_profit = indicators["current_price"] - (2.5 * atr)
            risk_reward = 1.67

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
                "signal_reasons": signal_strength["reasons"],
                "buy_strength": buy_confidence,
                "sell_strength": sell_confidence,
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )
