import pandas as pd
from datetime import datetime
from typing import Dict
from strategy.strategies.strategy import Strategy
from models.models import Signal
from indicators import calculate_bollinger_bands, calculate_rsi, calculate_atr


class BollingerStrategy(Strategy):
    """Strategy based on Bollinger Bands squeeze, breakouts and mean reversion"""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14):
        super().__init__("Bollinger Bands Strategy")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

        # Use config values for thresholds
        from config.config import Config

        self.min_volume_factor = 0.8

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """Calculate Bollinger Bands and supporting indicators"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")

        close_prices = data["close"]

        # Calculate Bollinger Bands
        bb_data = calculate_bollinger_bands(close_prices, self.bb_period, self.bb_std)

        # Calculate RSI for confluence
        rsi = calculate_rsi(close_prices, self.rsi_period)

        # Volume analysis
        volume_sma = data["volume"].rolling(window=20).mean()

        # Bollinger Bands analysis
        upper_band = bb_data["upper"]
        middle_band = bb_data["middle"]  # SMA
        lower_band = bb_data["lower"]

        current_price = close_prices.iloc[-1]

        # Band width (volatility measure)
        band_width = (
            (upper_band.iloc[-1] - lower_band.iloc[-1]) / middle_band.iloc[-1]
        ) * 100
        avg_band_width = ((upper_band - lower_band) / middle_band).rolling(
            window=20
        ).mean().iloc[-1] * 100

        # Position relative to bands
        bb_position = (current_price - lower_band.iloc[-1]) / (
            upper_band.iloc[-1] - lower_band.iloc[-1]
        )

        # Band squeeze detection (low volatility)
        squeeze = band_width < avg_band_width * 0.8

        # Band touches and breaks
        touching_upper = current_price >= upper_band.iloc[-1] * 0.99
        touching_lower = current_price <= lower_band.iloc[-1] * 1.01
        breaking_upper = current_price > upper_band.iloc[-1]
        breaking_lower = current_price < lower_band.iloc[-1]

        # Mean reversion signals
        oversold_rsi = rsi.iloc[-1] < 30 and touching_lower
        overbought_rsi = rsi.iloc[-1] > 70 and touching_upper

        # Trend following signals
        uptrend_breakout = breaking_upper and rsi.iloc[-1] > 50
        downtrend_breakout = breaking_lower and rsi.iloc[-1] < 50

        self.indicators = {
            "upper_band": upper_band.iloc[-1],
            "middle_band": middle_band.iloc[-1],
            "lower_band": lower_band.iloc[-1],
            "current_price": current_price,
            "bb_position": bb_position,
            "band_width": band_width,
            "avg_band_width": avg_band_width,
            "squeeze": squeeze,
            "touching_upper": touching_upper,
            "touching_lower": touching_lower,
            "breaking_upper": breaking_upper,
            "breaking_lower": breaking_lower,
            "oversold_rsi": oversold_rsi,
            "overbought_rsi": overbought_rsi,
            "uptrend_breakout": uptrend_breakout,
            "downtrend_breakout": downtrend_breakout,
            "rsi": rsi.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "avg_volume": volume_sma.iloc[-1],
        }

        return self.indicators

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on Bollinger Bands"""
        indicators = self.calculate_indicators(data)

        # Volume confirmation
        volume_ok = (
            indicators["volume"] > indicators["avg_volume"] * self.min_volume_factor
        )

        # Signal generation with confidence scoring
        buy_confidence = 0
        sell_confidence = 0

        # Buy signals (mean reversion and breakouts)
        if indicators["oversold_rsi"] and volume_ok:
            buy_confidence = 0.8  # Strong mean reversion signal
        elif indicators["uptrend_breakout"] and volume_ok:
            buy_confidence = 0.7  # Trend following breakout
        elif indicators["touching_lower"] and indicators["rsi"] < 40 and volume_ok:
            buy_confidence = 0.6  # Moderate oversold
        elif (
            indicators["bb_position"] < 0.2 and not indicators["squeeze"] and volume_ok
        ):
            buy_confidence = 0.5  # Near lower band

        # Sell signals (mean reversion and breakouts)
        if indicators["overbought_rsi"] and volume_ok:
            sell_confidence = 0.8  # Strong mean reversion signal
        elif indicators["downtrend_breakout"] and volume_ok:
            sell_confidence = 0.7  # Trend following breakdown
        elif indicators["touching_upper"] and indicators["rsi"] > 60 and volume_ok:
            sell_confidence = 0.6  # Moderate overbought
        elif (
            indicators["bb_position"] > 0.8 and not indicators["squeeze"] and volume_ok
        ):
            sell_confidence = 0.5  # Near upper band

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

        # Calculate stop loss and take profit
        atr = calculate_atr(data["high"], data["low"], data["close"]).iloc[-1]

        # Use band width for dynamic risk management
        band_range = indicators["upper_band"] - indicators["lower_band"]

        stop_loss = None
        take_profit = None
        risk_reward = None

        if action == "BUY":
            # For mean reversion: target middle band, stop below lower band
            if indicators["oversold_rsi"] or indicators["touching_lower"]:
                stop_loss = indicators["lower_band"] - (atr * 0.5)
                take_profit = indicators["middle_band"]
                risk_reward = abs(take_profit - indicators["current_price"]) / abs(
                    indicators["current_price"] - stop_loss
                )
            # For breakouts: use ATR-based levels
            else:
                stop_loss = indicators["current_price"] - (2 * atr)
                take_profit = indicators["current_price"] + (3 * atr)
                risk_reward = 1.5

        elif action == "SELL":
            # For mean reversion: target middle band, stop above upper band
            if indicators["overbought_rsi"] or indicators["touching_upper"]:
                stop_loss = indicators["upper_band"] + (atr * 0.5)
                take_profit = indicators["middle_band"]
                risk_reward = abs(indicators["current_price"] - take_profit) / abs(
                    stop_loss - indicators["current_price"]
                )
            # For breakouts: use ATR-based levels
            else:
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
                "bb_position": round(indicators["bb_position"], 2),
                "band_width": round(indicators["band_width"], 2),
                "squeeze": indicators["squeeze"],
                "signal_type": (
                    "mean_reversion"
                    if indicators["oversold_rsi"] or indicators["overbought_rsi"]
                    else "breakout"
                ),
                "rsi": round(indicators["rsi"], 1),
                "upper_band": round(indicators["upper_band"], 2),
                "lower_band": round(indicators["lower_band"], 2),
                "volume_confirmation": volume_ok,
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )
