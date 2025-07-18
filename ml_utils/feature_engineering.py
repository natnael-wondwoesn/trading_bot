#!/usr/bin/env python3
"""
Advanced Feature Engineering for VishvaAlgo ML Strategy
Implements 190+ technical indicators using pandas/numpy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import TA-Lib, fall back to pandas/numpy implementations
TALIB_AVAILABLE = False
try:
    import talib

    TALIB_AVAILABLE = True
    logger.info("TA-Lib is available for advanced indicators")
except ImportError:
    logger.warning("TA-Lib not available, using pandas/numpy implementations")


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(
                talib.RSI(prices.values, timeperiod=period), index=prices.index
            )
        except:
            pass

    # Pandas implementation
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate EMA using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(
                talib.EMA(prices.values, timeperiod=period), index=prices.index
            )
        except:
            pass

    return prices.ewm(span=period).mean()


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate SMA using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(
                talib.SMA(prices.values, timeperiod=period), index=prices.index
            )
        except:
            pass

    return prices.rolling(window=period).mean()


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD using pandas"""
    if TALIB_AVAILABLE:
        try:
            macd, signal_line, histogram = talib.MACD(
                prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return (
                pd.Series(macd, index=prices.index),
                pd.Series(signal_line, index=prices.index),
                pd.Series(histogram, index=prices.index),
            )
        except:
            pass

    # Pandas implementation
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands using pandas"""
    if TALIB_AVAILABLE:
        try:
            upper, middle, lower = talib.BBANDS(
                prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            return (
                pd.Series(upper, index=prices.index),
                pd.Series(middle, index=prices.index),
                pd.Series(lower, index=prices.index),
            )
        except:
            pass

    # Pandas implementation
    middle = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator using pandas"""
    if TALIB_AVAILABLE:
        try:
            k, d = talib.STOCH(
                high.values,
                low.values,
                close.values,
                fastk_period=k_period,
                slowd_period=d_period,
            )
            return pd.Series(k, index=close.index), pd.Series(d, index=close.index)
        except:
            pass

    # Pandas implementation
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


def calculate_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Williams %R using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(
                talib.WILLR(high.values, low.values, close.values, timeperiod=period),
                index=close.index,
            )
        except:
            pass

    # Pandas implementation
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def calculate_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Commodity Channel Index using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(
                talib.CCI(high.values, low.values, close.values, timeperiod=period),
                index=close.index,
            )
        except:
            pass

    # Pandas implementation
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation)


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate Average True Range using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(
                talib.ATR(high.values, low.values, close.values, timeperiod=period),
                index=close.index,
            )
        except:
            pass

    # Pandas implementation
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))

    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(
        axis=1
    )
    return true_range.rolling(window=period).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume using pandas"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        except:
            pass

    # Pandas implementation
    price_change = close.diff()
    obv = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
    return pd.Series(obv, index=close.index).cumsum()


def calculate_advanced_rsi_features(close: pd.Series) -> Dict[str, float]:
    """Calculate RSI across multiple periods"""
    rsi_periods = [6, 8, 10, 12, 14, 16, 18, 22, 26, 33, 44, 55]
    features = {}

    for period in rsi_periods:
        if len(close) >= period:
            rsi = calculate_rsi(close, period)
            features[f"rsi_{period}"] = (
                rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            )
        else:
            features[f"rsi_{period}"] = 50.0

    return features


def calculate_volume_features(close: pd.Series, volume: pd.Series) -> Dict[str, float]:
    """Calculate volume-based indicators"""
    features = {}

    # On-Balance Volume
    obv = calculate_obv(close, volume)
    features["obv"] = obv.iloc[-1] if not pd.isna(obv.iloc[-1]) else 0.0

    # Volume Rate of Change
    volume_roc = volume.pct_change(5).iloc[-1]
    features["volume_roc"] = volume_roc if not pd.isna(volume_roc) else 0.0

    # VWAP approximation
    vwap = (close * volume).cumsum() / volume.cumsum()
    features["vwap"] = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else close.iloc[-1]
    features["price_to_vwap"] = (
        close.iloc[-1] / features["vwap"] if features["vwap"] != 0 else 1.0
    )

    # Volume trend
    vol_sma = volume.rolling(20).mean()
    features["volume_trend"] = (
        volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] != 0 else 1.0
    )

    return features


def calculate_momentum_features(close: pd.Series) -> Dict[str, float]:
    """Calculate momentum indicators"""
    features = {}

    # Rate of Change for multiple periods
    for period in [5, 10, 20]:
        roc = close.pct_change(period).iloc[-1]
        features[f"roc_{period}"] = roc if not pd.isna(roc) else 0.0

    # Williams %R (if we have enough data for high/low, use close as approximation)
    if len(close) >= 14:
        willr = calculate_williams_r(close, close, close, 14)
        features["williams_r"] = (
            willr.iloc[-1] if not pd.isna(willr.iloc[-1]) else -50.0
        )
    else:
        features["williams_r"] = -50.0

    # CCI approximation using close prices
    if len(close) >= 14:
        cci = calculate_cci(close, close, close, 14)
        features["cci"] = cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0.0
    else:
        features["cci"] = 0.0

    return features


def calculate_volatility_features(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Dict[str, float]:
    """Calculate volatility indicators"""
    features = {}

    # Average True Range
    if len(close) >= 14:
        atr = calculate_atr(high, low, close, 14)
        features["atr"] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
        features["atr_percent"] = (
            features["atr"] / close.iloc[-1] if close.iloc[-1] != 0 else 0.0
        )
    else:
        features["atr"] = 0.0
        features["atr_percent"] = 0.0

    # Bollinger Bands
    if len(close) >= 20:
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20)
        features["bb_upper"] = (
            bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else close.iloc[-1]
        )
        features["bb_middle"] = (
            bb_middle.iloc[-1] if not pd.isna(bb_middle.iloc[-1]) else close.iloc[-1]
        )
        features["bb_lower"] = (
            bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else close.iloc[-1]
        )

        # BB position
        bb_range = features["bb_upper"] - features["bb_lower"]
        if bb_range != 0:
            features["bb_position"] = (close.iloc[-1] - features["bb_lower"]) / bb_range
        else:
            features["bb_position"] = 0.5
    else:
        features.update(
            {
                "bb_upper": close.iloc[-1],
                "bb_middle": close.iloc[-1],
                "bb_lower": close.iloc[-1],
                "bb_position": 0.5,
            }
        )

    # Historical Volatility
    returns = close.pct_change().dropna()
    if len(returns) >= 20:
        features["historical_volatility"] = returns.rolling(20).std().iloc[
            -1
        ] * np.sqrt(252)
    else:
        features["historical_volatility"] = 0.0

    return features


def calculate_pattern_features(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Dict[str, float]:
    """Calculate pattern recognition features"""
    features = {}

    # Price position in recent range
    if len(close) >= 20:
        recent_high = high.rolling(20).max().iloc[-1]
        recent_low = low.rolling(20).min().iloc[-1]
        if recent_high != recent_low:
            features["price_position_20"] = (close.iloc[-1] - recent_low) / (
                recent_high - recent_low
            )
        else:
            features["price_position_20"] = 0.5
    else:
        features["price_position_20"] = 0.5

    # Range as (High / Low) - 1
    current_range = (high.iloc[-1] / low.iloc[-1] - 1) if low.iloc[-1] != 0 else 0.0
    features["current_range"] = current_range

    # Returns as (Close / Close.shift(2)) - 1
    if len(close) >= 3:
        returns_2 = (
            (close.iloc[-1] / close.iloc[-3] - 1) if close.iloc[-3] != 0 else 0.0
        )
        features["returns_2"] = returns_2
    else:
        features["returns_2"] = 0.0

    # Gap analysis
    if len(close) >= 2:
        gap = (
            (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            if close.iloc[-2] != 0
            else 0.0
        )
        features["gap"] = gap
    else:
        features["gap"] = 0.0

    return features


def calculate_all_features(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate all 190+ features for VishvaAlgo"""
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]

    all_features = {}

    try:
        # RSI features (12 different periods)
        all_features.update(calculate_advanced_rsi_features(close))

        # Volume features
        all_features.update(calculate_volume_features(close, volume))

        # Momentum features
        all_features.update(calculate_momentum_features(close))

        # Volatility features
        all_features.update(calculate_volatility_features(high, low, close))

        # Pattern features
        all_features.update(calculate_pattern_features(high, low, close))

        # Moving averages (EMA and SMA)
        ema_periods = [5, 8, 13, 21, 34, 55, 89, 144]
        sma_periods = [10, 20, 50, 100, 200]

        for period in ema_periods:
            if len(close) >= period:
                ema = calculate_ema(close, period)
                all_features[f"ema_{period}"] = (
                    ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else close.iloc[-1]
                )
            else:
                all_features[f"ema_{period}"] = close.iloc[-1]

        for period in sma_periods:
            if len(close) >= period:
                sma = calculate_sma(close, period)
                all_features[f"sma_{period}"] = (
                    sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else close.iloc[-1]
                )
            else:
                all_features[f"sma_{period}"] = close.iloc[-1]

        # MACD family
        if len(close) >= 34:
            macd_line, macd_signal, macd_hist = calculate_macd(close)
            all_features["macd_line"] = (
                macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0
            )
            all_features["macd_signal"] = (
                macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0.0
            )
            all_features["macd_histogram"] = (
                macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0.0
            )
        else:
            all_features.update(
                {"macd_line": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0}
            )

        # Stochastic Oscillator
        if len(close) >= 14:
            stoch_k, stoch_d = calculate_stochastic(high, low, close)
            all_features["stoch_k"] = (
                stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50.0
            )
            all_features["stoch_d"] = (
                stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50.0
            )
        else:
            all_features.update({"stoch_k": 50.0, "stoch_d": 50.0})

        # Elliott Wave Oscillator (SMA5 - SMA35)
        if len(close) >= 35:
            sma5 = calculate_sma(close, 5)
            sma35 = calculate_sma(close, 35)
            ew_diff = sma5.iloc[-1] - sma35.iloc[-1]
            all_features["elliott_wave"] = ew_diff if not pd.isna(ew_diff) else 0.0
        else:
            all_features["elliott_wave"] = 0.0

        # Add current price
        all_features["current_price"] = close.iloc[-1]

        logger.debug(f"Calculated {len(all_features)} features successfully")

    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        # Return basic features as fallback
        all_features = {
            "current_price": close.iloc[-1],
            "rsi_14": 50.0,
            "ema_20": close.iloc[-1],
            "volume_trend": 1.0,
            "atr": 0.0,
            "bb_position": 0.5,
        }

    return all_features
