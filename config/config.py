import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the trading bot"""

    # MEXC API Settings
    MEXC_API_KEY = os.getenv("MEXC_API_KEY")
    MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")

    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Trading Settings
    DEFAULT_POSITION_SIZE = 25
    MAX_RISK_PER_TRADE = 0.02  # 2%
    DEFAULT_TIMEFRAME = "1h"

    # Trading Pairs
    TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

    # Strategy Settings
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    EMA_FAST = 9
    EMA_SLOW = 21

    # Risk Management
    DEFAULT_STOP_LOSS_ATR = 2
    DEFAULT_TAKE_PROFIT_ATR = 3
    MAX_OPEN_POSITIONS = 5
    MIN_VOLUME_FILTER = 100000  # Minimum 24h volume in USDT

    # Data Feed Settings
    WEBSOCKET_RECONNECT_DELAY = 5
    KLINE_INTERVAL = "1m"  # For real-time updates
    SIGNAL_CHECK_INTERVAL = 60  # Check for signals every 60 seconds

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "trading_bot.log"

    # Performance Tracking
    PERFORMANCE_REPORT_TIME = "18:00"  # Daily report time

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.MEXC_API_KEY:
            raise ValueError("MEXC_API_KEY not set in environment")
        if not cls.MEXC_API_SECRET:
            raise ValueError("MEXC_API_SECRET not set in environment")
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        if not cls.TELEGRAM_CHAT_ID:
            raise ValueError("TELEGRAM_CHAT_ID not set in environment")
