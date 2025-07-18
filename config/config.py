import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the trading bot"""

    # MEXC API Settings
    MEXC_API_KEY = os.getenv("MEXC_API_KEY")
    MEXC_API_SECRET = os.getenv("MEXC_API_SECRET")

    # Bybit API Settings
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Trading Settings
    DEFAULT_POSITION_SIZE = 25
    MAX_RISK_PER_TRADE = 0.02  # 2%
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_EXCHANGE = (
        "BYBIT"  # Options: "MEXC", "BYBIT" - Changed to Bybit for better pair support
    )

    # Trading Pairs - Expanded with high-potential RSI strategy pairs
    TRADING_PAIRS = [
        "BTCUSDT",  # Bitcoin - highest volume, excellent for RSI
        "ETHUSDT",  # Ethereum - strong volatility, good RSI signals
        "SOLUSDT",  # Solana - high volatility, trending behavior
        "DOGEUSDT",  # Dogecoin - extreme volatility, excellent RSI swings
        "XRPUSDT",  # Ripple - high volume, strong price movements
        "AVAXUSDT",  # Avalanche - L1 blockchain, high volatility
        "LINKUSDT",  # Chainlink - DeFi leader, consistent volatility
        "ADAUSDT",  # Cardano - maintained from original list
    ]

    # Strategy Settings
    ACTIVE_STRATEGY = "ENHANCED_RSI_EMA"  # Default to enhanced strategy

    # Supported Strategies
    SUPPORTED_STRATEGIES = [
        "RSI_EMA",
        "ENHANCED_RSI_EMA",
        "VISHVA_ML",
        "MACD",
        "BOLLINGER",
        "FOREX",
    ]

    RSI_PERIOD = 14
    RSI_OVERSOLD = 35  # Enhanced strategy uses 40, but keep config for flexibility
    RSI_OVERBOUGHT = 65  # Enhanced strategy uses 60, but keep config for flexibility
    EMA_FAST = 9
    EMA_SLOW = 21

    # Risk Management
    DEFAULT_STOP_LOSS_ATR = 2
    DEFAULT_TAKE_PROFIT_ATR = 3
    MAX_OPEN_POSITIONS = 5
    MIN_VOLUME_FILTER = 50000  # Reduced from 100000 - less strict volume filter

    # Data Feed Settings
    WEBSOCKET_RECONNECT_DELAY = 5
    KLINE_INTERVAL = "1m"  # For real-time updates
    SIGNAL_CHECK_INTERVAL = 900  # Check for signals every 15 minutes (900 seconds)

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "trading_bot.log"

    # Performance Tracking
    PERFORMANCE_REPORT_TIME = "18:00"  # Daily report time

    # ML Strategy Configuration
    ML_MODEL_PATH = "models/vishva_ml"
    ML_RETRAIN_INTERVAL_DAYS = 7
    ML_MIN_CONFIDENCE = 0.6
    ML_FEATURE_LOOKBACK = 200

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Check if at least one exchange is configured
        mexc_configured = cls.MEXC_API_KEY and cls.MEXC_API_SECRET
        bybit_configured = cls.BYBIT_API_KEY and cls.BYBIT_API_SECRET

        if not mexc_configured and not bybit_configured:
            raise ValueError("At least one exchange (MEXC or Bybit) must be configured")

        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        if not cls.TELEGRAM_CHAT_ID:
            raise ValueError("TELEGRAM_CHAT_ID not set in environment")
