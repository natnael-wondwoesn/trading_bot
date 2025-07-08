import os
from dotenv import load_dotenv

load_dotenv()


class MetaAPIConfig:
    """Configuration for MetaAPI integration"""

    # MetaAPI Credentials
    META_API_TOKEN = os.getenv("META_API_TOKEN")
    META_API_ACCOUNT_ID = os.getenv("META_API_ACCOUNT_ID")

    # Trading Settings
    DEFAULT_LOT_SIZE = 0.01
    MAX_RISK_PER_TRADE = 0.02  # 2%
    DEFAULT_SLIPPAGE = 10  # pips

    # Forex Pairs to Trade
    FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]

    # CFD Symbols
    CFD_SYMBOLS = [
        "XAUUSD",  # Gold
        "XAGUSD",  # Silver
        "US30",  # Dow Jones
        "NAS100",  # Nasdaq
        "GER40",  # DAX
        "UK100",  # FTSE
    ]

    # Strategy Settings
    USE_TRAILING_STOP = True
    TRAILING_STOP_DISTANCE = 30  # pips
    BREAK_EVEN_TRIGGER = 20  # pips

    # MetaAPI Settings
    REGION = "new-york"  # or 'london', 'singapore'
    RELIABILITY = "high"  # or 'regular'

    # Risk Management
    MAX_OPEN_POSITIONS = 5
    MAX_DAILY_LOSS = 0.05  # 5%
    MAX_CORRELATION_POSITIONS = 2  # Max correlated pairs

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.META_API_TOKEN:
            raise ValueError("META_API_TOKEN not set")
        if not cls.META_API_ACCOUNT_ID:
            raise ValueError("META_API_ACCOUNT_ID not set")
