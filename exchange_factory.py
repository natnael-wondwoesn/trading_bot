#!/usr/bin/env python3
"""
Exchange Factory
Factory pattern implementation for creating exchange clients and data feeds
"""

import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from config.config import Config
from exchange_interface import BaseExchangeClient, BaseDataFeed, BaseTradeExecutor
from mexc.mexc_client import MEXCClient, MEXCTradeExecutor
from mexc.data_feed import MEXCDataFeed
from bybit.bybit_client import BybitClient, BybitTradeExecutor
from bybit.data_feed import BybitDataFeed

logger = logging.getLogger(__name__)


@dataclass
class ExchangeCapabilities:
    """Exchange capabilities and status"""

    name: str
    available: bool
    api_configured: bool
    supports_spot: bool
    supports_futures: bool
    supports_websocket: bool
    error_message: Optional[str] = None


class ExchangeFactory:
    """Factory for creating exchange clients and related services"""

    # Supported exchanges and their implementations
    EXCHANGES = {
        "MEXC": {
            "client_class": MEXCClient,
            "data_feed_class": MEXCDataFeed,
            "executor_class": MEXCTradeExecutor,
            "api_key_config": "MEXC_API_KEY",
            "api_secret_config": "MEXC_API_SECRET",
            "testnet_config": None,
        },
        "BYBIT": {
            "client_class": BybitClient,
            "data_feed_class": BybitDataFeed,
            "executor_class": BybitTradeExecutor,
            "api_key_config": "BYBIT_API_KEY",
            "api_secret_config": "BYBIT_API_SECRET",
            "testnet_config": "BYBIT_TESTNET",
        },
    }

    @classmethod
    def get_available_exchanges(cls) -> Dict[str, ExchangeCapabilities]:
        """Get list of available exchanges and their capabilities"""
        exchanges = {}

        for exchange_name, config in cls.EXCHANGES.items():
            api_key = getattr(Config, config["api_key_config"], None)
            api_secret = getattr(Config, config["api_secret_config"], None)

            api_configured = bool(api_key and api_secret)

            # Determine capabilities based on exchange
            if exchange_name == "MEXC":
                capabilities = ExchangeCapabilities(
                    name="MEXC",
                    available=api_configured,
                    api_configured=api_configured,
                    supports_spot=True,
                    supports_futures=True,
                    supports_websocket=True,
                    error_message=(
                        None if api_configured else "API credentials not configured"
                    ),
                )
            elif exchange_name == "BYBIT":
                capabilities = ExchangeCapabilities(
                    name="Bybit",
                    available=api_configured,
                    api_configured=api_configured,
                    supports_spot=True,
                    supports_futures=True,
                    supports_websocket=True,
                    error_message=(
                        None if api_configured else "API credentials not configured"
                    ),
                )

            exchanges[exchange_name] = capabilities

        return exchanges

    @classmethod
    def create_client(
        cls, exchange_name: str, testnet: bool = False
    ) -> BaseExchangeClient:
        """Create exchange client instance"""
        exchange_name = exchange_name.upper()

        if exchange_name not in cls.EXCHANGES:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        config = cls.EXCHANGES[exchange_name]

        # Get API credentials
        api_key = getattr(Config, config["api_key_config"])
        api_secret = getattr(Config, config["api_secret_config"])

        if not api_key or not api_secret:
            raise ValueError(f"{exchange_name} API credentials not configured")

        # Get testnet setting if supported
        if config["testnet_config"]:
            testnet = getattr(Config, config["testnet_config"], testnet)

        # Create client instance
        client_class = config["client_class"]

        # Handle different client constructors
        if exchange_name == "MEXC":
            client = client_class(api_key, api_secret)
        elif exchange_name == "BYBIT":
            client = client_class(api_key, api_secret, testnet)
        else:
            # Default to 3-parameter constructor
            client = client_class(api_key, api_secret, testnet)

        logger.info(f"Created {exchange_name} client (testnet: {testnet})")
        return client

    @classmethod
    def create_data_feed(
        cls, exchange_name: str, testnet: bool = False
    ) -> BaseDataFeed:
        """Create data feed instance"""
        client = cls.create_client(exchange_name, testnet)

        exchange_name = exchange_name.upper()
        config = cls.EXCHANGES[exchange_name]

        # Create data feed instance
        data_feed_class = config["data_feed_class"]
        data_feed = data_feed_class(client)

        logger.info(f"Created {exchange_name} data feed")
        return data_feed

    @classmethod
    def create_executor(
        cls, exchange_name: str, testnet: bool = False
    ) -> BaseTradeExecutor:
        """Create trade executor instance"""
        client = cls.create_client(exchange_name, testnet)

        exchange_name = exchange_name.upper()
        config = cls.EXCHANGES[exchange_name]

        # Create executor instance
        executor_class = config["executor_class"]
        executor = executor_class(client)

        logger.info(f"Created {exchange_name} trade executor")
        return executor

    @classmethod
    def create_exchange_suite(
        cls, exchange_name: str, testnet: bool = False
    ) -> Tuple[BaseExchangeClient, BaseDataFeed, BaseTradeExecutor]:
        """Create complete exchange suite (client, data feed, executor)"""
        client = cls.create_client(exchange_name, testnet)

        exchange_name = exchange_name.upper()
        config = cls.EXCHANGES[exchange_name]

        # Create data feed and executor with same client
        data_feed_class = config["data_feed_class"]
        executor_class = config["executor_class"]

        data_feed = data_feed_class(client)
        executor = executor_class(client)

        logger.info(f"Created complete {exchange_name} suite")
        return client, data_feed, executor

    @classmethod
    def validate_exchange_config(cls, exchange_name: str) -> Tuple[bool, str]:
        """Validate exchange configuration"""
        exchange_name = exchange_name.upper()

        if exchange_name not in cls.EXCHANGES:
            return False, f"Exchange {exchange_name} is not supported"

        config = cls.EXCHANGES[exchange_name]

        # Check API credentials
        api_key = getattr(Config, config["api_key_config"], None)
        api_secret = getattr(Config, config["api_secret_config"], None)

        if not api_key:
            return False, f"{config['api_key_config']} not configured"

        if not api_secret:
            return False, f"{config['api_secret_config']} not configured"

        return True, f"{exchange_name} configuration is valid"

    @classmethod
    def get_default_exchange(cls) -> str:
        """Get default exchange based on configuration"""
        # Check if default is configured and available
        default = Config.DEFAULT_EXCHANGE.upper()

        available_exchanges = cls.get_available_exchanges()

        if default in available_exchanges and available_exchanges[default].available:
            return default

        # Find first available exchange
        for exchange_name, capabilities in available_exchanges.items():
            if capabilities.available:
                logger.info(f"Using {exchange_name} as default exchange")
                return exchange_name

        raise ValueError("No exchanges are configured and available")

    @classmethod
    async def test_exchange_connection(
        cls, exchange_name: str, testnet: bool = False
    ) -> Tuple[bool, str]:
        """Test connection to exchange"""
        try:
            client = cls.create_client(exchange_name, testnet)

            # Test basic connectivity
            server_time = await client.get_server_time()

            # Test account access (if API supports it)
            try:
                account_info = await client.get_account_info()
                await client.close()
                return True, f"Successfully connected to {exchange_name}"
            except Exception as e:
                await client.close()
                return (
                    True,
                    f"Connected to {exchange_name} but account access failed: {str(e)[:100]}...",
                )

        except Exception as e:
            return False, f"Failed to connect to {exchange_name}: {str(e)}"

    @classmethod
    def get_exchange_display_name(cls, exchange_name: str) -> str:
        """Get user-friendly display name for exchange"""
        display_names = {"MEXC": "MEXC", "BYBIT": "Bybit"}
        return display_names.get(exchange_name.upper(), exchange_name)

    @classmethod
    def get_supported_pairs(cls, exchange_name: str) -> List[str]:
        """Get supported trading pairs for exchange"""
        # This would typically fetch from the exchange API
        # For now, return common pairs
        common_pairs = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "LTCUSDT",
            "DOTUSDT",
            "LINKUSDT",
        ]

        return common_pairs

    @classmethod
    def normalize_symbol_for_exchange(cls, symbol: str, exchange_name: str) -> str:
        """Normalize symbol format for specific exchange"""
        exchange_name = exchange_name.upper()

        # Most exchanges use the same format, but this allows for customization
        if exchange_name == "MEXC":
            return symbol.upper()
        elif exchange_name == "BYBIT":
            return symbol.upper()

        return symbol.upper()


# Convenience functions for common operations
async def get_user_exchange_client(
    user_exchange: str, testnet: bool = False
) -> BaseExchangeClient:
    """Get exchange client for user's selected exchange"""
    return ExchangeFactory.create_client(user_exchange, testnet)


async def get_user_exchange_suite(
    user_exchange: str, testnet: bool = False
) -> Tuple[BaseExchangeClient, BaseDataFeed, BaseTradeExecutor]:
    """Get complete exchange suite for user's selected exchange"""
    return ExchangeFactory.create_exchange_suite(user_exchange, testnet)
