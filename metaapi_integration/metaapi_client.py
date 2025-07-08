import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from metaapi_cloud_sdk import MetaApi
from metaapi_cloud_sdk.clients.error_handler import InternalException

logger = logging.getLogger(__name__)


class MetaAPIClient:
    """MetaAPI client for MT4/MT5 integration"""

    def __init__(self, token: str, account_id: str, region: str = "new-york"):
        self.token = token
        self.account_id = account_id
        self.region = region
        self.api = None
        self.account = None
        self.connection = None
        self.streaming_connection = None
        self.terminal_state = None
        self.history_storage = None

    async def connect(self):
        """Connect to MetaAPI and MetaTrader account"""
        try:
            self.api = MetaApi(self.token, {"region": self.region})
            self.account = await self.api.metatrader_account_api.get_account(
                self.account_id
            )

            # Deploy and wait for account
            if self.account.state != "DEPLOYED":
                logger.info("Deploying MetaTrader account...")
                await self.account.deploy()

            logger.info("Waiting for API server...")
            await self.account.wait_connected()

            # Create connections
            self.connection = self.account.get_rpc_connection()
            self.streaming_connection = self.account.get_streaming_connection()

            # Connect and wait for synchronization
            await self.connection.connect()
            await self.connection.wait_synchronized()

            # Get terminal state and history
            self.terminal_state = self.connection.terminal_state
            self.history_storage = self.connection.history_storage

            logger.info(f"Connected to {self.account.name} ({self.account.server})")

            # Log account info
            account_info = await self.connection.get_account_information()
            logger.info(
                f'Account balance: {account_info["balance"]} {account_info["currency"]}'
            )
            logger.info(f'Account equity: {account_info["equity"]}')

        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            raise

    async def disconnect(self):
        """Disconnect from MetaAPI"""
        if self.connection:
            await self.connection.close()
        if self.api:
            await self.api.close()

    # Market Data Methods
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol specifications"""
        try:
            return await self.connection.get_symbol_specification(symbol)
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {str(e)}")
            return {}

    async def get_price(self, symbol: str) -> Dict:
        """Get current price for symbol"""
        try:
            return await self.connection.get_symbol_price(symbol)
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {str(e)}")
            return {}

    async def get_candles(
        self, symbol: str, timeframe: str = "1H", count: int = 100
    ) -> List[Dict]:
        """Get historical candles"""
        try:
            # Calculate start time
            now = datetime.now()
            if timeframe == "1M":
                start_time = now - timedelta(minutes=count)
            elif timeframe == "5M":
                start_time = now - timedelta(minutes=count * 5)
            elif timeframe == "15M":
                start_time = now - timedelta(minutes=count * 15)
            elif timeframe == "30M":
                start_time = now - timedelta(minutes=count * 30)
            elif timeframe == "1H":
                start_time = now - timedelta(hours=count)
            elif timeframe == "4H":
                start_time = now - timedelta(hours=count * 4)
            elif timeframe == "1D":
                start_time = now - timedelta(days=count)
            else:
                start_time = now - timedelta(hours=count)

            candles = await self.connection.get_candles(
                symbol,
                timeframe,
                start_time.replace(microsecond=0).isoformat() + "Z",
                count,
            )

            return candles

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {str(e)}")
            return []

    # Account Methods
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            return await self.connection.get_account_information()
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            return {}

    async def get_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            return await self.connection.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []

    async def get_orders(self) -> List[Dict]:
        """Get pending orders"""
        try:
            return await self.connection.get_orders()
        except Exception as e:
            logger.error(f"Failed to get orders: {str(e)}")
            return []

    async def get_history_orders(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict]:
        """Get historical orders"""
        try:
            return await self.connection.get_history_orders_by_time_range(
                start_time.isoformat() + "Z", end_time.isoformat() + "Z"
            )
        except Exception as e:
            logger.error(f"Failed to get history: {str(e)}")
            return []

    # Trading Methods
    async def market_order(
        self,
        symbol: str,
        volume: float,
        side: str,
        stop_loss: float = None,
        take_profit: float = None,
        comment: str = None,
    ) -> Dict:
        """Place market order"""
        try:
            order_type = (
                "ORDER_TYPE_BUY" if side.upper() == "BUY" else "ORDER_TYPE_SELL"
            )

            result = await self.connection.create_market_order(
                symbol=symbol,
                volume=volume,
                order_type=order_type,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment,
                options={"clientId": f"bot_{datetime.now().timestamp()}"},
            )

            logger.info(f"Market order placed: {result}")
            return result

        except InternalException as e:
            logger.error(f"Trade error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to place market order: {str(e)}")
            raise

    async def limit_order(
        self,
        symbol: str,
        volume: float,
        side: str,
        price: float,
        stop_loss: float = None,
        take_profit: float = None,
        comment: str = None,
    ) -> Dict:
        """Place limit order"""
        try:
            order_type = (
                "ORDER_TYPE_BUY_LIMIT"
                if side.upper() == "BUY"
                else "ORDER_TYPE_SELL_LIMIT"
            )

            result = await self.connection.create_limit_order(
                symbol=symbol,
                volume=volume,
                order_type=order_type,
                open_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment,
            )

            logger.info(f"Limit order placed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to place limit order: {str(e)}")
            raise

    async def stop_order(
        self,
        symbol: str,
        volume: float,
        side: str,
        stop_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        comment: str = None,
    ) -> Dict:
        """Place stop order"""
        try:
            order_type = (
                "ORDER_TYPE_BUY_STOP"
                if side.upper() == "BUY"
                else "ORDER_TYPE_SELL_STOP"
            )

            result = await self.connection.create_stop_order(
                symbol=symbol,
                volume=volume,
                order_type=order_type,
                open_price=stop_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment,
            )

            logger.info(f"Stop order placed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to place stop order: {str(e)}")
            raise

    async def modify_position(
        self, position_id: str, stop_loss: float = None, take_profit: float = None
    ) -> Dict:
        """Modify existing position"""
        try:
            result = await self.connection.modify_position(
                position_id=position_id, stop_loss=stop_loss, take_profit=take_profit
            )

            logger.info(f"Position modified: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to modify position: {str(e)}")
            raise

    async def close_position(self, position_id: str) -> Dict:
        """Close position"""
        try:
            result = await self.connection.close_position(position_id)
            logger.info(f"Position closed: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to close position: {str(e)}")
            raise

    async def close_all_positions(self) -> List[Dict]:
        """Close all open positions"""
        results = []
        positions = await self.get_positions()

        for position in positions:
            try:
                result = await self.close_position(position["id"])
                results.append(result)
            except Exception as e:
                logger.error(f'Failed to close position {position["id"]}: {str(e)}')

        return results

    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel pending order"""
        try:
            result = await self.connection.cancel_order(order_id)
            logger.info(f"Order cancelled: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to cancel order: {str(e)}")
            raise
