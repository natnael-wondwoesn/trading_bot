import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MTTradeExecutor:
    """Trade execution handler for MetaTrader"""

    def __init__(self, client, config):
        self.client = client
        self.config = config

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        stop_loss_pips: float = None,
        take_profit_pips: float = None,
        comment: str = None,
    ) -> Dict:
        """Execute market order with proper risk management"""
        try:
            # Get current price
            price_data = await self.client.get_price(symbol)
            if not price_data:
                raise Exception(f"No price data for {symbol}")

            current_price = price_data["bid"] if side == "SELL" else price_data["ask"]

            # Get symbol info for pip calculation
            symbol_info = await self.client.get_symbol_info(symbol)
            pip_value = self._calculate_pip_value(symbol, symbol_info)

            # Calculate SL/TP prices
            stop_loss = None
            take_profit = None

            if stop_loss_pips:
                if side == "BUY":
                    stop_loss = current_price - (stop_loss_pips * pip_value)
                else:
                    stop_loss = current_price + (stop_loss_pips * pip_value)

            if take_profit_pips:
                if side == "BUY":
                    take_profit = current_price + (take_profit_pips * pip_value)
                else:
                    take_profit = current_price - (take_profit_pips * pip_value)

            # Execute order
            result = await self.client.market_order(
                symbol=symbol,
                volume=lot_size,
                side=side,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
                or f"Bot trade {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            logger.info(
                f"Trade executed: {symbol} {side} {lot_size} lots @ {current_price}"
            )
            return result

        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            raise

    def _calculate_pip_value(self, symbol: str, symbol_info: Dict) -> float:
        """Calculate pip value for symbol"""
        if "JPY" in symbol:
            return 0.01
        else:
            return 0.0001

    async def calculate_lot_size(
        self, symbol: str, risk_amount: float, stop_loss_pips: float
    ) -> float:
        """Calculate lot size based on risk"""
        try:
            # Get account info
            account_info = await self.client.get_account_info()

            # Get symbol info
            symbol_info = await self.client.get_symbol_info(symbol)

            # Calculate pip value in account currency
            pip_value = self._calculate_pip_value(symbol, symbol_info)
            contract_size = symbol_info.get("contractSize", 100000)

            # Get current price for value calculation
            price_data = await self.client.get_price(symbol)
            current_price = price_data.get("bid", 1)

            # Calculate lot size
            # Risk = Lot Size × Pip Value × Stop Loss in Pips
            pip_value_per_lot = (pip_value * contract_size) / current_price

            if symbol_info.get("profitCurrency") != account_info.get("currency"):
                # Need currency conversion
                pip_value_per_lot = pip_value_per_lot * current_price

            lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

            # Apply symbol constraints
            min_lot = symbol_info.get("minVolume", 0.01)
            max_lot = symbol_info.get("maxVolume", 100)
            lot_step = symbol_info.get("volumeStep", 0.01)

            # Round to lot step
            lot_size = round(lot_size / lot_step) * lot_step
            lot_size = max(min_lot, min(lot_size, max_lot))

            return lot_size

        except Exception as e:
            logger.error(f"Lot size calculation failed: {str(e)}")
            return self.config.DEFAULT_LOT_SIZE

    async def set_trailing_stop(self, position_id: str, trailing_stop_pips: float):
        """Set trailing stop for position"""
        try:
            position = await self._get_position_by_id(position_id)
            if not position:
                raise Exception(f"Position {position_id} not found")

            symbol_info = await self.client.get_symbol_info(position["symbol"])
            pip_value = self._calculate_pip_value(position["symbol"], symbol_info)

            # Calculate trailing stop distance
            distance = trailing_stop_pips * pip_value

            # MetaAPI doesn't directly support trailing stops
            # Need to implement manual trailing logic
            logger.warning("Trailing stops need manual implementation with MetaAPI")

        except Exception as e:
            logger.error(f"Failed to set trailing stop: {str(e)}")

    async def move_to_breakeven(self, position_id: str, trigger_pips: float):
        """Move stop loss to breakeven after trigger"""
        try:
            position = await self._get_position_by_id(position_id)
            if not position:
                raise Exception(f"Position {position_id} not found")

            symbol_info = await self.client.get_symbol_info(position["symbol"])
            pip_value = self._calculate_pip_value(position["symbol"], symbol_info)

            current_price = position["currentPrice"]
            open_price = position["openPrice"]

            # Check if position is in profit by trigger amount
            if position["type"] == "POSITION_TYPE_BUY":
                profit_pips = (current_price - open_price) / pip_value
                if profit_pips >= trigger_pips:
                    # Move stop loss to breakeven + small buffer
                    new_sl = open_price + (2 * pip_value)  # 2 pip buffer
                    await self.client.modify_position(position_id, stop_loss=new_sl)
                    logger.info(f"Moved position {position_id} to breakeven")

            else:  # SELL position
                profit_pips = (open_price - current_price) / pip_value
                if profit_pips >= trigger_pips:
                    new_sl = open_price - (2 * pip_value)
                    await self.client.modify_position(position_id, stop_loss=new_sl)
                    logger.info(f"Moved position {position_id} to breakeven")

        except Exception as e:
            logger.error(f"Failed to move to breakeven: {str(e)}")

    async def _get_position_by_id(self, position_id: str) -> Optional[Dict]:
        """Get position by ID"""
        positions = await self.client.get_positions()
        for position in positions:
            if position["id"] == position_id:
                return position
        return None

    async def close_partial_position(self, position_id: str, close_percent: float):
        """Close partial position"""
        try:
            position = await self._get_position_by_id(position_id)
            if not position:
                raise Exception(f"Position {position_id} not found")

            current_volume = position["volume"]
            close_volume = round(current_volume * close_percent, 2)

            # MetaAPI requires closing full position and reopening
            # This is a limitation - need to close full and reopen partial
            logger.warning("Partial close requires full close and reopen with MetaAPI")

        except Exception as e:
            logger.error(f"Failed to close partial position: {str(e)}")
