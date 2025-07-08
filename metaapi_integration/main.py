import asyncio
import logging
from datetime import datetime
from typing import Dict

from config import MetaAPIConfig
from metaapi_client import MetaAPIClient
from mt_data_feed import MTDataFeed
from mt_trade_executor import MTTradeExecutor
from mt_account_manager import MTAccountManager
from strategies.forex_strategy import ForexStrategy

# Import from parent directory
import sys

sys.path.append("..")
from bot import TradingBot
from models import Signal, TradeSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaTraderSystem:
    """Main MetaTrader trading system"""

    def __init__(self, telegram_token: str, telegram_chat_id: str):
        self.config = MetaAPIConfig()
        self.client = None
        self.data_feed = None
        self.trade_executor = None
        self.account_manager = None
        self.strategy = ForexStrategy()
        self.bot = TradingBot(telegram_token, telegram_chat_id)
        self.running = False
        self.active_positions = {}

    async def initialize(self):
        """Initialize MetaTrader connection"""
        try:
            # Connect to MetaAPI
            self.client = MetaAPIClient(
                self.config.META_API_TOKEN,
                self.config.META_API_ACCOUNT_ID,
                self.config.REGION,
            )
            await self.client.connect()

            # Initialize components
            self.data_feed = MTDataFeed(self.client)
            self.trade_executor = MTTradeExecutor(self.client, self.config)
            self.account_manager = MTAccountManager(self.client, self.config)

            # Get account info
            account_info = await self.client.get_account_info()

            # Send startup message
            await self.bot.send_alert(
                "MetaTrader Connected",
                f"Connected to {account_info.get('name', 'Unknown')} "
                f"(Server: {account_info.get('server', 'Unknown')})\n"
                f"Balance: {account_info.get('balance', 0)} {account_info.get('currency', 'USD')}\n"
                f"Monitoring: {', '.join(self.config.FOREX_PAIRS[:3])}...",
                "success",
            )

            # Start data feed
            all_symbols = self.config.FOREX_PAIRS + self.config.CFD_SYMBOLS
            await self.data_feed.start(all_symbols, "1H")

            # Subscribe to candle updates
            for symbol in all_symbols:
                self.data_feed.subscribe(symbol, "candle_closed", self.on_candle_closed)

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    async def on_candle_closed(self, symbol: str, candle_data: Dict):
        """Handle closed candle for signal generation"""
        try:
            # Check risk limits before generating signals
            risk_checks = await self.account_manager.check_risk_limits()
            if not all(risk_checks.values()):
                logger.warning(f"Risk limits exceeded: {risk_checks}")
                return

            # Get candle data
            candles = self.data_feed.get_candles(symbol)
            if len(candles) < 50:
                return

            # Generate signal
            signal = self.strategy.generate_signal(candles)

            if signal.action != "HOLD":
                await self.process_signal(signal)

        except Exception as e:
            logger.error(f"Error processing candle for {symbol}: {str(e)}")

    async def process_signal(self, signal: Signal):
        """Process trading signal"""
        try:
            # Check if we already have a position
            positions = await self.client.get_positions()
            for position in positions:
                if position["symbol"] == signal.pair:
                    logger.info(f"Already have position in {signal.pair}")
                    return

            # Check correlation exposure
            correlation_count = await self.account_manager.get_correlation_exposure(
                signal.pair
            )
            if correlation_count >= self.config.MAX_CORRELATION_POSITIONS:
                logger.info(f"Too many correlated positions for {signal.pair}")
                return

            # Get account info for lot size calculation
            account_info = await self.client.get_account_info()
            risk_amount = account_info["balance"] * self.config.MAX_RISK_PER_TRADE

            # Calculate lot size
            stop_loss_pips = signal.indicators.get("stop_loss_pips", 50)
            lot_size = await self.trade_executor.calculate_lot_size(
                signal.pair, risk_amount, stop_loss_pips
            )

            # Send signal notification
            await self.bot.send_signal_notification(signal, lot_size * 100000)

            # Store for approval
            self.bot.pending_trades[f"{signal.pair}_{signal.timestamp.timestamp()}"] = {
                "signal": signal,
                "lot_size": lot_size,
            }

        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")

    async def execute_trade(self, signal: Signal, lot_size: float):
        """Execute approved trade"""
        try:
            # Execute trade
            result = await self.trade_executor.execute_trade(
                symbol=signal.pair,
                side=signal.action,
                lot_size=lot_size,
                stop_loss_pips=signal.indicators.get("stop_loss_pips"),
                take_profit_pips=signal.indicators.get("take_profit_pips"),
                comment=f"Signal: {signal.confidence:.0%} confidence",
            )

            if result:
                # Create trade setup for notification
                trade = TradeSetup(
                    pair=signal.pair,
                    entry_price=result.get("price", signal.current_price),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=lot_size,
                    risk_reward=signal.risk_reward,
                    confidence=signal.confidence,
                )

                # Store active position
                self.active_positions[result["positionId"]] = {
                    "signal": signal,
                    "trade": trade,
                    "position_id": result["positionId"],
                }

                # Send confirmation
                await self.bot.send_alert(
                    "Trade Executed",
                    self.bot.format_trade_execution_message(trade),
                    "success",
                )

                # Start monitoring
                asyncio.create_task(self.monitor_position(result["positionId"]))

        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            await self.bot.send_alert(
                "Trade Failed",
                f"Failed to execute {signal.action} on {signal.pair}: {str(e)}",
                "error",
            )

    async def monitor_position(self, position_id: str):
        """Monitor position for trailing stop and breakeven"""
        try:
            while position_id in self.active_positions:
                # Check if position still exists
                positions = await self.client.get_positions()
                position = next((p for p in positions if p["id"] == position_id), None)

                if not position:
                    # Position closed
                    del self.active_positions[position_id]
                    break

                # Check for breakeven move
                if self.config.BREAK_EVEN_TRIGGER > 0:
                    await self.trade_executor.move_to_breakeven(
                        position_id, self.config.BREAK_EVEN_TRIGGER
                    )

                await asyncio.sleep(10)  # Check every 10 seconds

        except Exception as e:
            logger.error(f"Error monitoring position: {str(e)}")

    async def send_performance_report(self):
        """Send daily performance report"""
        try:
            # Get performance metrics
            metrics = await self.account_manager.calculate_performance_metrics(30)
            account_info = await self.client.get_account_info()

            message = f"""📊 **MetaTrader Performance Report**

📅 Period: Last 30 days

📈 Trading Statistics:
• Total Trades: {metrics.total_trades}
• Win Rate: {metrics.win_rate:.1%}
• Profit Factor: {metrics.profit_factor:.2f}
• Average Win: ${metrics.average_win:.2f}
• Average Loss: ${metrics.average_loss:.2f}

💰 Account Status:
• Balance: {account_info['balance']:.2f} {account_info['currency']}
• Equity: {account_info['equity']:.2f}
• Margin Level: {account_info.get('marginLevel', 0):.1f}%
• Total P&L: ${metrics.total_pnl:.2f}

📊 Risk Metrics:
• Open Positions: {len(await self.client.get_positions())}
• Daily Loss: ${(await self.account_manager.get_risk_metrics()).daily_loss:.2f}"""

            await self.bot.send_alert("Performance Report", message, "money")

        except Exception as e:
            logger.error(f"Failed to send performance report: {str(e)}")

    async def start(self):
        """Start the trading system"""
        self.running = True

        # Initialize
        await self.initialize()

        # Set up callbacks
        self.bot.trade_callback = self.execute_trade

        # Start performance reporting
        asyncio.create_task(self._performance_loop())

        # Keep running
        while self.running:
            await asyncio.sleep(1)

    async def _performance_loop(self):
        """Send periodic performance reports"""
        while self.running:
            await asyncio.sleep(3600 * 24)  # Daily
            await self.send_performance_report()

    async def stop(self):
        """Stop the trading system"""
        self.running = False

        # Close all positions if configured
        if hasattr(self, "close_all_on_stop") and self.close_all_on_stop:
            await self.client.close_all_positions()

        # Disconnect
        if self.data_feed:
            await self.data_feed.stop()

        if self.client:
            await self.client.disconnect()

        await self.bot.send_alert(
            "System Stopped", "MetaTrader connection closed", "warning"
        )


async def main():
    """Main entry point"""
    # Validate config
    MetaAPIConfig.validate()

    # Get Telegram credentials from parent config
    import os

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    # Create system
    system = MetaTraderSystem(telegram_token, telegram_chat_id)

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await system.stop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        await system.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
