import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from config.config import Config
from bot import TradingBot
from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
from strategy.strategies.macd_strategy import MACDStrategy
from strategy.strategies.bollinger_strategy import BollingerStrategy
from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
from models.models import PerformanceStats, Signal, TradeSetup
from mexc.mexc_client import MEXCClient, MEXCTradeExecutor
from mexc.data_feed import MEXCDataFeed
from user_settings import user_settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading system coordinator with MEXC integration"""

    def __init__(self):
        # Initialize MEXC client
        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
        self.trade_executor = MEXCTradeExecutor(self.mexc_client)
        self.data_feed = MEXCDataFeed(self.mexc_client)

        # Initialize bot and strategy
        self.bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

        # Create Telegram application for the bot
        from telegram.ext import Application

        self.bot.application = (
            Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        )

        # Initialize strategy based on user settings
        self.strategy = self.get_current_strategy()

        self.running = False
        self.active_trades = {}
        self.performance_tracker = PerformanceTracker()

        # Set up trade callback for bot
        self.bot.trade_callback = self.execute_trade
        self.bot.trading_system = self

    def get_current_strategy(self):
        """Get the current strategy based on user settings"""
        strategy_type = user_settings.get_strategy()

        if strategy_type == "MACD":
            return MACDStrategy()
        elif strategy_type == "BOLLINGER":
            return BollingerStrategy()
        elif strategy_type == "ENHANCED_RSI_EMA":
            return EnhancedRSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )
        else:  # Default to ENHANCED_RSI_EMA
            return EnhancedRSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )

    def update_strategy(self):
        """Update strategy if user changed it"""
        new_strategy = self.get_current_strategy()
        if type(new_strategy) != type(self.strategy):
            self.strategy = new_strategy
            logger.info(f"Strategy updated to: {user_settings.get_strategy()}")

    async def initialize(self):
        """Initialize the trading system"""
        logger.info("Initializing trading system...")

        # Check account balance
        balance = await self.mexc_client.get_balance()
        logger.info(f"Account balance: {balance}")

        # Send startup message
        await self.bot.send_alert(
            "System Started",
            f"Trading bot connected to MEXC\nMonitoring pairs: {', '.join(Config.TRADING_PAIRS)}",
            "success",
        )

        # Start data feed
        await self.data_feed.start(Config.TRADING_PAIRS, Config.KLINE_INTERVAL)

        # Subscribe to kline updates for signal generation
        for pair in Config.TRADING_PAIRS:
            self.data_feed.subscribe(pair, "kline_closed", self.on_kline_update)

    async def on_kline_update(self, symbol: str, data: any):
        """Handle kline updates and check for signals"""
        try:
            # Check if trading is enabled and not in emergency mode
            if not user_settings.is_trading_enabled():
                return

            # Update strategy if it was changed
            self.update_strategy()

            # Get latest data
            kline_data = self.data_feed.get_latest_data(symbol)

            if len(kline_data) < 50:  # Need enough data for indicators
                return

            # Generate signal
            signal = self.strategy.generate_signal(kline_data)

            # Process signal if not HOLD
            if signal.action != "HOLD":
                await self.process_signal(signal)

        except Exception as e:
            logger.error(f"Error processing kline update for {symbol}: {str(e)}")

    async def process_signal(self, signal: Signal):
        """Process trading signal"""
        # Double-check trading is enabled
        if not user_settings.is_trading_enabled():
            logger.info("Trading disabled, skipping signal")
            return

        # Check if we already have a position
        if signal.pair in self.active_trades:
            logger.info(f"Already have position in {signal.pair}, skipping signal")
            return

        # Check maximum open positions limit
        risk_settings = user_settings.get_risk_settings()
        if len(self.active_trades) >= risk_settings["max_open_positions"]:
            logger.info(
                f"Max open positions ({risk_settings['max_open_positions']}) reached, skipping signal"
            )
            await self.bot.send_alert(
                "Position Limit Reached",
                f"Cannot open new position for {signal.pair}. Max positions: {risk_settings['max_open_positions']}",
                "warning",
            )
            return

        # Check volume filter
        ticker = await self.mexc_client.get_ticker(signal.pair)
        volume_usdt = float(ticker["quoteVolume"])

        if volume_usdt < Config.MIN_VOLUME_FILTER:
            logger.info(f"Volume too low for {signal.pair}: {volume_usdt}")
            return

        # Get account balance
        balance_data = await self.mexc_client.get_balance("USDT")
        available_balance = balance_data["free"]

        # Use user settings for risk management
        risk_percent = risk_settings["max_risk_per_trade"]

        # Apply custom stop loss/take profit if set
        stop_loss_price = signal.stop_loss
        take_profit_price = signal.take_profit

        if risk_settings["custom_stop_loss"]:
            stop_loss_price = (
                signal.current_price * (1 - risk_settings["custom_stop_loss"] / 100)
                if signal.action == "BUY"
                else signal.current_price
                * (1 + risk_settings["custom_stop_loss"] / 100)
            )

        if risk_settings["custom_take_profit"]:
            take_profit_price = (
                signal.current_price * (1 + risk_settings["custom_take_profit"] / 100)
                if signal.action == "BUY"
                else signal.current_price
                * (1 - risk_settings["custom_take_profit"] / 100)
            )

        # Update signal with custom levels
        signal.stop_loss = stop_loss_price
        signal.take_profit = take_profit_price

        # Calculate position size
        position_size = await self.trade_executor.calculate_position_size(
            symbol=signal.pair,
            account_balance=available_balance,
            risk_percent=risk_percent,
            stop_loss_price=stop_loss_price,
            entry_price=signal.current_price,
        )

        # Send signal notification only if notifications are enabled
        if user_settings.settings["notifications"]["signal_alerts"]:
            await self.bot.send_signal_notification(
                signal, position_size * signal.current_price
            )

        # Store signal for approval
        self.bot.pending_trades[f"{signal.pair}_{signal.timestamp.timestamp()}"] = {
            "signal": signal,
            "position_size": position_size,
        }

    async def execute_trade(self, signal: Signal, position_size: float):
        """Execute trade based on approved signal"""
        try:
            # Execute market order
            order = await self.trade_executor.execute_market_order(
                symbol=signal.pair, side=signal.action, quantity=position_size
            )

            # Create trade setup
            trade = TradeSetup(
                pair=signal.pair,
                entry_price=float(order["fills"][0]["price"]),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size=position_size,
                risk_reward=signal.risk_reward,
                confidence=signal.confidence,
            )

            # Store active trade
            self.active_trades[signal.pair] = {
                "trade": trade,
                "order_id": order["orderId"],
                "entry_time": datetime.now(),
            }

            # Send execution confirmation only if notifications are enabled
            if user_settings.settings["notifications"]["trade_execution"]:
                await self.bot.send_alert(
                    "Trade Executed",
                    self.bot.format_trade_execution_message(trade),
                    "success",
                )

            # Start monitoring for stop loss and take profit
            asyncio.create_task(self.monitor_trade(signal.pair))

        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")
            await self.bot.send_alert(
                "Trade Execution Failed",
                f"Failed to execute {signal.action} order for {signal.pair}: {str(e)}",
                "error",
            )

    async def monitor_trade(self, symbol: str):
        """Monitor trade for stop loss and take profit"""
        if symbol not in self.active_trades:
            return

        trade_data = self.active_trades[symbol]
        trade = trade_data["trade"]

        while symbol in self.active_trades:
            try:
                # Get current price
                ticker = await self.mexc_client.get_ticker(symbol)
                current_price = float(ticker["lastPrice"])

                # Check stop loss
                if trade.entry_price > trade.stop_loss:  # Long position
                    if current_price <= trade.stop_loss:
                        await self.close_position(symbol, "stop_loss", current_price)
                        break
                    elif current_price >= trade.take_profit:
                        await self.close_position(symbol, "take_profit", current_price)
                        break
                else:  # Short position
                    if current_price >= trade.stop_loss:
                        await self.close_position(symbol, "stop_loss", current_price)
                        break
                    elif current_price <= trade.take_profit:
                        await self.close_position(symbol, "take_profit", current_price)
                        break

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error monitoring trade {symbol}: {str(e)}")
                await asyncio.sleep(10)

    async def close_position(self, symbol: str, reason: str, exit_price: float):
        """Close position and send notification"""
        if symbol not in self.active_trades:
            return

        trade_data = self.active_trades[symbol]
        trade = trade_data["trade"]
        entry_time = trade_data["entry_time"]

        # Calculate P&L
        if trade.entry_price < exit_price:  # Long position
            pnl = (exit_price - trade.entry_price) * trade.position_size
            pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # Short position
            pnl = (trade.entry_price - exit_price) * trade.position_size
            pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        # Execute closing order
        side = "SELL" if trade.entry_price < exit_price else "BUY"
        await self.trade_executor.execute_market_order(
            symbol, side, trade.position_size
        )

        # Remove from active trades
        del self.active_trades[symbol]

        # Send notification based on reason
        trade_info = {
            "pair": symbol,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "duration": str(datetime.now() - entry_time),
            "volume": trade.position_size * trade.entry_price,
        }

        if reason == "stop_loss":
            trade_info["loss"] = abs(pnl)
            trade_info["loss_percent"] = abs(pnl_percent)
            message = self.bot.format_stop_loss_hit_message(trade_info)
            await self.bot.send_alert("Stop Loss Hit", message, "warning")
        else:  # take_profit
            trade_info["profit"] = pnl
            trade_info["profit_percent"] = pnl_percent
            trade_info["rr_achieved"] = trade.risk_reward
            message = self.bot.format_take_profit_hit_message(trade_info)
            await self.bot.send_alert("Take Profit Reached", message, "money")

        # Update performance tracker
        self.performance_tracker.add_trade(symbol, pnl, reason == "take_profit")

    async def get_daily_performance(self) -> PerformanceStats:
        """Get daily performance statistics"""
        try:
            account = await self.mexc_client.get_account()
            current_balance = sum(
                float(b["free"]) + float(b["locked"])
                for b in account["balances"]
                if b["asset"] == "USDT"
            )
        except Exception as e:
            logger.warning(f"Could not fetch account data: {str(e)}")
            # Use a default balance if we can't fetch from API
            current_balance = 1000.0  # Default for demo purposes

        return self.performance_tracker.get_daily_stats(current_balance)

    async def get_daily_performance_thread_safe(self) -> PerformanceStats:
        """Thread-safe version for bot commands - creates new HTTP client"""
        try:
            # Create a new MEXC client for this thread/event loop
            from mexc.mexc_client import MEXCClient
            from config.config import Config

            thread_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)

            account = await thread_client.get_account()
            current_balance = sum(
                float(b["free"]) + float(b["locked"])
                for b in account["balances"]
                if b["asset"] == "USDT"
            )

            # Close the client session properly
            await thread_client.close()

        except Exception as e:
            logger.warning(f"Could not fetch account data in thread: {str(e)}")
            # Use a default balance if we can't fetch from API
            current_balance = 1000.0  # Default for demo purposes

        return self.performance_tracker.get_daily_stats(current_balance)

    async def execute_trade_thread_safe(self, signal: Signal, position_size: float):
        """Thread-safe version for bot callbacks - creates new HTTP client"""
        try:
            # Create a new MEXC client for this thread/event loop
            from mexc.mexc_client import MEXCClient, MEXCTradeExecutor
            from config.config import Config

            thread_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
            thread_executor = MEXCTradeExecutor(thread_client)

            # Calculate position size based on available balance
            account = await thread_client.get_account()
            usdt_balance = 0
            for balance in account["balances"]:
                if balance["asset"] == "USDT":
                    usdt_balance = float(balance["free"])
                    break

            # Use minimum of requested size and 10% of available balance
            max_position = usdt_balance * 0.1
            actual_position_size = min(position_size, max_position)

            if actual_position_size < 5:  # Minimum $5 position
                raise ValueError(
                    f"Insufficient balance. Available: ${usdt_balance:.2f}, Required: $5 minimum"
                )

            # Calculate quantity for the order
            quantity = actual_position_size / signal.current_price

            # Execute the market order
            order = await thread_executor.execute_market_order(
                symbol=signal.pair, side=signal.action, quantity=quantity
            )

            # Store trade information
            trade_info = {
                "pair": signal.pair,
                "action": signal.action,
                "entry_price": signal.current_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "position_size": actual_position_size,
                "quantity": quantity,
                "order_id": order.get("orderId"),
                "timestamp": datetime.now(),
            }

            # Add to active trades (thread-safe)
            self.active_trades[signal.pair] = trade_info

            # Close the client session
            await thread_client.close()

            logger.info(
                f"Thread-safe trade executed: {signal.pair} {signal.action} ${actual_position_size:.2f}"
            )
            return trade_info

        except Exception as e:
            logger.error(f"Thread-safe trade execution failed: {str(e)}")
            # Close client on error
            try:
                await thread_client.close()
            except:
                pass
            raise

    async def send_performance_report(self):
        """Send daily performance report"""
        stats = await self.get_daily_performance()
        message = self.bot.format_performance_message(stats)
        await self.bot.send_alert("Daily Performance Summary", message, "money")

    async def start(self):
        """Start the trading system"""
        self.running = True

        # Initialize system
        await self.initialize()

        # Set up bot callbacks
        self.bot.trade_callback = self.execute_trade
        self.bot.trading_system = self

        # Start bot and performance reporting concurrently (no separate thread)
        asyncio.create_task(self.run_bot_async())
        asyncio.create_task(self.performance_report_loop())

        # Keep running
        while self.running:
            await asyncio.sleep(1)

    def setup_bot_handlers(self, application=None):
        """Setup bot command handlers"""
        from telegram.ext import CommandHandler, CallbackQueryHandler

        app = application or self.bot.application
        if app:
            app.add_handler(CommandHandler("start", self.bot.start_command))
            app.add_handler(CommandHandler("status", self.bot.status_command))
            app.add_handler(CommandHandler("performance", self.bot.performance_command))
            app.add_handler(CommandHandler("test_signal", self.bot.test_signal_command))
            app.add_handler(CallbackQueryHandler(self.bot.handle_callback))

    async def run_bot_async(self):
        """Run bot asynchronously in main event loop"""
        try:
            # Set up handlers for existing application
            self.setup_bot_handlers()

            # Initialize and start the application
            await self.bot.application.initialize()
            await self.bot.application.start()

            # Start polling
            logger.info("Bot polling started")
            await self.bot.application.updater.start_polling()

            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)

            logger.info("Bot received stop signal")

        except Exception as e:
            logger.error(f"Bot async error: {str(e)}")
        finally:
            # Clean up
            logger.info("Cleaning up bot...")
            try:
                # Stop polling first
                if (
                    hasattr(self.bot.application.updater, "_running")
                    and self.bot.application.updater._running
                ):
                    await self.bot.application.updater.stop()
                    logger.info("Bot polling stopped")
            except Exception as e:
                logger.error(f"Error stopping polling: {str(e)}")

            try:
                # Stop and shutdown application
                await self.bot.application.stop()
                await self.bot.application.shutdown()
                logger.info("Bot application shutdown complete")
            except Exception as e:
                logger.error(f"Bot cleanup error: {str(e)}")

    async def performance_report_loop(self):
        """Send performance reports at scheduled time"""
        while self.running:
            now = datetime.now()
            report_time = datetime.strptime(
                Config.PERFORMANCE_REPORT_TIME, "%H:%M"
            ).time()

            if (
                now.time().hour == report_time.hour
                and now.time().minute == report_time.minute
            ):
                await self.send_performance_report()
                await asyncio.sleep(60)  # Wait a minute to avoid duplicate reports

            await asyncio.sleep(30)  # Check every 30 seconds

    async def stop(self):
        """Stop the trading system"""
        self.running = False

        # Close all positions
        for symbol in list(self.active_trades.keys()):
            ticker = await self.mexc_client.get_ticker(symbol)
            current_price = float(ticker["lastPrice"])
            await self.close_position(symbol, "manual_close", current_price)

        # Stop data feed
        await self.data_feed.stop()

        # Send shutdown message first, before stopping bot
        try:
            await self.bot.send_alert(
                "System Stopped",
                "Trading bot has been shut down. All positions closed.",
                "warning",
            )
        except Exception as e:
            logger.error(f"Error sending shutdown message: {str(e)}")

        # Bot will stop automatically when self.running becomes False
        logger.info("Bot will stop when thread completes")


class PerformanceTracker:
    """Track trading performance"""

    def __init__(self):
        self.trades = []
        self.start_balance = None
        self.daily_start_balance = None
        self.daily_trades = []

    def add_trade(self, symbol: str, pnl: float, is_win: bool):
        """Add completed trade"""
        trade = {
            "symbol": symbol,
            "pnl": pnl,
            "is_win": is_win,
            "timestamp": datetime.now(),
        }
        self.trades.append(trade)
        self.daily_trades.append(trade)

    def get_daily_stats(self, current_balance: float) -> PerformanceStats:
        """Calculate daily performance statistics"""
        if not self.daily_start_balance:
            self.daily_start_balance = current_balance

        total_trades = len(self.daily_trades)
        wins = sum(1 for t in self.daily_trades if t["is_win"])
        win_rate = wins / total_trades if total_trades > 0 else 0

        daily_pnl = sum(t["pnl"] for t in self.daily_trades)
        daily_pnl_percent = (
            (daily_pnl / self.daily_start_balance * 100)
            if self.daily_start_balance > 0
            else 0
        )

        # Find best and worst trades
        best_trade = (
            max(self.daily_trades, key=lambda x: x["pnl"])
            if self.daily_trades
            else None
        )
        worst_trade = (
            min(self.daily_trades, key=lambda x: x["pnl"])
            if self.daily_trades
            else None
        )

        return PerformanceStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_trades=total_trades,
            win_rate=win_rate,
            pnl=daily_pnl,
            pnl_percent=daily_pnl_percent,
            best_trade=(
                {"pair": best_trade["symbol"], "profit": best_trade["pnl"]}
                if best_trade
                else {"pair": "N/A", "profit": 0}
            ),
            worst_trade=(
                {"pair": worst_trade["symbol"], "loss": abs(worst_trade["pnl"])}
                if worst_trade
                else {"pair": "N/A", "loss": 0}
            ),
            start_balance=self.daily_start_balance,
            current_balance=current_balance,
            total_return=(
                (current_balance - self.daily_start_balance)
                / self.daily_start_balance
                * 100
                if self.daily_start_balance > 0
                else 0
            ),
        )

    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trades = []
        self.daily_start_balance = None


async def main():
    """Main entry point"""
    # Validate configuration
    Config.validate()

    # Create trading system
    system = TradingSystem()

    try:
        # Run the system
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await system.stop()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        await system.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
