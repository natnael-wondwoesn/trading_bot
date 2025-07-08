import logging
from typing import Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    current_risk: float
    daily_loss: float
    open_positions: int
    margin_level: float
    free_margin: float


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_pnl: float


class MTAccountManager:
    """Account and risk management for MetaTrader"""

    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.daily_loss = 0
        self.daily_trades = []

    async def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""
        try:
            # Get account info
            account = await self.client.get_account_info()
            positions = await self.client.get_positions()

            # Calculate metrics
            current_risk = 0
            for position in positions:
                # Calculate risk per position
                if position.get("stopLoss"):
                    risk = (
                        abs(position["openPrice"] - position["stopLoss"])
                        * position["volume"]
                    )
                    current_risk += risk

            # Calculate daily loss
            daily_loss = await self._calculate_daily_loss()

            return RiskMetrics(
                current_risk=current_risk,
                daily_loss=daily_loss,
                open_positions=len(positions),
                margin_level=account.get("marginLevel", 0),
                free_margin=account.get("freeMargin", 0),
            )

        except Exception as e:
            logger.error(f"Failed to get risk metrics: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0)

    async def _calculate_daily_loss(self) -> float:
        """Calculate today's P&L"""
        try:
            # Get today's closed trades
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)

            history = await self.client.get_history_orders(today, tomorrow)

            daily_pnl = 0
            for order in history:
                if order.get("state") == "ORDER_STATE_FILLED":
                    daily_pnl += order.get("profit", 0)

            return daily_pnl

        except Exception as e:
            logger.error(f"Failed to calculate daily loss: {str(e)}")
            return 0

    async def check_risk_limits(self) -> Dict[str, bool]:
        """Check if risk limits are exceeded"""
        try:
            metrics = await self.get_risk_metrics()
            account = await self.client.get_account_info()

            checks = {
                "daily_loss_ok": abs(metrics.daily_loss)
                < (account["balance"] * self.config.MAX_DAILY_LOSS),
                "position_limit_ok": metrics.open_positions
                < self.config.MAX_OPEN_POSITIONS,
                "margin_level_ok": metrics.margin_level
                > 200,  # 200% minimum margin level
                "free_margin_ok": metrics.free_margin > 0,
            }

            return checks

        except Exception as e:
            logger.error(f"Failed to check risk limits: {str(e)}")
            return {
                "daily_loss_ok": False,
                "position_limit_ok": False,
                "margin_level_ok": False,
                "free_margin_ok": False,
            }

    async def get_correlation_exposure(self, symbol: str) -> int:
        """Check correlated positions"""
        try:
            positions = await self.client.get_positions()
            correlated_pairs = self._get_correlated_pairs(symbol)

            correlated_count = 0
            for position in positions:
                if position["symbol"] in correlated_pairs:
                    correlated_count += 1

            return correlated_count

        except Exception as e:
            logger.error(f"Failed to check correlation: {str(e)}")
            return 0

    def _get_correlated_pairs(self, symbol: str) -> List[str]:
        """Get correlated currency pairs"""
        correlations = {
            "EURUSD": ["GBPUSD", "EURCAD", "EURGBP"],
            "GBPUSD": ["EURUSD", "GBPJPY", "EURGBP"],
            "USDJPY": ["USDCAD", "USDCHF"],
            "AUDUSD": ["NZDUSD", "AUDCAD", "AUDNZD"],
            "NZDUSD": ["AUDUSD", "NZDCAD", "AUDNZD"],
            "USDCAD": ["USDJPY", "CADJPY"],
            "USDCHF": ["USDJPY"],
            "XAUUSD": ["XAGUSD"],
            "XAGUSD": ["XAUUSD"],
        }

        return correlations.get(symbol, [])

    async def calculate_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """Calculate performance metrics for period"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()

            history = await self.client.get_history_orders(start_date, end_date)

            # Filter completed trades
            trades = [
                order for order in history if order.get("state") == "ORDER_STATE_FILLED"
            ]

            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get("profit", 0) > 0)
            losing_trades = sum(1 for t in trades if t.get("profit", 0) < 0)

            total_profit = sum(
                t.get("profit", 0) for t in trades if t.get("profit", 0) > 0
            )
            total_loss = abs(
                sum(t.get("profit", 0) for t in trades if t.get("profit", 0) < 0)
            )

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

            average_win = total_profit / winning_trades if winning_trades > 0 else 0
            average_loss = total_loss / losing_trades if losing_trades > 0 else 0

            total_pnl = sum(t.get("profit", 0) for t in trades)

            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                total_pnl=total_pnl,
            )

        except Exception as e:
            logger.error(f"Failed to calculate performance: {str(e)}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    async def get_exposure_by_currency(self) -> Dict[str, float]:
        """Calculate exposure by currency"""
        try:
            positions = await self.client.get_positions()
            exposure = {}

            for position in positions:
                symbol = position["symbol"]
                volume = position["volume"]

                # Extract base and quote currencies
                if len(symbol) >= 6:
                    base = symbol[:3]
                    quote = symbol[3:6]

                    # Add exposure
                    if position["type"] == "POSITION_TYPE_BUY":
                        exposure[base] = exposure.get(base, 0) + volume
                        exposure[quote] = exposure.get(quote, 0) - volume
                    else:
                        exposure[base] = exposure.get(base, 0) - volume
                        exposure[quote] = exposure.get(quote, 0) + volume

            return exposure

        except Exception as e:
            logger.error(f"Failed to calculate exposure: {str(e)}")
            return {}
