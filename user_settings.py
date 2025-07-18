import json
import os
from typing import Dict, Optional
from datetime import datetime


class UserSettings:
    """Manage user settings for trading preferences and risk management"""

    def __init__(self, settings_file: str = "user_settings.json"):
        self.settings_file = settings_file
        self.default_settings = {
            "strategy": "ENHANCED_RSI_EMA",  # ENHANCED_RSI_EMA, RSI_EMA, MACD, BOLLINGER
            "mexc_auto_mode": False,  # New: MEXC automated trading mode with $5 max
            "risk_management": {
                "max_risk_per_trade": 0.02,  # 2%
                "stop_loss_atr": 2.0,
                "take_profit_atr": 3.0,
                "max_open_positions": 5,
                "emergency_stop": False,
                "trading_enabled": True,
                "custom_stop_loss": None,  # Custom % if set
                "custom_take_profit": None,  # Custom % if set
                "mexc_max_volume": 5.0,  # $5 maximum per trade in MEXC auto mode
            },
            "notifications": {
                "signal_alerts": True,
                "trade_execution": True,
                "risk_warnings": True,
            },
            "emergency": {
                "emergency_mode": False,
                "auto_close_on_loss": False,
                "max_daily_loss": 0.05,  # 5%
                "emergency_contact": None,
            },
            "last_updated": datetime.now().isoformat(),
        }
        self.settings = self.load_settings()

    def load_settings(self) -> Dict:
        """Load settings from file or create defaults"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_settings(self.default_settings, settings)
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.default_settings.copy()
        else:
            return self.default_settings.copy()

    def save_settings(self):
        """Save current settings to file"""
        try:
            self.settings["last_updated"] = datetime.now().isoformat()
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _merge_settings(self, defaults: Dict, user_settings: Dict) -> Dict:
        """Merge user settings with defaults to ensure all keys exist"""
        result = defaults.copy()
        for key, value in user_settings.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        return result

    # Strategy Settings
    def get_strategy(self) -> str:
        return self.settings["strategy"]

    def set_strategy(self, strategy: str):
        if strategy in ["ENHANCED_RSI_EMA", "RSI_EMA", "MACD", "BOLLINGER"]:
            self.settings["strategy"] = strategy
            self.save_settings()
            return True
        return False

    def get_strategy_options(self) -> Dict[str, str]:
        """Get available strategy options with descriptions"""
        return {
            "ENHANCED_RSI_EMA": "Enhanced RSI + EMA - Improved signal generation with better market adaptability",
            "RSI_EMA": "RSI + EMA - Basic RSI and EMA combination strategy",
            "MACD": "MACD Strategy - Moving Average Convergence Divergence signals",
            "BOLLINGER": "Bollinger Bands - Mean reversion and volatility breakout strategy",
        }

    # Risk Management
    def get_risk_settings(self) -> Dict:
        return self.settings["risk_management"]

    def set_max_risk_per_trade(self, risk_percent: float):
        if 0.001 <= risk_percent <= 0.1:  # 0.1% to 10%
            self.settings["risk_management"]["max_risk_per_trade"] = risk_percent
            self.save_settings()
            return True
        return False

    def set_stop_loss_atr(self, atr_multiplier: float):
        if 0.5 <= atr_multiplier <= 5.0:
            self.settings["risk_management"]["stop_loss_atr"] = atr_multiplier
            self.save_settings()
            return True
        return False

    def set_take_profit_atr(self, atr_multiplier: float):
        if 1.0 <= atr_multiplier <= 10.0:
            self.settings["risk_management"]["take_profit_atr"] = atr_multiplier
            self.save_settings()
            return True
        return False

    def set_custom_stop_loss(self, percentage: Optional[float]):
        if percentage is None or (0.5 <= percentage <= 20.0):
            self.settings["risk_management"]["custom_stop_loss"] = percentage
            self.save_settings()
            return True
        return False

    def set_custom_take_profit(self, percentage: Optional[float]):
        if percentage is None or (1.0 <= percentage <= 50.0):
            self.settings["risk_management"]["custom_take_profit"] = percentage
            self.save_settings()
            return True
        return False

    def set_max_open_positions(self, max_positions: int):
        if 1 <= max_positions <= 20:
            self.settings["risk_management"]["max_open_positions"] = max_positions
            self.save_settings()
            return True
        return False

    # Emergency Functions
    def enable_emergency_stop(self):
        self.settings["risk_management"]["emergency_stop"] = True
        self.settings["emergency"]["emergency_mode"] = True
        self.save_settings()

    def disable_emergency_stop(self):
        self.settings["risk_management"]["emergency_stop"] = False
        self.settings["emergency"]["emergency_mode"] = False
        self.save_settings()

    def set_trading_enabled(self, enabled: bool):
        self.settings["risk_management"]["trading_enabled"] = enabled
        self.save_settings()

    def is_trading_enabled(self) -> bool:
        return (
            self.settings["risk_management"]["trading_enabled"]
            and not self.settings["risk_management"]["emergency_stop"]
        )

    def is_emergency_mode(self) -> bool:
        return self.settings["emergency"]["emergency_mode"]

    def set_max_daily_loss(self, loss_percent: float):
        if 0.01 <= loss_percent <= 0.5:  # 1% to 50%
            self.settings["emergency"]["max_daily_loss"] = loss_percent
            self.save_settings()
            return True
        return False

    # MEXC Automated Trading Mode
    def enable_mexc_auto_mode(self):
        """Enable MEXC automated trading mode with $5 maximum volume"""
        self.settings["mexc_auto_mode"] = True
        self.save_settings()
        return True

    def disable_mexc_auto_mode(self):
        """Disable MEXC automated trading mode"""
        self.settings["mexc_auto_mode"] = False
        self.save_settings()
        return True

    def is_mexc_auto_mode(self) -> bool:
        """Check if MEXC automated trading mode is enabled"""
        return self.settings.get("mexc_auto_mode", False)

    def get_mexc_max_volume(self) -> float:
        """Get maximum volume for MEXC automated trading"""
        return self.settings["risk_management"].get("mexc_max_volume", 5.0)

    def set_mexc_max_volume(self, max_volume: float):
        """Set maximum volume for MEXC automated trading (1-10 USD)"""
        if 1.0 <= max_volume <= 10.0:
            self.settings["risk_management"]["mexc_max_volume"] = max_volume
            self.save_settings()
            return True
        return False

    # Notification Settings
    def toggle_signal_alerts(self):
        current = self.settings["notifications"]["signal_alerts"]
        self.settings["notifications"]["signal_alerts"] = not current
        self.save_settings()
        return not current

    def toggle_trade_execution_alerts(self):
        current = self.settings["notifications"]["trade_execution"]
        self.settings["notifications"]["trade_execution"] = not current
        self.save_settings()
        return not current

    def get_settings_summary(self) -> str:
        """Get formatted settings summary"""
        risk = self.settings["risk_management"]
        emergency = self.settings["emergency"]

        strategy_names = {
            "RSI_EMA": "RSI + EMA",
            "MACD": "MACD",
            "BOLLINGER": "Bollinger Bands",
        }

        return f"""📊 **CURRENT SETTINGS**

🔧 **Strategy**: {strategy_names.get(self.settings['strategy'], self.settings['strategy'])}
📈 **Trading**: {'✅ Enabled' if risk['trading_enabled'] else '❌ Disabled'}
🤖 **MEXC Auto Mode**: {'✅ Enabled ($' + str(risk.get('mexc_max_volume', 5.0)) + ' max)' if self.settings.get('mexc_auto_mode', False) else '❌ Disabled'}
🚨 **Emergency Mode**: {'🔴 ACTIVE' if emergency['emergency_mode'] else '✅ Normal'}

💰 **Risk Management**:
• Max Risk per Trade: {risk['max_risk_per_trade']*100:.1f}%
• Stop Loss: {risk['custom_stop_loss'] or f"{risk['stop_loss_atr']}x ATR"}{'%' if risk['custom_stop_loss'] else ''}
• Take Profit: {risk['custom_take_profit'] or f"{risk['take_profit_atr']}x ATR"}{'%' if risk['custom_take_profit'] else ''}
• Max Open Positions: {risk['max_open_positions']}

🚨 **Emergency Settings**:
• Max Daily Loss: {emergency['max_daily_loss']*100:.1f}%
• Auto Close on Loss: {'✅' if emergency['auto_close_on_loss'] else '❌'}

📱 **Notifications**:
• Signal Alerts: {'✅' if self.settings['notifications']['signal_alerts'] else '❌'}
• Trade Execution: {'✅' if self.settings['notifications']['trade_execution'] else '❌'}
• Risk Warnings: {'✅' if self.settings['notifications']['risk_warnings'] else '❌'}

🕐 Last Updated: {self.settings['last_updated'][:19]}"""

    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.settings = self.default_settings.copy()
        self.save_settings()


# Global settings instance
user_settings = UserSettings()
