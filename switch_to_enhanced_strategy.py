#!/usr/bin/env python3
"""
Switch to Enhanced Strategy
Updates the system to use the Enhanced RSI EMA Strategy
"""

import json
import os
from datetime import datetime


def update_config_strategy():
    """Update the config to use enhanced strategy"""
    config_file = "config/config.py"

    # Read current config
    with open(config_file, "r") as f:
        content = f.read()

    # Update the active strategy line
    if 'ACTIVE_STRATEGY = "RSI_EMA"' in content:
        content = content.replace(
            'ACTIVE_STRATEGY = "RSI_EMA"', 'ACTIVE_STRATEGY = "ENHANCED_RSI_EMA"'
        )

        # Write back to file
        with open(config_file, "w") as f:
            f.write(content)

        print("✅ Config updated to use ENHANCED_RSI_EMA")
        return True
    else:
        print("❌ Could not find ACTIVE_STRATEGY line in config")
        return False


def update_user_settings():
    """Update user settings to use enhanced strategy"""
    settings_file = "user_settings.json"

    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)

            # Update strategy
            settings["strategy"] = "ENHANCED_RSI_EMA"
            settings["last_updated"] = datetime.now().isoformat()

            # Write back
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=2)

            print("✅ User settings updated to use ENHANCED_RSI_EMA")
            return True

        except Exception as e:
            print(f"❌ Error updating user settings: {e}")
            return False
    else:
        print("ℹ️ User settings file not found - will use config default")
        return True


def update_main_system():
    """Update main.py to support the enhanced strategy"""
    main_file = "main.py"

    # Read current main.py
    with open(main_file, "r") as f:
        content = f.read()

    # Add import for enhanced strategy if not present
    if (
        "from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy"
        not in content
    ):
        # Find the import section
        import_line = (
            "from strategy.strategies.bollinger_strategy import BollingerStrategy"
        )
        if import_line in content:
            content = content.replace(
                import_line,
                import_line
                + "\nfrom strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy",
            )

        # Update the strategy selection logic
        original_logic = """else:  # Default to RSI_EMA
            return RSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )"""

        enhanced_logic = """elif strategy_type == "ENHANCED_RSI_EMA":
            return EnhancedRSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )
        else:  # Default to RSI_EMA
            return RSIEMAStrategy(
                rsi_period=Config.RSI_PERIOD,
                ema_fast=Config.EMA_FAST,
                ema_slow=Config.EMA_SLOW,
            )"""

        if original_logic in content:
            content = content.replace(original_logic, enhanced_logic)

        # Write back to file
        with open(main_file, "w") as f:
            f.write(content)

        print("✅ Main system updated to support Enhanced RSI EMA Strategy")
        return True
    else:
        print("ℹ️ Enhanced strategy already imported in main system")
        return True


def main():
    """Main execution"""
    print("🔄 SWITCHING TO ENHANCED RSI EMA STRATEGY")
    print("=" * 50)
    print()

    # Step 1: Update config
    print("1. Updating configuration...")
    config_success = update_config_strategy()

    # Step 2: Update user settings
    print("\n2. Updating user settings...")
    settings_success = update_user_settings()

    # Step 3: Update main system
    print("\n3. Updating main system...")
    main_success = update_main_system()

    print("\n" + "=" * 50)

    if config_success and settings_success and main_success:
        print("✅ SUCCESSFULLY SWITCHED TO ENHANCED STRATEGY!")
        print()
        print("📋 What changed:")
        print("   • More practical RSI thresholds (40/60 instead of 35/65)")
        print("   • Lower confidence requirement (0.4 instead of 0.7)")
        print("   • Weighted scoring instead of all-or-nothing")
        print("   • Less strict volume requirements")
        print("   • Multiple signal types and crossover detection")
        print()
        print("🚀 Next steps:")
        print("   1. Restart your trading bot")
        print("   2. Run the debug script: python debug_strategy_signals.py")
        print("   3. Monitor for signals over the next few hours")
        print("   4. Adjust confidence threshold if needed")
        print()
        print("⚠️ Consider paper trading first to validate the new strategy!")
    else:
        print("❌ SOME UPDATES FAILED")
        print("   Please check the error messages above and fix manually")
        print("   You may need to restart your trading system")


if __name__ == "__main__":
    main()
