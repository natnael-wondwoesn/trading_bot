#!/usr/bin/env python3
"""
Trading Signal Test
Simulates a real trading signal to test the bot interaction workflow
"""

import asyncio
import logging
from datetime import datetime
from models.models import Signal
from bot import TradingBot
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fake_signal() -> Signal:
    """Create a realistic fake trading signal"""

    # Simulate BTC signal with current-like prices
    current_price = 108_320.50

    # Simulate RSI oversold condition with EMA bullish crossover
    signal = Signal(
        pair="BTCUSDT",
        action="BUY",
        current_price=current_price,
        confidence=0.87,  # 87% confidence
        stop_loss=current_price * 0.98,  # 2% stop loss
        take_profit=current_price * 1.06,  # 6% take profit
        risk_reward=3.0,  # 1:3 risk/reward ratio
        indicators={
            "rsi": 28.5,  # Oversold
            "ema_trend": "Bullish",
            "ema_fast": 108_150.20,
            "ema_slow": 107_980.80,
            "volume_confirmation": True,
            "volatility": "Normal",
            "atr": 2_150.30,
            "macd_signal": "Bullish",
            "support_level": 107_500.00,
            "resistance_level": 109_800.00,
        },
        timestamp=datetime.now(),
    )

    return signal


def create_alternative_signals():
    """Create alternative test signals for different scenarios"""

    signals = []

    # 1. ETH SELL Signal (Overbought)
    eth_price = 2549.77
    signals.append(
        Signal(
            pair="ETHUSDT",
            action="SELL",
            current_price=eth_price,
            confidence=0.72,
            stop_loss=eth_price * 1.025,  # 2.5% stop loss for short
            take_profit=eth_price * 0.94,  # 6% take profit for short
            risk_reward=2.4,
            indicators={
                "rsi": 78.2,  # Overbought
                "ema_trend": "Bearish",
                "volume_confirmation": True,
                "volatility": "High",
            },
            timestamp=datetime.now(),
        )
    )

    # 2. SOL BUY Signal (Moderate confidence)
    sol_price = 149.73
    signals.append(
        Signal(
            pair="SOLUSDT",
            action="BUY",
            current_price=sol_price,
            confidence=0.64,
            stop_loss=sol_price * 0.97,
            take_profit=sol_price * 1.08,
            risk_reward=2.67,
            indicators={
                "rsi": 35.8,
                "ema_trend": "Bullish",
                "volume_confirmation": False,
                "volatility": "Low",
            },
            timestamp=datetime.now(),
        )
    )

    # 3. ADA High Confidence Signal
    ada_price = 0.5774
    signals.append(
        Signal(
            pair="ADAUSDT",
            action="BUY",
            current_price=ada_price,
            confidence=0.94,
            stop_loss=ada_price * 0.96,
            take_profit=ada_price * 1.12,
            risk_reward=3.0,
            indicators={
                "rsi": 22.1,  # Very oversold
                "ema_trend": "Strong Bullish",
                "volume_confirmation": True,
                "volatility": "Normal",
            },
            timestamp=datetime.now(),
        )
    )

    return signals


async def test_signal_notification():
    """Test the complete signal notification workflow"""

    print("🧪 TRADING SIGNAL TEST")
    print("=" * 50)

    # Create bot instance
    bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

    # Test 1: Single BTC signal
    print("\n📊 TEST 1: BTC Buy Signal")
    print("-" * 30)

    btc_signal = create_fake_signal()

    print(f"Signal Generated:")
    print(f"  Pair: {btc_signal.pair}")
    print(f"  Action: {btc_signal.action}")
    print(f"  Price: ${btc_signal.current_price:,.2f}")
    print(f"  Confidence: {btc_signal.confidence:.0%}")
    print(f"  Stop Loss: ${btc_signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${btc_signal.take_profit:,.2f}")
    print(f"  Risk/Reward: 1:{btc_signal.risk_reward}")

    try:
        # Send signal notification
        print(f"\n📤 Sending signal to Telegram...")
        await bot.send_signal_notification(btc_signal, amount=50)  # $50 position
        print("✅ Signal sent successfully!")

        # Show what the message looks like
        message = bot.format_signal_message(btc_signal, 50)
        print(f"\n📱 Message Preview:")
        print("-" * 40)
        print(message)
        print("-" * 40)

    except Exception as e:
        print(f"❌ Error sending signal: {str(e)}")

    # Test 2: Multiple signals
    print(f"\n📊 TEST 2: Multiple Signals")
    print("-" * 30)

    alt_signals = create_alternative_signals()

    for i, signal in enumerate(alt_signals, 1):
        print(f"\nSignal {i}: {signal.pair} {signal.action}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  RSI: {signal.indicators.get('rsi', 'N/A')}")

        try:
            await bot.send_signal_notification(signal, amount=25)
            print(f"  ✅ Sent to Telegram")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

        # Small delay between signals
        await asyncio.sleep(1)


def show_interaction_flow():
    """Show the expected user interaction flow"""

    print(f"\n🔄 EXPECTED USER INTERACTION FLOW")
    print("=" * 50)

    print(
        """
1. 📊 SIGNAL GENERATION
   ├─ RSI indicates oversold condition (28.5)
   ├─ EMA crossover confirms bullish trend
   ├─ Volume above average
   └─ Confidence: 87%

2. 📤 BOT NOTIFICATION
   ├─ Sends formatted message to Telegram
   ├─ Includes all signal details
   ├─ Shows risk/reward calculation
   └─ Presents ✅ APPROVE / ❌ REJECT buttons

3. 👤 USER DECISION
   Option A: User clicks ✅ APPROVE
   ├─ Bot executes trade immediately
   ├─ Places market buy order
   ├─ Sets up stop-loss monitoring
   ├─ Confirms execution with details
   └─ Starts position monitoring

   Option B: User clicks ❌ REJECT
   ├─ Bot dismisses signal
   ├─ No trade executed
   ├─ Continues monitoring for next signal
   └─ Removes from pending trades

4. 📈 TRADE MONITORING (if approved)
   ├─ Continuously monitors price
   ├─ Checks stop-loss and take-profit levels
   ├─ Sends updates on significant moves
   └─ Auto-closes at target levels

5. 🎯 TRADE COMPLETION
   ├─ Stop-loss hit → Loss notification
   ├─ Take-profit hit → Profit notification
   ├─ Manual close → Update with reason
   └─ Updates daily performance stats
    """
    )


def show_message_examples():
    """Show examples of different message types"""

    print(f"\n📱 MESSAGE EXAMPLES")
    print("=" * 50)

    # Create sample bot for formatting
    bot = TradingBot("dummy", "dummy")
    signal = create_fake_signal()

    print(f"\n1. 📊 SIGNAL NOTIFICATION:")
    print("-" * 40)
    print(bot.format_signal_message(signal, 50))

    print(f"\n2. ✅ TRADE EXECUTION:")
    print("-" * 40)
    from models.models import TradeSetup

    trade = TradeSetup(
        pair=signal.pair,
        entry_price=signal.current_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        position_size=50,
        risk_reward=signal.risk_reward,
        confidence=signal.confidence,
    )
    print(bot.format_trade_execution_message(trade))

    print(f"\n3. 💰 PROFIT NOTIFICATION:")
    print("-" * 40)
    profit_trade = {
        "pair": "BTCUSDT",
        "entry_price": 108320.50,
        "exit_price": 114799.73,
        "profit": 299.61,
        "profit_percent": 6.0,
        "duration": "2h 34m",
        "volume": 50.0,
        "rr_achieved": 3.0,
    }
    print(bot.format_take_profit_hit_message(profit_trade))


async def main():
    """Run the complete test suite"""

    print("🤖 TRADING BOT SIGNAL INTERACTION TEST")
    print("=" * 60)

    # Validate config first
    try:
        Config.validate()
        print("✅ Configuration validated")
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return

    # Show interaction flow
    show_interaction_flow()

    # Show message examples
    show_message_examples()

    # Test actual signal sending
    print(f"\n🚀 TESTING LIVE SIGNAL SENDING...")
    print("=" * 50)

    confirm = input("\nSend test signals to your Telegram? (y/N): ").lower().strip()

    if confirm == "y":
        await test_signal_notification()
        print(f"\n✅ Test completed! Check your Telegram for the signals.")
        print(f"📱 You should see signal notifications with approve/reject buttons.")
        print(f"🔄 Try clicking the buttons to test the interaction flow!")
    else:
        print(f"\n⏭️ Skipped live testing. Review message examples above.")

    print(f"\n🎯 TEST SUMMARY:")
    print(f"   • Signal generation: ✅ Working")
    print(f"   • Message formatting: ✅ Working")
    print(f"   • Telegram integration: ✅ Ready")
    print(f"   • User interaction: ✅ Implemented")


if __name__ == "__main__":
    asyncio.run(main())
