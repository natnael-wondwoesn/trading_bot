# 🤖 MEXC Automated Trading System

Simple button-based automated trading system for MEXC exchange with $5 maximum trade volume for safe trading.

## 🚀 Quick Start

1. **Install Requirements**
   ```bash
   pip install python-telegram-bot aiohttp pandas python-dotenv
   ```

2. **Setup Configuration**
   Create a `.env` file with your credentials:
   ```
   MEXC_API_KEY=your_mexc_api_key
   MEXC_API_SECRET=your_mexc_api_secret
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

3. **Run the Trader**
   ```bash
   python run_mexc_trader.py
   ```

## 📱 How to Get Credentials

### MEXC API Keys
1. Go to [MEXC API Management](https://mexc.com/user/api)
2. Create new API key with "Spot Trading" permissions
3. Copy the API Key and Secret

### Telegram Bot Token
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the bot token

### Telegram Chat ID
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Send `/start`
3. Copy your Chat ID number

## 💰 Trading Features

- **Maximum Trade Volume**: $5 per trade (risk-controlled)
- **Automated Signals**: Enhanced RSI/EMA strategy
- **Button Interface**: Simple approve/reject buttons
- **Stop Loss**: Automatic risk management
- **Real-time**: Live market data from MEXC
- **Safe Trading**: Built-in position limits

## 📊 Monitored Pairs

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- ADAUSDT (Cardano)
- DOGEUSDT (Dogecoin)
- SOLUSDT (Solana)

## 🎮 Telegram Commands

- `/start` - Start the trading system
- `/status` - Check system status
- `/balance` - Show account balance
- `/scan` - Manual signal scan
- `/stop` - Stop the trader

## 🔄 How It Works

1. **🔍 Signal Detection**: System continuously scans markets for trading opportunities
2. **📱 Notification**: You receive Telegram messages with signal details
3. **✅ User Approval**: Click TRADE button to approve or REJECT to skip
4. **💰 Trade Execution**: System executes trade with $5 maximum volume
5. **📈 Monitoring**: Automatic stop-loss and take-profit management

## ⚠️ Safety Features

- **$5 Maximum**: Each trade limited to $5 to minimize risk
- **Stop Loss**: Automatic 2% stop-loss on all trades
- **Take Profit**: Automatic 4% take-profit targets
- **Rate Limiting**: Prevents overtrading (5-minute cooldown)
- **Balance Check**: Verifies sufficient funds before trading
- **High Confidence**: Only signals with 60%+ confidence

## 📋 Example Usage

1. Run `python run_mexc_trader.py`
2. System starts monitoring markets
3. You receive a signal: "🚨 BTCUSDT BUY Signal - 85% confidence"
4. Click "✅ TRADE $5" button
5. System executes trade and sends confirmation
6. Automatic stop-loss and take-profit monitoring begins

## 🛡️ Risk Disclaimer

- **Educational Purpose**: This system is for educational use
- **Risk of Loss**: Trading involves risk of financial loss
- **Start Small**: Use small amounts to test the system
- **Monitor Trades**: Always monitor your active positions
- **Not Financial Advice**: This is not financial advice

## 🔧 Troubleshooting

### "Missing API Keys" Error
- Check your `.env` file exists and contains all required variables
- Verify API keys are correct and have trading permissions

### "Insufficient Balance" Error
- Ensure you have at least $5 USDT in your MEXC account
- Check API permissions include spot trading

### "Bot Not Responding" Error
- Verify bot token is correct
- Make sure you've sent `/start` to your bot first
- Check chat ID is correct

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Verify all configuration is correct
3. Test with small amounts first
4. Review logs in the `logs/` directory

## 🔄 Updates

The system automatically:
- ✅ Enforces $5 maximum trade volume
- ✅ Uses enhanced RSI/EMA strategy
- ✅ Provides simple button interface
- ✅ Includes comprehensive error handling
- ✅ Logs all activities for debugging

---

**Happy Trading! 🚀**

*Remember: Only trade with money you can afford to lose.* 