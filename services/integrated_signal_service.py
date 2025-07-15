#!/usr/bin/env python3
"""
Integrated Signal Service
Bridges live_signal_monitor.py functionality into production_main.py
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class IntegratedSignalService:
    """Integrated signal monitoring for production system"""
    
    def __init__(self):
        self.strategy = None
        self.mexc_client = None
        self.running = False
        self.signal_callbacks = []
        self.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "AVAXUSDT"]
        
    async def initialize(self):
        """Initialize the integrated signal service"""
        try:
            # Try to import required modules
            try:
                from strategy.strategies.enhanced_rsi_ema_strategy import EnhancedRSIEMAStrategy
                self.strategy = EnhancedRSIEMAStrategy()
                logger.info("Enhanced strategy loaded successfully")
            except ImportError:
                logger.warning("Enhanced strategy not found, using basic strategy")
                try:
                    from strategy.strategies.rsi_ema_strategy import RSIEMAStrategy
                    self.strategy = RSIEMAStrategy()
                    logger.info("Basic RSI EMA strategy loaded")
                except ImportError:
                    logger.error("No strategy available")
                    return
            
            # Try to initialize MEXC client
            try:
                from config.config import Config
                
                # Try Bybit first (more reliable)
                if hasattr(Config, 'BYBIT_API_KEY') and Config.BYBIT_API_KEY:
                    from bybit.bybit_client import BybitClient
                    self.bybit_client = BybitClient(
                        Config.BYBIT_API_KEY, 
                        Config.BYBIT_API_SECRET, 
                        testnet=Config.BYBIT_TESTNET
                    )
                    logger.info("Bybit client initialized successfully (primary)")
                    self.primary_exchange = "Bybit"
                else:
                    # Fallback to MEXC
                    from mexc.mexc_client import MEXCClient
                    if hasattr(Config, 'MEXC_API_KEY') and Config.MEXC_API_KEY:
                        self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
                        logger.info("MEXC client initialized successfully (fallback)")
                        self.primary_exchange = "MEXC"
                    else:
                        logger.warning("No exchange credentials found")
                    
            except Exception as e:
                logger.warning(f"MEXC initialization failed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize integrated signal service: {e}")
    
    def add_signal_callback(self, callback):
        """Add callback for signal notifications"""
        self.signal_callbacks.append(callback)
        logger.info(f"Added signal callback, total callbacks: {len(self.signal_callbacks)}")
    
    async def start_monitoring(self):
        """Start continuous signal monitoring"""
        if not self.strategy:
            logger.error("No strategy available for monitoring")
            return
            
        self.running = True
        logger.info("Integrated signal monitoring started")
        
        scan_count = 0
        while self.running:
            try:
                scan_count += 1
                logger.info(f"Starting signal scan #{scan_count}")
                
                signals = await self.scan_for_signals_with_accurate_prices()
                
                if signals:
                    logger.info(f"Found {len(signals)} signals in scan #{scan_count}")
                
                # Notify all callbacks
                for i, callback in enumerate(self.signal_callbacks):
                    try:
                        await callback(signals)
                    except Exception as e:
                        logger.error(f"Signal callback {i} error: {e}")
                
                # Wait before next scan
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Signal monitoring error: {e}")
                await asyncio.sleep(30)  # Shorter wait on error
    
    async def scan_for_signals(self) -> List[Dict]:
        """Scan all pairs for signals"""
        signals = []
        
        if not (hasattr(self, 'bybit_client') or hasattr(self, 'mexc_client')) or not self.strategy:
            logger.warning("MEXC client or strategy not available for scanning")
            return signals
        
        for pair in self.trading_pairs:
            try:
                # Get recent data
                if hasattr(self, 'bybit_client'):
                    klines = await self.bybit_client.get_klines(pair, "1h", 100)
                else:
                    klines = await self.mexc_client.get_klines(pair, "1h", 100)
                
                if len(klines) < 50:
                    logger.debug(f"Insufficient data for {pair}: {len(klines)} candles")
                    continue
                
                # Set pair attribute
                klines.attrs = {"pair": pair}
                
                # Generate signal
                signal = self.strategy.generate_signal(klines)
                
                if signal.action != "HOLD" and signal.confidence > 0.4:
                    signals.append({
                        'pair': pair,
                        'action': signal.action,
                        'confidence': signal.confidence,
                        'price': signal.current_price,
                        'timestamp': signal.timestamp,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit
                    })
                    
                    logger.info(f"SIGNAL: {pair} {signal.action} @ ${signal.current_price:.4f} (Confidence: {signal.confidence:.1%})")
                else:
                    logger.debug(f"{pair}: {signal.action} (Confidence: {signal.confidence:.1%})")
                
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
        
        return signals
    
    
    async def scan_for_signals_with_accurate_prices(self) -> List[Dict]:
        """Scan all pairs for signals using accurate price fetching"""
        signals = []
        
        if not (hasattr(self, 'bybit_client') or hasattr(self, 'mexc_client')) or not self.strategy:
            logger.warning("MEXC client or strategy not available for scanning")
            return signals
        
        for pair in self.trading_pairs:
            try:
                # Get recent data
                if hasattr(self, 'bybit_client'):
                    klines = await self.bybit_client.get_klines(pair, "1h", 100)
                else:
                    klines = await self.mexc_client.get_klines(pair, "1h", 100)
                
                if len(klines) < 50:
                    continue
                
                # Get accurate current price
                if hasattr(self, 'bybit_client'):
                    # Use latest price from klines for Bybit
                    accurate_price = float(klines['close'].iloc[-1])
                else:
                    accurate_price = await self.mexc_client.get_accurate_price(pair)
                
                # Update the latest price in klines data
                klines.loc[klines.index[-1], 'close'] = accurate_price
                
                # Set pair attribute
                klines.attrs = {"pair": pair}
                
                # Generate signal
                signal = self.strategy.generate_signal(klines)
                
                if signal.action != "HOLD" and signal.confidence > 0.4:
                    signals.append({
                        'pair': pair,
                        'action': signal.action,
                        'confidence': signal.confidence,
                        'price': accurate_price,  # Use accurate price
                        'timestamp': signal.timestamp,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'accurate_price': True  # Flag to indicate accurate pricing
                    })
                    
                    logger.info(f"SIGNAL: {pair} {signal.action} @ ${accurate_price:.4f} (Confidence: {signal.confidence:.1%})")
                    
                    # Send Telegram notification (same as live_signal_monitor.py)
                    try:
                        # Convert signal to format expected by send_telegram_alert
                        signal_info = {
                            'symbol': pair,
                            'action': signal.action,
                            'price': accurate_price,
                            'confidence': signal.confidence,
                            'rsi': signal.indicators.get('rsi', 0),
                            'stop_loss': signal.stop_loss or 0,
                            'take_profit': signal.take_profit or 0,
                            'timestamp': signal.timestamp
                        }
                        
                        await self.send_telegram_alert(signal_info)
                        logger.info(f"📱 Sent Telegram notification for {pair} {signal.action}")
                        
                    except Exception as telegram_error:
                        logger.error(f"Failed to send Telegram notification: {telegram_error}")
                else:
                    logger.debug(f"{pair}: {signal.action} @ ${accurate_price:.4f} (Confidence: {signal.confidence:.1%})")
                
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
        
        return signals

    async def force_scan(self) -> List[Dict]:
        """Force a single scan for testing"""
        logger.info("Forcing signal scan for testing...")
        return await self.scan_for_signals()
    
    
    async def send_telegram_alert(self, signal_info):
        """Send Telegram alert about the signal (same as live_signal_monitor.py)"""
        try:
            from bot import TradingBot
            from config.config import Config

            bot = TradingBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)

            message = f"""📈 **{signal_info['symbol']}** - {signal_info['action']}
💰 Price: ${signal_info['price']:,.4f}
🎲 Confidence: {signal_info['confidence']:.1%}
📊 RSI: {signal_info['rsi']:.1f}

🛑 Stop Loss: ${signal_info['stop_loss']:,.4f}
🎯 Take Profit: ${signal_info['take_profit']:,.4f}

⏰ {signal_info['timestamp'].strftime('%H:%M:%S')}"""

            await bot.send_alert("ENHANCED STRATEGY SIGNAL", message, "money")
            logger.info("📱 Telegram alert sent successfully")

        except Exception as e:
            logger.warning(f"⚠️ Telegram alert failed: {e}")
            # Log more details for debugging
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")

    async def stop_monitoring(self):
        """Stop signal monitoring"""
        self.running = False
        if self.mexc_client:
            try:
                await self.mexc_client.close()
            except:
                pass
        logger.info("Integrated signal monitoring stopped")


# Global service instance
integrated_signal_service = IntegratedSignalService()
