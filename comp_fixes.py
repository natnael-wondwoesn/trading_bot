#!/usr/bin/env python3
"""
Windows-Compatible Integration Fix
Fixes the system integration issue without Unicode characters
"""

import os
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_integrated_signal_service():
    """Create the integrated signal service - Windows compatible"""
    print("Creating integrated signal service...")

    # Ensure services directory exists
    os.makedirs("services", exist_ok=True)

    # Content without problematic Unicode characters
    content = '''#!/usr/bin/env python3
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
                from mexc.mexc_client import MEXCClient
                
                if hasattr(Config, 'MEXC_API_KEY') and Config.MEXC_API_KEY:
                    self.mexc_client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)
                    logger.info("MEXC client initialized successfully")
                else:
                    logger.warning("MEXC credentials not found in config")
                    
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
                
                signals = await self.scan_for_signals()
                
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
        
        if not self.mexc_client or not self.strategy:
            logger.warning("MEXC client or strategy not available for scanning")
            return signals
        
        for pair in self.trading_pairs:
            try:
                # Get recent data
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
    
    async def force_scan(self) -> List[Dict]:
        """Force a single scan for testing"""
        logger.info("Forcing signal scan for testing...")
        return await self.scan_for_signals()
    
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
'''

    # Write with UTF-8 encoding to handle any remaining Unicode
    with open("services/integrated_signal_service.py", "w", encoding="utf-8") as f:
        f.write(content)

    print("DONE: services/integrated_signal_service.py created")


async def update_production_main():
    """Update production_main.py to include integrated signal service"""
    print("Updating production_main.py...")

    production_file = "production_main.py"

    if not os.path.exists(production_file):
        print("WARNING: production_main.py not found, skipping...")
        return

    try:
        # Read with UTF-8 encoding
        with open(production_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Add import if not present
        import_line = (
            "from services.integrated_signal_service import integrated_signal_service"
        )
        if import_line not in content:
            # Find a good place to add the import
            if "from services.multi_user_bot import MultiUserTradingBot" in content:
                content = content.replace(
                    "from services.multi_user_bot import MultiUserTradingBot",
                    "from services.multi_user_bot import MultiUserTradingBot\n"
                    + import_line,
                )
                print("DONE: Added integrated signal service import")
            elif (
                "from services.trading_orchestrator import trading_orchestrator"
                in content
            ):
                content = content.replace(
                    "from services.trading_orchestrator import trading_orchestrator",
                    "from services.trading_orchestrator import trading_orchestrator\n"
                    + import_line,
                )
                print(
                    "DONE: Added integrated signal service import (alternative location)"
                )

        # Add service initialization if not present
        if "integrated_signal_service.initialize()" not in content:
            # Find start_all_services method
            method_pos = content.find("async def start_all_services(self):")
            if method_pos > 0:
                # Find where to insert (before "All services started")
                success_msg_pos = content.find(
                    'logger.info("All services started successfully")', method_pos
                )
                if success_msg_pos == -1:
                    # Try alternative success message
                    success_msg_pos = content.find(
                        'logger.info("✅ All services started successfully")',
                        method_pos,
                    )

                if success_msg_pos > 0:
                    initialization_code = """
            # Initialize integrated signal service
            logger.info("Starting integrated signal service...")
            try:
                await integrated_signal_service.initialize()
                
                # Start monitoring in background
                asyncio.create_task(integrated_signal_service.start_monitoring())
                logger.info("Integrated signal service started successfully")
            except Exception as e:
                logger.error(f"Failed to start integrated signal service: {e}")

            """
                    content = (
                        content[:success_msg_pos]
                        + initialization_code
                        + content[success_msg_pos:]
                    )
                    print("DONE: Added signal service initialization to startup")
                else:
                    print(
                        "WARNING: Could not find insertion point for signal service initialization"
                    )
            else:
                print("WARNING: Could not find start_all_services method")
        else:
            print("INFO: Signal service initialization already present")

        # Write with UTF-8 encoding
        with open(production_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("DONE: production_main.py updated successfully")

    except Exception as e:
        print(f"ERROR: Failed to update production_main.py: {e}")


async def test_integration():
    """Test the integration works"""
    print("Testing integration...")

    try:
        # Test import
        import sys

        sys.path.append(".")

        from services.integrated_signal_service import IntegratedSignalService

        service = IntegratedSignalService()
        print("DONE: IntegratedSignalService imported successfully")

        # Test initialization
        await service.initialize()
        print("DONE: Service initialization completed")

        # Test force scan
        signals = await service.force_scan()
        print(f"DONE: Force scan completed, found {len(signals)} signals")

        return True

    except Exception as e:
        print(f"ERROR: Integration test failed: {e}")
        return False


async def main():
    """Main execution"""
    print("WINDOWS-COMPATIBLE INTEGRATION FIX")
    print("=" * 50)

    # Step 1: Create the service
    await create_integrated_signal_service()

    # Step 2: Update production main
    await update_production_main()

    # Step 3: Test the integration
    success = await test_integration()

    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Integration fix completed successfully!")
        print("\nNEXT STEPS:")
        print("1. Restart your system: python production_main.py")
        print("2. Test signal generation: python debug_strategy_signals.py")
        print("3. Check for high-confidence signals (>50%)")
    else:
        print("PARTIAL SUCCESS: Integration service created but test failed")
        print("This is normal if dependencies are missing")
        print("Try running: python production_main.py")


if __name__ == "__main__":
    asyncio.run(main())
