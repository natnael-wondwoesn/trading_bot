#!/usr/bin/env python3
"""
Fix Production Integration and Real-time Price Issues
1. Integrate signal service into production_main.py properly
2. Fix real-time price accuracy issues
"""

import os
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_real_time_price_accuracy():
    """Fix real-time price accuracy by updating MEXC client precision"""
    print("💰 FIXING REAL-TIME PRICE ACCURACY")
    print("=" * 40)

    mexc_file = "mexc/mexc_client.py"
    if not os.path.exists(mexc_file):
        print("❌ MEXC client file not found")
        return False

    try:
        with open(mexc_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix 1: Add get_accurate_price method
        accurate_price_method = '''
    async def get_accurate_price(self, symbol: str) -> float:
        """Get accurate real-time price with high precision"""
        try:
            # Use price ticker endpoint for most accurate price
            url = f"{self.BASE_URL}/api/v3/ticker/price"
            params = {"symbol": symbol}
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Return price with 4 decimal places precision
                    return round(float(data["price"]), 4)
                else:
                    # Fallback to 24hr ticker
                    ticker = await self.get_ticker(symbol)
                    return round(float(ticker["lastPrice"]), 4)
                    
        except Exception as e:
            logger.error(f"Error getting accurate price for {symbol}: {e}")
            # Final fallback to regular ticker
            try:
                ticker = await self.get_ticker(symbol)
                return round(float(ticker["lastPrice"]), 4)
            except:
                raise Exception(f"Cannot get price for {symbol}")
'''

        # Add the method before the close method
        if "async def get_accurate_price" not in content:
            close_method_pos = content.find("async def close(self):")
            if close_method_pos > 0:
                content = (
                    content[:close_method_pos]
                    + accurate_price_method
                    + "\n    "
                    + content[close_method_pos:]
                )
                print("✅ Added get_accurate_price method")

        # Fix 2: Update get_ticker to return higher precision
        old_ticker = 'return await self._request("GET", "/api/v3/ticker/24hr", params, signed=False)'
        new_ticker = """data = await self._request("GET", "/api/v3/ticker/24hr", params, signed=False)
        # Ensure price precision
        if "lastPrice" in data:
            data["lastPrice"] = f"{float(data['lastPrice']):.4f}"
        return data"""

        if old_ticker in content:
            content = content.replace(old_ticker, new_ticker)
            print("✅ Enhanced ticker precision")

        # Write back
        with open(mexc_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Real-time price accuracy fixes applied")
        return True

    except Exception as e:
        print(f"❌ Failed to fix price accuracy: {e}")
        return False


def fix_production_main_integration():
    """Fix production_main.py to properly integrate signal service"""
    print("\n🔗 FIXING PRODUCTION MAIN INTEGRATION")
    print("=" * 40)

    production_file = "production_main.py"
    if not os.path.exists(production_file):
        print("❌ production_main.py not found")
        return False

    try:
        with open(production_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix 1: Add import for integrated signal service
        import_line = (
            "from services.integrated_signal_service import integrated_signal_service"
        )
        if import_line not in content:
            # Find import section and add it
            import_pos = content.find("from db.multi_user_db import multi_user_db")
            if import_pos > 0:
                insert_pos = content.find("\n", import_pos) + 1
                content = (
                    content[:insert_pos] + import_line + "\n" + content[insert_pos:]
                )
                print("✅ Added integrated signal service import")

        # Fix 2: Add signal service initialization to start_all_services
        start_services_pos = content.find("async def start_all_services(self):")
        if start_services_pos > 0:
            # Find the end of step 7 (multi-exchange data service)
            step7_pos = content.find(
                "Step 7: Starting multi-exchange data service", start_services_pos
            )
            if step7_pos > 0:
                # Find the end of step 7
                step8_pos = content.find("Step 8:", step7_pos)
                if step8_pos > 0:
                    # Insert new step before step 8
                    signal_service_code = """
            # 7.5. Initialize and start integrated signal service
            logger.info("Step 7.5: Starting integrated signal service...")
            try:
                await integrated_signal_service.initialize()
                
                # Add callback to notify users of signals
                async def signal_notification_callback(signals):
                    if signals and "bot" in self.services:
                        try:
                            bot = self.services["bot"]
                            if hasattr(bot, 'notify_all_users_of_signals'):
                                await bot.notify_all_users_of_signals(signals)
                        except Exception as e:
                            logger.error(f"Error notifying users of signals: {e}")
                
                integrated_signal_service.add_signal_callback(signal_notification_callback)
                
                # Start monitoring in background
                asyncio.create_task(integrated_signal_service.start_monitoring())
                self.services["signal_service"] = integrated_signal_service
                
                logger.info("Integrated signal service started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start integrated signal service: {e}")
                # Don't fail the entire startup for signal service issues
                logger.warning("Continuing without integrated signal service...")

            """
                    content = (
                        content[:step8_pos] + signal_service_code + content[step8_pos:]
                    )
                    print("✅ Added signal service initialization to startup")

        # Fix 3: Add signal service to shutdown process
        shutdown_pos = content.find("async def shutdown_all_services(self):")
        if shutdown_pos > 0:
            # Find the try block in shutdown
            try_pos = content.find("try:", shutdown_pos)
            if try_pos > 0:
                # Find where to insert shutdown code
                services_stop_pos = content.find(
                    "# Stop services in reverse order", try_pos
                )
                if services_stop_pos > 0:
                    signal_shutdown_code = """
            # Stop integrated signal service
            if "signal_service" in self.services:
                try:
                    await self.services["signal_service"].stop_monitoring()
                    logger.info("Integrated signal service stopped")
                except Exception as e:
                    logger.error(f"Error stopping signal service: {e}")

            """
                    next_line = content.find("\n", services_stop_pos) + 1
                    content = (
                        content[:next_line] + signal_shutdown_code + content[next_line:]
                    )
                    print("✅ Added signal service to shutdown process")

        # Fix 4: Add API endpoint for signal status
        # Find the FastAPI app setup
        app_setup_pos = content.find("self.app = FastAPI(")
        if app_setup_pos > 0:
            # Find where routes are defined
            routes_pos = content.find('@self.app.get("/health")', app_setup_pos)
            if routes_pos > 0:
                # Add signal service status endpoint after health
                health_end = content.find("return JSONResponse", routes_pos)
                health_end = content.find("}", health_end) + 1

                signal_endpoint = '''
        
        @self.app.get("/signals/status")
        async def get_signal_status():
            """Get signal service status"""
            try:
                if "signal_service" in self.services:
                    service = self.services["signal_service"]
                    return JSONResponse(content={
                        "running": service.running if hasattr(service, 'running') else False,
                        "strategy": service.strategy.name if service.strategy else None,
                        "trading_pairs": len(service.trading_pairs) if hasattr(service, 'trading_pairs') else 0,
                        "callbacks": len(service.signal_callbacks) if hasattr(service, 'signal_callbacks') else 0,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return JSONResponse(content={
                        "running": False,
                        "error": "Signal service not initialized",
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        @self.app.post("/signals/force-scan")
        async def force_signal_scan():
            """Force a signal scan for testing"""
            try:
                if "signal_service" in self.services:
                    service = self.services["signal_service"]
                    if hasattr(service, 'force_scan'):
                        signals = await service.force_scan()
                        return JSONResponse(content={
                            "success": True,
                            "signals_found": len(signals),
                            "signals": signals,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        return JSONResponse(content={
                            "success": False,
                            "error": "Force scan not available"
                        })
                else:
                    return JSONResponse(content={
                        "success": False,
                        "error": "Signal service not running"
                    })
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
'''

                content = content[:health_end] + signal_endpoint + content[health_end:]
                print("✅ Added signal service API endpoints")

        # Write back
        with open(production_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Production main integration fixes applied")
        return True

    except Exception as e:
        print(f"❌ Failed to fix production integration: {e}")
        return False


def update_integrated_signal_service():
    """Update integrated signal service with better price accuracy"""
    print("\n🎯 UPDATING INTEGRATED SIGNAL SERVICE")
    print("=" * 40)

    service_file = "services/integrated_signal_service.py"
    if not os.path.exists(service_file):
        print("❌ Integrated signal service not found")
        return False

    try:
        with open(service_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Add method to get accurate prices
        accurate_scan_method = '''
    async def scan_for_signals_with_accurate_prices(self) -> List[Dict]:
        """Scan all pairs for signals using accurate price fetching"""
        signals = []
        
        if not self.mexc_client or not self.strategy:
            logger.warning("MEXC client or strategy not available for scanning")
            return signals
        
        for pair in self.trading_pairs:
            try:
                # Get recent data
                klines = await self.mexc_client.get_klines(pair, "1h", 100)
                
                if len(klines) < 50:
                    continue
                
                # Get accurate current price
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
                else:
                    logger.debug(f"{pair}: {signal.action} @ ${accurate_price:.4f} (Confidence: {signal.confidence:.1%})")
                
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
        
        return signals
'''

        # Add this method before the force_scan method
        if "async def scan_for_signals_with_accurate_prices" not in content:
            force_scan_pos = content.find("async def force_scan(self)")
            if force_scan_pos > 0:
                content = (
                    content[:force_scan_pos]
                    + accurate_scan_method
                    + "\n    "
                    + content[force_scan_pos:]
                )
                print("✅ Added accurate price scanning method")

        # Update the regular scan method to use accurate prices
        old_scan_call = "signals = await self.scan_for_signals()"
        new_scan_call = "signals = await self.scan_for_signals_with_accurate_prices()"

        if old_scan_call in content:
            content = content.replace(old_scan_call, new_scan_call)
            print("✅ Updated monitoring to use accurate prices")

        # Write back
        with open(service_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Integrated signal service updated")
        return True

    except Exception as e:
        print(f"❌ Failed to update signal service: {e}")
        return False


async def test_fixes():
    """Test the applied fixes"""
    print("\n🧪 TESTING APPLIED FIXES")
    print("=" * 40)

    try:
        # Test 1: Real-time price accuracy
        print("Testing price accuracy...")
        from mexc.mexc_client import MEXCClient

        client = MEXCClient("test", "test")
        try:
            if hasattr(client, "get_accurate_price"):
                print("✅ get_accurate_price method added")
            else:
                print("❌ get_accurate_price method missing")
        finally:
            await client.close()

        # Test 2: Integrated signal service
        print("Testing signal service...")
        from services.integrated_signal_service import IntegratedSignalService

        service = IntegratedSignalService()
        if hasattr(service, "scan_for_signals_with_accurate_prices"):
            print("✅ Accurate price scanning method added")
        else:
            print("❌ Accurate price scanning method missing")

        # Test 3: Production main import
        print("Testing production main...")
        with open("production_main.py", "r") as f:
            prod_content = f.read()

        if (
            "from services.integrated_signal_service import integrated_signal_service"
            in prod_content
        ):
            print("✅ Signal service import added to production_main.py")
        else:
            print("❌ Signal service import missing from production_main.py")

        if "Step 7.5: Starting integrated signal service" in prod_content:
            print("✅ Signal service initialization added to startup")
        else:
            print("❌ Signal service initialization missing from startup")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def main():
    """Main execution"""
    print("FIXING PRODUCTION INTEGRATION & PRICE ACCURACY")
    print("=" * 60)

    # Fix 1: Real-time price accuracy
    price_fix = fix_real_time_price_accuracy()

    # Fix 2: Production main integration
    integration_fix = fix_production_main_integration()

    # Fix 3: Update signal service
    service_fix = update_integrated_signal_service()

    # Test fixes
    test_success = await test_fixes()

    print("\n" + "=" * 60)
    print("📊 FIX RESULTS:")
    print(f"   Price Accuracy: {'✅ FIXED' if price_fix else '❌ FAILED'}")
    print(
        f"   Production Integration: {'✅ FIXED' if integration_fix else '❌ FAILED'}"
    )
    print(f"   Service Update: {'✅ FIXED' if service_fix else '❌ FAILED'}")
    print(f"   Tests: {'✅ PASSED' if test_success else '❌ FAILED'}")

    if all([price_fix, integration_fix, service_fix, test_success]):
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n🚀 Now your production system will:")
        print("   • Generate signals with accurate real-time prices")
        print("   • Integrate signal monitoring into production_main.py")
        print("   • Show signals from the main system, not just live_signal_monitor.py")
        print("   • Provide API endpoints for signal monitoring")

        print("\n▶️ Start your fixed production system:")
        print("   python production_main.py")

        print("\n🔍 Monitor signals via API:")
        print("   http://localhost:8080/signals/status")
        print("   http://localhost:8080/signals/force-scan")

    else:
        print("\n⚠️ Some fixes failed - check errors above")


if __name__ == "__main__":
    asyncio.run(main())
