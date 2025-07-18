#!/usr/bin/env python3
"""
Production Main Entry Point for Multi-User Trading Service
Orchestrates all services for 24/7 operation
"""

import asyncio
import logging
import signal
import sys
import os

from pathlib import Path
from datetime import datetime, timedelta
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Import all services
from services.user_service import user_service
from services.multi_user_bot import MultiUserTradingBot
from services.integrated_signal_service import integrated_signal_service
from services.trading_orchestrator import trading_orchestrator
from services.monitoring_service import monitoring_service
from services.multi_exchange_data_service import multi_exchange_data_service
from db.multi_user_db import multi_user_db

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_service.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class ProductionTradingService:
    """Main production service orchestrator"""

    def __init__(self):
        self.services = {}
        self.running = False
        self.startup_time = None

        # Get configuration from environment
        self.config = {
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "webhook_url": os.getenv("WEBHOOK_URL"),
            "max_concurrent_users": int(os.getenv("MAX_CONCURRENT_USERS", 1000)),
            "database_url": os.getenv("DATABASE_URL", "./data/trading_service.db"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

        # Validate required configuration
        if not self.config["telegram_bot_token"]:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        # Initialize FastAPI for health checks and monitoring
        self.app = FastAPI(
            title="Multi-User Trading Service",
            description="24/7 Trading Bot Service for Thousands of Users",
            version="1.0.0",
        )

        self._setup_api_routes()

    def _setup_api_routes(self):
        """Setup FastAPI routes for monitoring and health checks"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            if not self.running:
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "message": "Service not running"}
        
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
,
                )

            try:
                health_status = monitoring_service.get_health_status()
                return JSONResponse(content=health_status)
            except Exception as e:
                return JSONResponse(
                    status_code=500, content={"status": "error", "message": str(e)}
                )

        @self.app.get("/stats")
        async def get_stats():
            """Get system statistics"""
            try:
                # Collect stats from all services
                stats = {
                    "service_uptime": (
                        (datetime.now() - self.startup_time).total_seconds()
                        if self.startup_time
                        else 0
                    ),
                    "bot_stats": (
                        self.services.get("bot", {}).get_system_stats()
                        if "bot" in self.services
                        else {}
                    ),
                    "orchestrator_stats": trading_orchestrator.get_system_stats(),
                    "user_service_stats": {
                        "active_users": await multi_user_db.get_active_users_count(),
                    },
                    "monitoring_stats": monitoring_service.get_health_status(),
                }
                return JSONResponse(content=stats)
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.get("/alerts")
        async def get_alerts():
            """Get recent alerts"""
            try:
                alerts = monitoring_service.get_alerts(limit=20)
                return JSONResponse(content={"alerts": alerts})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.post("/admin/maintenance")
        async def toggle_maintenance():
            """Toggle maintenance mode"""
            try:
                if "bot" in self.services:
                    bot = self.services["bot"]
                    bot.maintenance_mode = not bot.maintenance_mode
                    return JSONResponse(
                        content={
                            "maintenance_mode": bot.maintenance_mode,
                            "message": "Maintenance mode toggled",
                        }
                    )
                else:
                    return JSONResponse(
                        status_code=404, content={"error": "Bot service not available"}
                    )
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.get("/users/overview")
        async def get_users_overview():
            """Get user overview statistics"""
            try:
                system_health = await multi_user_db.get_system_health()
                return JSONResponse(content=system_health)
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.get("/market/prices")
        async def get_current_prices():
            """Get current market prices for all monitored pairs"""
            try:
                prices = multi_exchange_data_service.get_current_prices()
                stats = multi_exchange_data_service.get_service_stats()
                return JSONResponse(
                    content={
                        "prices": prices,
                        "exchange_status": stats["exchanges"],
                        "service_status": {
                            "running": stats["running"],
                            "trading_pairs": stats["trading_pairs"],
                            "mexc_updates": stats["mexc_updates"],
                            "bybit_updates": stats["bybit_updates"],
                            "signals_generated": stats["signals_generated"],
                            "uptime_seconds": stats["uptime_seconds"],
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Error getting market prices: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.post("/admin/force_signals")
        async def force_signal_generation():
            """Force signal generation for all pairs (testing)"""
            try:
                result = await self._force_signal_generation()
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Error forcing signals: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.get("/exchanges/status")
        async def get_exchange_status():
            """Get status of all exchanges"""
            try:
                # Get exchange status from multi-exchange service
                if multi_exchange_data_service.initialized:
                    exchange_status = multi_exchange_data_service.get_exchange_status()
                    service_stats = multi_exchange_data_service.get_service_stats()

                    return JSONResponse(
                        content={
                            "exchanges": exchange_status,
                            "service_running": service_stats["running"],
                            "trading_pairs": service_stats["trading_pairs"],
                            "mexc_updates": service_stats["mexc_updates"],
                            "bybit_updates": service_stats["bybit_updates"],
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    # Fallback to factory if service not initialized
                    from exchange_factory import ExchangeFactory

                    available_exchanges = ExchangeFactory.get_available_exchanges()

                    return JSONResponse(
                        content={
                            "exchanges": {
                                name: {
                                    "available": cap.available,
                                    "api_configured": cap.api_configured,
                                    "error": cap.error_message,
                                }
                                for name, cap in available_exchanges.items()
                            },
                            "service_running": False,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            except Exception as e:
                logger.error(f"Error getting exchange status: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

    async def start_all_services(self):
        """Start all services in the correct order"""
        logger.info("Starting Multi-User Trading Service...")
        self.startup_time = datetime.now()

        try:
            # 1. Initialize database first
            logger.info("Step 1: Initializing database...")
            await multi_user_db.initialize()

            # 2. Validate exchange configurations
            logger.info("Step 2: Validating exchange configurations...")
            await self._validate_exchange_configs()

            # 3. Initialize user service
            logger.info("Step 3: Starting user service...")
            await user_service.initialize()

            # 4. Initialize trading orchestrator
            logger.info("Step 4: Starting trading orchestrator...")
            await trading_orchestrator.initialize()

            # 5. Initialize monitoring service
            logger.info("Step 5: Starting monitoring service...")
            await monitoring_service.start()

            # 6. Initialize and start Telegram bot
            logger.info("Step 6: Starting Telegram bot...")
            bot = MultiUserTradingBot(self.config["telegram_bot_token"])
            bot_app = await bot.start()
            self.services["bot"] = bot
            self.services["bot_app"] = bot_app

            # Set the global instance for other services to use
            import services.multi_user_bot as bot_module

            bot_module.multi_user_bot = bot

            # 7. Start multi-exchange data service (initializes both MEXC and Bybit)
            logger.info("Step 7: Starting multi-exchange data service...")
            await multi_exchange_data_service.start()

            # 8. Start background tasks
            logger.info(")
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

            Step 8: Starting background tasks...")
            asyncio.create_task(self._periodic_maintenance())

            self.running = True
            logger.info("All services started successfully!")

            # Log startup metrics
            await multi_user_db.log_system_metric("service_restart", 1)
            await multi_user_db.log_system_metric(
                "startup_time", (datetime.now() - self.startup_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.shutdown_all_services()
            raise

    async def _validate_exchange_configs(self):
        """Validate exchange configurations at startup"""
        from exchange_factory import ExchangeFactory

        logger.info("Validating exchange configurations...")

        # Get available exchanges
        available_exchanges = ExchangeFactory.get_available_exchanges()

        configured_count = 0
        for exchange_name, capabilities in available_exchanges.items():
            if capabilities.api_configured:
                logger.info(f"✅ {capabilities.name}: API credentials configured")
                configured_count += 1

                # Test connection
                try:
                    success, message = await ExchangeFactory.test_exchange_connection(
                        exchange_name
                    )
                    if success:
                        logger.info(
                            f"✅ {capabilities.name}: Connection test successful"
                        )
                    else:
                        logger.warning(
                            f"⚠️ {capabilities.name}: Connection test failed - {message}"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ {capabilities.name}: Connection test error - {e}"
                    )
            else:
                logger.info(f"⚠️ {capabilities.name}: {capabilities.error_message}")

        if configured_count == 0:
            raise Exception(
                "No exchanges are properly configured! Please check your API keys."
            )

        logger.info(
            f"Exchange validation completed: {configured_count} exchange(s) configured"
        )

    async def shutdown_all_services(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down services...")
        self.running = False

        try:
            # Stop services in reverse order of startup

            # Stop integrated signal service
            if "signal_service" in self.services:
                try:
                    await self.services["signal_service"].stop_monitoring()
                    logger.info("Integrated signal service stopped")
                except Exception as e:
                    logger.error(f"Error stopping signal service: {e}")

            
            # 1. Stop Telegram bot first (stops new user requests)
            if "bot" in self.services:
                logger.info("Stopping Telegram bot...")
                try:
                    await self.services["bot"].stop()
                except Exception as e:
                    logger.error(f"Error stopping bot: {e}")

            # 2. Stop multi-exchange data service (stops new signals)
            logger.info("Stopping multi-exchange data service...")
            try:
                await multi_exchange_data_service.stop()
            except Exception as e:
                logger.error(f"Error stopping multi-exchange data service: {e}")

            # 3. Stop trading orchestrator (finishes pending trades)
            logger.info("Stopping trading orchestrator...")
            try:
                await trading_orchestrator.shutdown()
            except Exception as e:
                logger.error(f"Error stopping trading orchestrator: {e}")

            # 4. Stop monitoring service
            logger.info("Stopping monitoring service...")
            try:
                await monitoring_service.stop()
            except Exception as e:
                logger.error(f"Error stopping monitoring service: {e}")

            # 5. Log final shutdown metrics before closing database
            logger.info("Logging final metrics...")
            try:
                await multi_user_db.log_system_metric("service_shutdown", 1)
                # Wait a moment for final database operations to complete
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error logging shutdown metrics: {e}")

            # 6. Close database connections last (after all services are stopped)
            logger.info("Closing database connections...")
            try:
                await multi_user_db.shutdown()
            except Exception as e:
                logger.error(f"Error closing database: {e}")

            logger.info("All services stopped gracefully")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            # Even if there's an error, try to close database
            try:
                await multi_user_db.shutdown()
            except Exception as db_error:
                logger.error(f"Final database shutdown error: {db_error}")

    async def _force_signal_generation(self):
        """Force signal generation for testing purposes"""
        try:
            signals_generated = (
                await multi_exchange_data_service.force_signal_generation()
            )
            return {
                "status": "success",
                "message": f"Signals generated for all exchanges",
                "signals_generated": signals_generated,
            }
        except Exception as e:
            logger.error(f"Error forcing signal generation: {e}")
            return {"status": "error", "message": str(e)}

    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        logger.info("Starting periodic maintenance tasks...")

        while self.running:
            try:
                # Daily maintenance at midnight UTC
                now = datetime.now()
                if now.hour == 0 and now.minute < 5:
                    logger.info("Performing daily maintenance...")

                    # Reset monthly counters if needed
                    await user_service.reset_monthly_counters()

                    # Clean up old data
                    await self._cleanup_old_data()

                    # System health report
                    health = monitoring_service.get_health_status()
                    logger.info(f"Daily health report: {health['overall_status']}")

                # Wait 5 minutes before next check
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Error in maintenance task: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_data(self):
        """Clean up old data to maintain performance"""
        try:
            # Clean up old system metrics (keep 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)

            async with multi_user_db.get_connection() as db:
                await db.execute(
                    """
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                await db.commit()

            logger.info("Old data cleanup completed")

        except Exception as e:
            logger.error(f"Error in data cleanup: {e}")


async def main():
    """Main entry point"""
    service = None
    server = None

    try:
        # Create data and logs directories
        Path("./data").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

        # Initialize service
        service = ProductionTradingService()

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            if service:
                # Create shutdown task but don't block signal handler
                asyncio.create_task(service.shutdown_all_services())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start all services
        await service.start_all_services()

        # Start FastAPI server for monitoring
        config = uvicorn.Config(
            app=service.app, host="0.0.0.0", port=8080, log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info("Starting FastAPI monitoring server on port 8080...")

        # Run server and keep services running
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Ensure proper cleanup
        if service:
            try:
                await service.shutdown_all_services()
            except Exception as cleanup_error:
                logger.error(f"Error during final cleanup: {cleanup_error}")

        # Give the event loop a moment to finish cleanup
        try:
            await asyncio.sleep(0.1)
        except:
            pass

        logger.info("Main function completed")


if __name__ == "__main__":
    # Set up event loop policy for Windows compatibility
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the service
    asyncio.run(main())
