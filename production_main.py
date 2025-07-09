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
from services.trading_orchestrator import trading_orchestrator
from services.monitoring_service import monitoring_service
from services.real_market_data_service import real_market_data_service
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
                    content={"status": "unhealthy", "message": "Service not running"},
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
                prices = real_market_data_service.get_current_prices()
                stats = real_market_data_service.get_service_stats()
                return JSONResponse(
                    content={
                        "prices": prices,
                        "service_status": {
                            "running": stats["running"],
                            "trading_pairs": stats["trading_pairs"],
                            "data_updates": stats["data_updates"],
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

    async def start_all_services(self):
        """Start all services in the correct order"""
        logger.info("Starting Multi-User Trading Service...")
        self.startup_time = datetime.now()

        try:
            # 1. Initialize database first
            logger.info("Step 1: Initializing database...")
            await multi_user_db.initialize()

            # 2. Initialize user service
            logger.info("Step 2: Starting user service...")
            await user_service.initialize()

            # 3. Initialize trading orchestrator
            logger.info("Step 3: Starting trading orchestrator...")
            await trading_orchestrator.initialize()

            # 4. Initialize monitoring service
            logger.info("Step 4: Starting monitoring service...")
            await monitoring_service.start()

            # 5. Initialize and start Telegram bot
            logger.info("Step 5: Starting Telegram bot...")
            bot = MultiUserTradingBot(self.config["telegram_bot_token"])
            bot_app = await bot.start()
            self.services["bot"] = bot
            self.services["bot_app"] = bot_app

            # Set the global instance for other services to use
            import services.multi_user_bot as bot_module

            bot_module.multi_user_bot = bot

            # 6. Start real market data service
            logger.info("Step 6: Starting real market data service...")
            await real_market_data_service.start()

            # 7. Start background tasks
            logger.info("Step 7: Starting background tasks...")
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

    async def shutdown_all_services(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down services...")
        self.running = False

        try:
            # Stop services in reverse order of startup

            # 1. Stop Telegram bot first (stops new user requests)
            if "bot" in self.services:
                logger.info("Stopping Telegram bot...")
                try:
                    await self.services["bot"].stop()
                except Exception as e:
                    logger.error(f"Error stopping bot: {e}")

            # 2. Stop real market data service (stops new signals)
            logger.info("Stopping real market data service...")
            try:
                await real_market_data_service.stop()
            except Exception as e:
                logger.error(f"Error stopping market data service: {e}")

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
            await real_market_data_service.force_signal_generation()
            return {"status": "success", "message": "Signals generated for all pairs"}
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
