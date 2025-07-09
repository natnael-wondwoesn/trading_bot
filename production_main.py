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
from db.multi_user_db import multi_user_db

# Configure logging
log_dir = os.getenv("LOG_DIR", "./logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "trading_service.log")),
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

            # 6. Start background tasks
            logger.info("Step 6: Starting background tasks...")
            asyncio.create_task(self._market_data_simulation())
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
            # Stop services in reverse order
            if "bot" in self.services:
                logger.info("Stopping Telegram bot...")
                await self.services["bot"].stop()

            logger.info("Stopping monitoring service...")
            await monitoring_service.stop()

            logger.info("Stopping trading orchestrator...")
            await trading_orchestrator.shutdown()

            # Log shutdown
            await multi_user_db.log_system_metric("service_shutdown", 1)

            logger.info("All services stopped gracefully")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def _market_data_simulation(self):
        """Simulate market data for development/testing"""
        # This would be replaced with real market data feeds in production
        logger.info("Starting market data simulation...")

        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"]

        while self.running:
            try:
                for symbol in symbols:
                    # Simulate market data
                    market_data = {
                        "symbol": symbol,
                        "price": 50000.0 + (hash(symbol) % 10000),  # Fake price
                        "volume": 1000000.0,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Process signals for all users
                    await trading_orchestrator.process_market_signal(
                        symbol, market_data
                    )

                # Wait before next cycle
                await asyncio.sleep(60)  # Process every minute

            except Exception as e:
                logger.error(f"Error in market data simulation: {e}")
                await asyncio.sleep(30)

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

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            asyncio.create_task(self.shutdown_all_services())
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    try:
        # Create data and logs directories
        Path("./data").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

        # Initialize service
        service = ProductionTradingService()

        # Setup signal handlers
        service.setup_signal_handlers()

        # Start all services
        await service.start_all_services()

        # Start FastAPI server for monitoring
        config = uvicorn.Config(
            app=service.app, host="0.0.0.0", port=8080, log_level="info"
        )
        server = uvicorn.Server(config)

        # Run server and keep services running
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if "service" in locals():
            await service.shutdown_all_services()


if __name__ == "__main__":
    # Set up event loop policy for Windows compatibility
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the service
    asyncio.run(main())
