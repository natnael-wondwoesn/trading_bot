#!/usr/bin/env python3
"""
Render Deployment Entrypoint
Optimized for 24/7 hosting on Render platform
"""

import asyncio
import logging
import os
import sys
import signal
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Import our services
from services.user_service import user_service
from services.multi_user_bot import MultiUserTradingBot
from services.trading_orchestrator import trading_orchestrator
from services.monitoring_service import monitoring_service
from db.multi_user_db import multi_user_db

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Render captures stdout for logs
)

logger = logging.getLogger(__name__)


class RenderTradingService:
    """Render-optimized trading service"""

    def __init__(self):
        # Get port from Render environment
        self.port = int(os.getenv("PORT", 8080))
        self.host = "0.0.0.0"  # Required for Render

        # Configuration optimized for Render
        self.config = {
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "database_url": os.getenv(
                "DATABASE_URL", "/tmp/trading_data/trading_service.db"
            ),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            "max_concurrent_users": int(os.getenv("MAX_CONCURRENT_USERS", 1000)),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "environment": os.getenv("ENVIRONMENT", "production"),
            "sentry_dsn": os.getenv("SENTRY_DSN"),
        }

        # Validate critical configuration
        if not self.config["telegram_bot_token"]:
            raise ValueError("TELEGRAM_BOT_TOKEN is required for Render deployment")

        # Initialize FastAPI with Render-specific configuration
        self.app = FastAPI(
            title="24/7 Trading Bot Service",
            description="Multi-User Trading Bot hosted on Render",
            version="2.0.0",
            docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
            redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
        )

        # Initialize Sentry for error tracking if configured
        if self.config["sentry_dsn"]:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.asyncio import AsyncioIntegration

            sentry_sdk.init(
                dsn=self.config["sentry_dsn"],
                integrations=[
                    FastApiIntegration(auto_enabling_integrations=False),
                    AsyncioIntegration(),
                ],
                traces_sample_rate=0.1,
                environment=self.config["environment"],
            )

        self.services = {}
        self.running = False
        self.startup_time = None

        self._setup_routes()
        self._setup_shutdown_handlers()

    def _setup_routes(self):
        """Setup FastAPI routes optimized for Render"""

        @self.app.get("/")
        async def root():
            """Root endpoint showing service status"""
            return {
                "service": "24/7 Trading Bot",
                "status": "running" if self.running else "starting",
                "uptime_seconds": (
                    (datetime.now() - self.startup_time).total_seconds()
                    if self.startup_time
                    else 0
                ),
                "environment": self.config["environment"],
                "port": self.port,
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for Render"""
            if not self.running:
                return JSONResponse(
                    status_code=503,
                    content={"status": "starting", "message": "Service is starting up"},
                )

            try:
                # Use comprehensive health checker
                from health_check_render import health_checker

                health_status = await health_checker.comprehensive_health_check()

                # Return appropriate status code based on health
                if health_status["status"] == "healthy":
                    return JSONResponse(content=health_status)
                elif health_status["status"] == "degraded":
                    return JSONResponse(
                        status_code=200, content=health_status
                    )  # Still serving traffic
                else:  # unhealthy
                    return JSONResponse(status_code=503, content=health_status)

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    status_code=500, content={"status": "error", "message": str(e)}
                )

        @self.app.get("/metrics")
        async def get_metrics():
            """System metrics endpoint"""
            try:
                return {
                    "active_users": (
                        await multi_user_db.get_active_users_count()
                        if self.running
                        else 0
                    ),
                    "memory_usage": self._get_memory_usage(),
                    "uptime_seconds": (
                        (datetime.now() - self.startup_time).total_seconds()
                        if self.startup_time
                        else 0
                    ),
                    "environment": self.config["environment"],
                    "port": self.port,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                return {"error": str(e)}

        @self.app.get("/status")
        async def get_detailed_status():
            """Detailed status for monitoring"""
            if not self.running:
                return {"status": "starting"}

            try:
                return {
                    "status": "running",
                    "services": {
                        "bot": "running" if "bot" in self.services else "stopped",
                        "orchestrator": (
                            "running" if trading_orchestrator.running else "stopped"
                        ),
                        "monitoring": (
                            "running" if monitoring_service.running else "stopped"
                        ),
                    },
                    "configuration": {
                        "max_users": self.config["max_concurrent_users"],
                        "environment": self.config["environment"],
                        "log_level": self.config["log_level"],
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

    async def _check_database_health(self):
        """Check database connectivity"""
        try:
            await multi_user_db.get_active_users_count()
            return True
        except:
            return False

    def _check_bot_health(self):
        """Check bot service health"""
        return "bot" in self.services and hasattr(self.services["bot"], "running")

    def _check_orchestrator_health(self):
        """Check orchestrator health"""
        return hasattr(trading_orchestrator, "running") and trading_orchestrator.running

    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
            }
        except:
            return {"error": "Unable to get memory info"}

    def _setup_shutdown_handlers(self):
        """Setup graceful shutdown handlers"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def start_services(self):
        """Start all trading services"""
        logger.info("Starting 24/7 Trading Service on Render...")
        self.startup_time = datetime.now()

        try:
            # 1. Initialize database with Render-compatible path
            logger.info("Initializing database...")
            await multi_user_db.initialize()

            # 2. Start user service
            logger.info("Starting user service...")
            await user_service.initialize()

            # 3. Initialize and start multi-user bot
            logger.info("Starting multi-user bot...")
            bot = MultiUserTradingBot(
                token=self.config["telegram_bot_token"],
                max_concurrent_users=self.config["max_concurrent_users"],
            )
            await bot.initialize()
            self.services["bot"] = bot

            # 4. Start trading orchestrator
            logger.info("Starting trading orchestrator...")
            await trading_orchestrator.initialize()
            await trading_orchestrator.start()

            # 5. Start monitoring service
            logger.info("Starting monitoring service...")
            await monitoring_service.initialize()
            await monitoring_service.start()

            self.running = True
            logger.info(
                f"✅ All services started successfully! Running on port {self.port}"
            )

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.shutdown()
            raise

    async def shutdown(self):
        """Graceful shutdown of all services"""
        logger.info("Shutting down services...")
        self.running = False

        try:
            # Stop services in reverse order
            if "bot" in self.services:
                await self.services["bot"].stop()

            await monitoring_service.stop()
            await trading_orchestrator.stop()

            logger.info("All services stopped gracefully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def run(self):
        """Main run method for Render deployment"""
        try:
            # Start all services first
            await self.start_services()

            # Configure uvicorn for Render
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True,
                use_colors=False,  # Better for container logs
                loop="asyncio",
            )

            server = uvicorn.Server(config)

            logger.info(f"🚀 Starting server on {self.host}:{self.port}")
            await server.serve()

        except Exception as e:
            logger.error(f"Server error: {e}")
            await self.shutdown()
            raise


async def main():
    """Main entry point for Render"""
    try:
        service = RenderTradingService()
        await service.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
