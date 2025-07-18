#!/usr/bin/env python3
"""
Production Main Entry Point - Minimal Working Version
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

# Import services
from services.user_service import user_service
from services.multi_user_bot import MultiUserTradingBot
from services.trading_orchestrator import trading_orchestrator
from services.monitoring_service import monitoring_service
from services.multi_exchange_data_service import multi_exchange_data_service
from services.integrated_signal_service import integrated_signal_service
from db.multi_user_db import multi_user_db

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

        self.config = {
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "webhook_url": os.getenv("WEBHOOK_URL"),
            "max_concurrent_users": int(os.getenv("MAX_CONCURRENT_USERS", 1000)),
            "database_url": os.getenv("DATABASE_URL", "./data/trading_service.db"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

        if not self.config["telegram_bot_token"]:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        # Setup FastAPI
        self.app = FastAPI(
            title="Trading Service API",
            description="Multi-User Trading Service",
            version="1.0.0",
        )

        self.setup_routes()

    def setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse(
                content={
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "uptime": (
                        str(datetime.now() - self.startup_time)
                        if self.startup_time
                        else "0"
                    ),
                    "services": {
                        "database": "online",
                        "bot": "online" if "bot" in self.services else "offline",
                        "signal_service": (
                            "online" if "signal_service" in self.services else "offline"
                        ),
                    },
                }
            )

        @self.app.get("/signals/status")
        async def get_signal_status():
            """Get signal service status"""
            try:
                if "signal_service" in self.services:
                    service = self.services["signal_service"]
                    return JSONResponse(
                        content={
                            "running": getattr(service, "running", False),
                            "strategy": (
                                service.strategy.name if service.strategy else None
                            ),
                            "trading_pairs": len(getattr(service, "trading_pairs", [])),
                            "callbacks": len(getattr(service, "signal_callbacks", [])),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                else:
                    return JSONResponse(
                        content={
                            "running": False,
                            "error": "Signal service not initialized",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @self.app.post("/signals/force-scan")
        async def force_signal_scan():
            """Force a signal scan for testing"""
            try:
                if "signal_service" in self.services:
                    service = self.services["signal_service"]
                    if hasattr(service, "force_scan"):
                        signals = await service.force_scan()
                        return JSONResponse(
                            content={
                                "success": True,
                                "signals_found": len(signals),
                                "signals": signals,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                return JSONResponse(
                    content={"success": False, "error": "Signal service not available"}
                )
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

    async def start_all_services(self):
        """Start all services in the correct order"""
        logger.info("Starting Multi-User Trading Service...")
        self.startup_time = datetime.now()

        try:
            # 1. Initialize database
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

            # 6. Start multi-exchange data service
            logger.info("Step 6: Starting multi-exchange data service...")
            await multi_exchange_data_service.start()

            # 7. Initialize and start integrated signal service
            logger.info("Step 7: Starting integrated signal service...")
            try:
                await integrated_signal_service.initialize()

                # Add signal notification callback
                async def signal_notification_callback(signals):
                    if signals and "bot" in self.services:
                        try:
                            bot = self.services["bot"]
                            active_user_count = len(bot.active_users)

                            if active_user_count == 0:
                                logger.warning(
                                    "🚨 NO ACTIVE USERS - Signals detected but no users to send to!"
                                )
                                logger.info(
                                    "💡 Users need to start your bot with /start to receive signals"
                                )
                                logger.info(
                                    "📱 Share your bot link: t.me/your_bot_username"
                                )
                                return  # No point broadcasting to 0 users

                            logger.info(
                                f"Broadcasting {len(signals)} signals to {active_user_count} active users"
                            )

                            # Send each signal to all active users using the correct broadcast method
                            total_sent = 0
                            for signal in signals:
                                # Log original signal for debugging
                                logger.debug(
                                    f"Processing signal for broadcast: {signal}"
                                )

                                # Convert signal dict to format expected by broadcast method
                                original_confidence = signal.get("confidence", 0)
                                # Ensure confidence is in percentage (0-100)
                                if original_confidence <= 1:
                                    confidence_percent = original_confidence * 100
                                else:
                                    confidence_percent = original_confidence

                                signal_data = {
                                    "symbol": signal.get(
                                        "pair", signal.get("symbol", "UNKNOWN")
                                    ),
                                    "action": signal.get("action", "HOLD"),
                                    "price": signal.get("price", 0),
                                    "confidence": confidence_percent,
                                    "stop_loss": signal.get("stop_loss"),
                                    "take_profit": signal.get("take_profit"),
                                    "timestamp": signal.get(
                                        "timestamp", datetime.now()
                                    ),
                                }

                                logger.debug(f"Converted signal data: {signal_data}")

                                # Use the correct broadcast method
                                try:
                                    sent_count = (
                                        await bot.broadcast_signal_to_all_users(
                                            signal_data
                                        )
                                    )
                                    total_sent += sent_count
                                    logger.info(
                                        f"📊 {signal_data['symbol']} {signal_data['action']} signal sent to {sent_count} users"
                                    )
                                except Exception as broadcast_error:
                                    logger.error(
                                        f"Error broadcasting signal {signal_data['symbol']}: {broadcast_error}"
                                    )

                            logger.info(
                                f"🎯 Total broadcast complete: {total_sent} messages sent across {len(signals)} signals"
                            )

                        except Exception as e:
                            logger.error(f"Error broadcasting signals to users: {e}")

                integrated_signal_service.add_signal_callback(
                    signal_notification_callback
                )

                # Start monitoring in background
                asyncio.create_task(integrated_signal_service.start_monitoring())
                self.services["signal_service"] = integrated_signal_service

                logger.info("Integrated signal service started successfully")

            except Exception as e:
                logger.error(f"Failed to start integrated signal service: {e}")
                logger.warning("Continuing without integrated signal service...")

            # 8. Start background tasks
            logger.info("Step 8: Starting background tasks...")
            asyncio.create_task(self._periodic_maintenance())

            self.running = True
            logger.info("All services started successfully!")

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.shutdown_all_services()
            raise

    async def shutdown_all_services(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down services...")
        self.running = False

        try:
            # Stop signal service
            if "signal_service" in self.services:
                try:
                    await self.services["signal_service"].stop_monitoring()
                    logger.info("Integrated signal service stopped")
                except Exception as e:
                    logger.error(f"Error stopping signal service: {e}")

            # Stop other services
            if "bot" in self.services:
                try:
                    await self.services["bot"].stop()
                except:
                    pass

            await monitoring_service.stop()
            await multi_exchange_data_service.stop()
            await multi_user_db.shutdown()

            logger.info("All services stopped gracefully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if not self.running:
                    break
                # Add maintenance tasks here
            except Exception as e:
                logger.error(f"Error in maintenance task: {e}")
                await asyncio.sleep(300)


async def main():
    """Main entry point"""
    service = None

    try:
        # Create directories
        Path("./data").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

        # Initialize service
        service = ProductionTradingService()

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            if service:
                asyncio.create_task(service.shutdown_all_services())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start all services
        await service.start_all_services()

        # Start FastAPI server
        config = uvicorn.Config(
            app=service.app, host="0.0.0.0", port=8080, log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info("Starting FastAPI monitoring server on port 8080...")
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if service:
            try:
                await service.shutdown_all_services()
            except Exception as cleanup_error:
                logger.error(f"Error during final cleanup: {cleanup_error}")

        logger.info("Main function completed")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
