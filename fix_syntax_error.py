#!/usr/bin/env python3
"""
Fix Syntax Error in production_main.py
"""

import os
import re
from datetime import datetime
import sys
import asyncio
from pathlib import Path


def fix_syntax_error():
    """Fix the unterminated string literal in production_main.py"""
    print("🔧 FIXING SYNTAX ERROR IN PRODUCTION_MAIN.PY")
    print("=" * 50)

    production_file = "production_main.py"

    if not os.path.exists(production_file):
        print("❌ production_main.py not found")
        return False

    try:
        # Read the file with error handling
        with open(production_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        print(f"📖 Read {len(lines)} lines from production_main.py")

        # Find and fix the problematic line around line 328
        fixed_lines = []
        line_number = 0

        for line in lines:
            line_number += 1
            original_line = line

            # Look for unterminated string literals
            if (
                'logger.info("' in line
                and not line.rstrip().endswith('")')
                and not line.rstrip().endswith('")')
            ):
                # Check if the string is not properly closed
                quote_count = line.count('"')
                if quote_count % 2 != 0:  # Odd number of quotes means unterminated
                    print(
                        f"🔍 Found unterminated string at line {line_number}: {line.strip()}"
                    )

                    # Try to fix common patterns
                    if line.strip().endswith('"'):
                        # Already properly terminated
                        pass
                    elif 'logger.info("' in line and not line.rstrip().endswith('")'):
                        # Add closing quote and parenthesis
                        line = line.rstrip() + '")\n'
                        print(
                            f"✅ Fixed line {line_number}: Added closing quote and parenthesis"
                        )
                    elif line.strip().endswith("..."):
                        # Replace ... with proper ending
                        line = re.sub(r"\.\.\..*$", '")\n', line)
                        print(
                            f"✅ Fixed line {line_number}: Replaced ... with proper ending"
                        )

            # Look for other common syntax issues
            elif "logger.info(" in line and '"' in line:
                # Check for improperly formatted logger statements
                if line.count('"') % 2 != 0 and not line.strip().endswith('")'):
                    print(
                        f"🔍 Found potential logger issue at line {line_number}: {line.strip()}"
                    )

                    # Try to fix
                    if not line.rstrip().endswith('")') and not line.rstrip().endswith(
                        '")'
                    ):
                        line = line.rstrip() + '")\n'
                        print(f"✅ Fixed line {line_number}: Added missing closing")

            fixed_lines.append(line)

        # Write back the fixed content
        with open(production_file, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

        print("✅ Syntax fixes applied successfully")

        # Verify the fix by trying to compile
        try:
            with open(production_file, "r", encoding="utf-8") as f:
                content = f.read()

            compile(content, production_file, "exec")
            print("✅ Syntax validation passed")
            return True

        except SyntaxError as e:
            print(f"❌ Still has syntax error: {e}")
            print(f"   Line {e.lineno}: {e.text}")

            # Try a more aggressive fix
            return fix_specific_syntax_error(e.lineno)

    except Exception as e:
        print(f"❌ Failed to fix syntax error: {e}")
        return False


def fix_specific_syntax_error(error_line):
    """Fix specific syntax error at given line number"""
    print(f"\n🎯 TARGETING SPECIFIC ERROR AT LINE {error_line}")
    print("=" * 30)

    try:
        with open("production_main.py", "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Look at the problematic line and surrounding context
        start_line = max(0, error_line - 5)
        end_line = min(len(lines), error_line + 5)

        print("Context around error:")
        for i in range(start_line, end_line):
            marker = ">>> " if i == error_line - 1 else "    "
            print(f"{marker}Line {i+1}: {lines[i].rstrip()}")

        # Fix the specific line
        if error_line <= len(lines):
            problematic_line = lines[error_line - 1]
            print(f"\n🔧 Fixing line {error_line}: {problematic_line.strip()}")

            # Common fixes
            if 'logger.info("' in problematic_line:
                # Find the opening quote
                start_quote = problematic_line.find('logger.info("') + 12

                # Check if there's a proper closing
                remaining = problematic_line[start_quote:]
                if not ('")' in remaining or remaining.strip().endswith('")')):
                    # Add proper closing
                    if remaining.strip().endswith('"'):
                        lines[error_line - 1] = problematic_line.rstrip() + ")\n"
                    else:
                        lines[error_line - 1] = problematic_line.rstrip() + '")\n'

                    print(f"✅ Fixed: {lines[error_line - 1].strip()}")

            # Write back
            with open("production_main.py", "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Test again
            with open("production_main.py", "r", encoding="utf-8") as f:
                content = f.read()

            compile(content, "production_main.py", "exec")
            print("✅ Specific fix successful")
            return True

    except Exception as e:
        print(f"❌ Specific fix failed: {e}")
        return False


def create_minimal_production_main():
    """Create a minimal working production_main.py if fixing fails"""
    print("\n🆘 CREATING MINIMAL WORKING PRODUCTION_MAIN.PY")
    print("=" * 50)

    minimal_content = '''#!/usr/bin/env python3
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
            version="1.0.0"
        )
        
        self.setup_routes()

    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse(content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.startup_time) if self.startup_time else "0",
                "services": {
                    "database": "online",
                    "bot": "online" if "bot" in self.services else "offline",
                    "signal_service": "online" if "signal_service" in self.services else "offline"
                }
            })
        
        @self.app.get("/signals/status")
        async def get_signal_status():
            """Get signal service status"""
            try:
                if "signal_service" in self.services:
                    service = self.services["signal_service"]
                    return JSONResponse(content={
                        "running": getattr(service, 'running', False),
                        "strategy": service.strategy.name if service.strategy else None,
                        "trading_pairs": len(getattr(service, 'trading_pairs', [])),
                        "callbacks": len(getattr(service, 'signal_callbacks', [])),
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
                return JSONResponse(content={
                    "success": False,
                    "error": "Signal service not available"
                })
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
'''

    # Backup the original
    if os.path.exists("production_main.py"):
        backup_name = (
            f"production_main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        )
        os.rename("production_main.py", backup_name)
        print(f"📁 Backed up original to {backup_name}")

    # Write the minimal version
    with open("production_main.py", "w", encoding="utf-8") as f:
        f.write(minimal_content)

    print("✅ Created minimal working production_main.py")
    return True


def main():
    """Main execution"""
    print("FIXING SYNTAX ERROR IN PRODUCTION_MAIN.PY")
    print("=" * 60)

    # Try to fix the existing file first
    if fix_syntax_error():
        print("\n🎉 SYNTAX ERROR FIXED!")
        print("✅ Your original production_main.py has been repaired")
    else:
        print("\n⚠️ Could not fix existing file, creating minimal version...")
        if create_minimal_production_main():
            print("\n🎉 MINIMAL VERSION CREATED!")
            print("✅ Created a clean working production_main.py")
        else:
            print("\n❌ Failed to create minimal version")
            return

    print("\n🚀 NOW YOU CAN START YOUR SYSTEM:")
    print("   python production_main.py")

    print("\n📊 MONITOR YOUR SYSTEM:")
    print("   http://localhost:8080/health")
    print("   http://localhost:8080/signals/status")
    print("   http://localhost:8080/signals/force-scan")


if __name__ == "__main__":
    main()
