#!/usr/bin/env python3
"""
System Health Check Script
Monitors trading bot system health and provides diagnostics
"""

import asyncio
import logging
import sys
import os
import psutil
from datetime import datetime
import aiosqlite
import httpx
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive system health checker"""

    def __init__(self):
        self.results = {}
        self.errors = []

    async def check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage(".")
            disk_percent = (disk.used / disk.total) * 100

            self.results["system_resources"] = {
                "status": (
                    "healthy"
                    if cpu_percent < 80 and memory_percent < 90 and disk_percent < 85
                    else "warning"
                ),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "details": {
                    "cpu_warning": cpu_percent > 80,
                    "memory_warning": memory_percent > 90,
                    "disk_warning": disk_percent > 85,
                },
            }

        except Exception as e:
            self.results["system_resources"] = {"status": "error", "error": str(e)}
            self.errors.append(f"System resources check failed: {e}")

    async def check_database(self):
        """Check database connectivity and health"""
        try:
            db_path = "./data/trading_service.db"

            if not os.path.exists(db_path):
                self.results["database"] = {
                    "status": "error",
                    "error": "Database file not found",
                }
                return

            # Test database connection
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                )
                table_count = (await cursor.fetchone())[0]

                # Check if main tables exist
                cursor = await db.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('users', 'user_trades', 'system_metrics')
                """
                )
                existing_tables = [row[0] for row in await cursor.fetchall()]

                self.results["database"] = {
                    "status": "healthy" if len(existing_tables) >= 3 else "warning",
                    "table_count": table_count,
                    "required_tables": existing_tables,
                    "file_size_mb": os.path.getsize(db_path) / (1024 * 1024),
                }

        except Exception as e:
            self.results["database"] = {"status": "error", "error": str(e)}
            self.errors.append(f"Database check failed: {e}")

    async def check_telegram_bot(self):
        """Check Telegram bot token and connectivity"""
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

            if not bot_token:
                self.results["telegram_bot"] = {
                    "status": "error",
                    "error": "TELEGRAM_BOT_TOKEN environment variable not set",
                }
                return

            # Test bot API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10
                )

                if response.status_code == 200:
                    bot_info = response.json()
                    self.results["telegram_bot"] = {
                        "status": "healthy",
                        "bot_username": bot_info.get("result", {}).get("username"),
                        "bot_name": bot_info.get("result", {}).get("first_name"),
                    }
                else:
                    self.results["telegram_bot"] = {
                        "status": "error",
                        "error": f"API returned {response.status_code}: {response.text}",
                    }

        except Exception as e:
            self.results["telegram_bot"] = {"status": "error", "error": str(e)}
            self.errors.append(f"Telegram bot check failed: {e}")

    async def check_running_processes(self):
        """Check for running trading bot processes"""
        try:
            trading_processes = []

            for proc in psutil.process_iter(["pid", "name", "cmdline", "status"]):
                try:
                    if proc.info["name"] and "python" in proc.info["name"].lower():
                        cmdline = proc.info["cmdline"]
                        if cmdline and any(
                            "production_main.py" in str(arg)
                            or "trading" in str(arg).lower()
                            for arg in cmdline
                        ):
                            trading_processes.append(
                                {
                                    "pid": proc.info["pid"],
                                    "status": proc.info["status"],
                                    "cmdline": " ".join(cmdline) if cmdline else "N/A",
                                }
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            self.results["running_processes"] = {
                "status": "warning" if len(trading_processes) > 1 else "healthy",
                "process_count": len(trading_processes),
                "processes": trading_processes,
            }

        except Exception as e:
            self.results["running_processes"] = {"status": "error", "error": str(e)}
            self.errors.append(f"Process check failed: {e}")

    async def check_log_files(self):
        """Check log files for recent errors"""
        try:
            log_files = ["logs/trading_service.log", "debug_bot.log"]

            recent_errors = []
            file_statuses = {}

            for log_file in log_files:
                if os.path.exists(log_file):
                    file_size = os.path.getsize(log_file)
                    modified_time = datetime.fromtimestamp(os.path.getmtime(log_file))

                    file_statuses[log_file] = {
                        "exists": True,
                        "size_mb": file_size / (1024 * 1024),
                        "last_modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Check for recent errors (last 100 lines)
                    try:
                        with open(
                            log_file, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            lines = f.readlines()
                            recent_lines = lines[-100:] if len(lines) > 100 else lines

                            for line in recent_lines:
                                if "ERROR" in line or "CRITICAL" in line:
                                    recent_errors.append(
                                        {
                                            "file": log_file,
                                            "line": line.strip()[
                                                :200
                                            ],  # Truncate long lines
                                        }
                                    )
                    except Exception as read_error:
                        file_statuses[log_file]["read_error"] = str(read_error)
                else:
                    file_statuses[log_file] = {"exists": False}

            self.results["log_files"] = {
                "status": "warning" if recent_errors else "healthy",
                "files": file_statuses,
                "recent_errors_count": len(recent_errors),
                "recent_errors": (
                    recent_errors[-10:] if recent_errors else []
                ),  # Last 10 errors
            }

        except Exception as e:
            self.results["log_files"] = {"status": "error", "error": str(e)}
            self.errors.append(f"Log file check failed: {e}")

    async def check_api_endpoints(self):
        """Check if monitoring API is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8080/health", timeout=5)

                if response.status_code == 200:
                    health_data = response.json()
                    self.results["api_endpoints"] = {
                        "status": "healthy",
                        "monitoring_api": "running",
                        "health_data": health_data,
                    }
                else:
                    self.results["api_endpoints"] = {
                        "status": "warning",
                        "monitoring_api": "error",
                        "error": f"API returned {response.status_code}",
                    }

        except Exception as e:
            self.results["api_endpoints"] = {
                "status": "warning",
                "monitoring_api": "not_running",
                "error": str(e),
            }

    async def run_all_checks(self):
        """Run all health checks"""
        print("🔍 Running comprehensive system health check...")
        print("=" * 60)

        checks = [
            ("System Resources", self.check_system_resources),
            ("Database", self.check_database),
            ("Telegram Bot", self.check_telegram_bot),
            ("Running Processes", self.check_running_processes),
            ("Log Files", self.check_log_files),
            ("API Endpoints", self.check_api_endpoints),
        ]

        for check_name, check_func in checks:
            print(f"Checking {check_name}...", end=" ")
            await check_func()
            status = self.results.get(check_name.lower().replace(" ", "_"), {}).get(
                "status", "unknown"
            )

            if status == "healthy":
                print("✅ HEALTHY")
            elif status == "warning":
                print("⚠️ WARNING")
            elif status == "error":
                print("❌ ERROR")
            else:
                print("❓ UNKNOWN")

    def print_detailed_report(self):
        """Print detailed health report"""
        print("\n📊 DETAILED HEALTH REPORT")
        print("=" * 60)

        overall_status = "HEALTHY"

        for check_name, result in self.results.items():
            status = result.get("status", "unknown")
            print(f"\n🔸 {check_name.replace('_', ' ').title()}: {status.upper()}")

            if status == "error":
                overall_status = "ERROR"
            elif status == "warning" and overall_status != "ERROR":
                overall_status = "WARNING"

            # Print specific details
            if check_name == "system_resources" and status in ["healthy", "warning"]:
                print(f"   CPU: {result['cpu_percent']:.1f}%")
                print(f"   Memory: {result['memory_percent']:.1f}%")
                print(f"   Disk: {result['disk_percent']:.1f}%")

            elif check_name == "database" and status in ["healthy", "warning"]:
                print(f"   Tables: {result['table_count']}")
                print(f"   Size: {result['file_size_mb']:.2f} MB")

            elif check_name == "telegram_bot" and status == "healthy":
                print(f"   Bot: @{result.get('bot_username', 'unknown')}")

            elif check_name == "running_processes":
                print(f"   Active processes: {result['process_count']}")

            elif check_name == "log_files":
                error_count = result.get("recent_errors_count", 0)
                if error_count > 0:
                    print(f"   Recent errors: {error_count}")

            # Print errors
            if "error" in result:
                print(f"   ❌ {result['error']}")

        print(f"\n🎯 OVERALL SYSTEM STATUS: {overall_status}")

        if self.errors:
            print(f"\n⚠️ ISSUES FOUND ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")

        if self.results.get("system_resources", {}).get("status") == "warning":
            details = self.results["system_resources"].get("details", {})
            if details.get("cpu_warning"):
                print(
                    "   • High CPU usage detected - consider reducing concurrent operations"
                )
            if details.get("memory_warning"):
                print("   • High memory usage - restart system or increase memory")
            if details.get("disk_warning"):
                print("   • Low disk space - clean up old logs and data")

        if self.results.get("running_processes", {}).get("process_count", 0) > 1:
            print(
                "   • Multiple trading processes detected - stop duplicates to prevent conflicts"
            )

        if self.results.get("log_files", {}).get("recent_errors_count", 0) > 0:
            print("   • Recent errors found in logs - check log files for details")

        if self.results.get("telegram_bot", {}).get("status") == "error":
            print(
                "   • Telegram bot token issue - check TELEGRAM_BOT_TOKEN environment variable"
            )


async def main():
    """Main health check entry point"""
    checker = HealthChecker()

    try:
        await checker.run_all_checks()
        checker.print_detailed_report()

        # Exit with appropriate code
        if any(result.get("status") == "error" for result in checker.results.values()):
            sys.exit(1)  # Error found
        elif any(
            result.get("status") == "warning" for result in checker.results.values()
        ):
            sys.exit(2)  # Warnings found
        else:
            sys.exit(0)  # All healthy

    except KeyboardInterrupt:
        print("\n\n⏸️ Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
