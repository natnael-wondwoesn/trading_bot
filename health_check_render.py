#!/usr/bin/env python3
"""
Health Check Module for Render Deployment
Provides comprehensive health monitoring for 24/7 operation
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any

import aiohttp
import psutil
from db.multi_user_db import multi_user_db

logger = logging.getLogger(__name__)


class RenderHealthChecker:
    """Comprehensive health checker for Render deployment"""

    def __init__(self):
        self.startup_time = datetime.now()
        self.last_health_check = None
        self.health_history = []
        self.max_history = 100

    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_result = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Core system checks
        health_result["checks"]["system"] = await self._check_system_health()
        health_result["checks"]["database"] = await self._check_database_health()
        health_result["checks"]["memory"] = self._check_memory_health()
        health_result["checks"]["disk"] = self._check_disk_health()
        health_result["checks"]["network"] = await self._check_network_health()

        # Service-specific checks
        health_result["checks"]["services"] = await self._check_services_health()

        # Determine overall health status
        failed_checks = [
            name
            for name, result in health_result["checks"].items()
            if not result.get("healthy", False)
        ]

        if failed_checks:
            health_result["status"] = (
                "degraded" if len(failed_checks) <= 2 else "unhealthy"
            )
            health_result["errors"].extend(failed_checks)

        # Store in history
        self._store_health_result(health_result)
        self.last_health_check = datetime.now()

        return health_result

    async def _check_system_health(self) -> Dict[str, Any]:
        """Check basic system health"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Load average (Unix-like systems)
            load_avg = None
            if hasattr(os, "getloadavg"):
                load_avg = os.getloadavg()

            # Process information
            process = psutil.Process()
            process_info = {
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": process.create_time(),
                "num_threads": process.num_threads(),
            }

            system_healthy = cpu_percent < 90 and process_info["memory_percent"] < 85

            return {
                "healthy": system_healthy,
                "cpu_percent": cpu_percent,
                "load_average": load_avg,
                "process": process_info,
                "warnings": ["High CPU usage"] if cpu_percent > 80 else [],
            }

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            start_time = time.time()

            # Test basic connectivity
            user_count = await multi_user_db.get_active_users_count()

            # Test write operation (if possible)
            response_time = time.time() - start_time

            db_healthy = response_time < 5.0  # Should respond within 5 seconds

            return {
                "healthy": db_healthy,
                "response_time_ms": round(response_time * 1000, 2),
                "active_users": user_count,
                "database_url": os.getenv("DATABASE_URL", "Not configured"),
                "warnings": ["Slow database response"] if response_time > 2.0 else [],
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage and availability"""
        try:
            # System memory
            memory = psutil.virtual_memory()

            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()

            # Memory health thresholds
            system_memory_healthy = memory.percent < 85
            process_memory_mb = process_memory.rss / 1024 / 1024
            process_memory_healthy = (
                process_memory_mb < 400
            )  # Adjust based on your needs

            memory_healthy = system_memory_healthy and process_memory_healthy

            warnings = []
            if memory.percent > 80:
                warnings.append("High system memory usage")
            if process_memory_mb > 300:
                warnings.append("High process memory usage")

            return {
                "healthy": memory_healthy,
                "system": {
                    "total_mb": round(memory.total / 1024 / 1024, 2),
                    "available_mb": round(memory.available / 1024 / 1024, 2),
                    "percent_used": memory.percent,
                },
                "process": {
                    "rss_mb": round(process_memory_mb, 2),
                    "vms_mb": round(process_memory.vms / 1024 / 1024, 2),
                },
                "warnings": warnings,
            }

        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk usage and availability"""
        try:
            # Check main disk usage
            disk_usage = psutil.disk_usage("/")

            # Check trading data directory if it exists
            trading_data_path = "/tmp/trading_data"
            trading_data_usage = None
            if os.path.exists(trading_data_path):
                trading_data_usage = psutil.disk_usage(trading_data_path)

            disk_healthy = disk_usage.percent < 90

            warnings = []
            if disk_usage.percent > 80:
                warnings.append("High disk usage")

            result = {
                "healthy": disk_healthy,
                "root": {
                    "total_gb": round(disk_usage.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk_usage.used / 1024 / 1024 / 1024, 2),
                    "free_gb": round(disk_usage.free / 1024 / 1024 / 1024, 2),
                    "percent_used": disk_usage.percent,
                },
                "warnings": warnings,
            }

            if trading_data_usage:
                result["trading_data"] = {
                    "total_gb": round(trading_data_usage.total / 1024 / 1024 / 1024, 2),
                    "free_gb": round(trading_data_usage.free / 1024 / 1024 / 1024, 2),
                    "percent_used": trading_data_usage.percent,
                }

            return result

        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Test external connectivity
            test_urls = [
                "https://httpbin.org/status/200",  # Simple test endpoint
                "https://api.telegram.org/bot",  # Telegram API
            ]

            network_results = []

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                for url in test_urls:
                    try:
                        start_time = time.time()
                        async with session.get(url) as response:
                            response_time = time.time() - start_time
                            network_results.append(
                                {
                                    "url": url,
                                    "status": response.status,
                                    "response_time_ms": round(response_time * 1000, 2),
                                    "success": response.status < 400,
                                }
                            )
                    except Exception as e:
                        network_results.append(
                            {"url": url, "error": str(e), "success": False}
                        )

            successful_tests = sum(
                1 for result in network_results if result.get("success", False)
            )
            network_healthy = (
                successful_tests >= len(test_urls) // 2
            )  # At least half should succeed

            return {
                "healthy": network_healthy,
                "tests": network_results,
                "successful_tests": successful_tests,
                "warnings": (
                    ["Network connectivity issues"] if not network_healthy else []
                ),
            }

        except Exception as e:
            logger.error(f"Network health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def _check_services_health(self) -> Dict[str, Any]:
        """Check application-specific services"""
        try:
            services_status = {
                "telegram_bot": self._check_telegram_config(),
                "exchange_apis": self._check_exchange_config(),
                "environment": self._check_environment_config(),
            }

            # Count healthy services
            healthy_services = sum(
                1 for status in services_status.values() if status.get("healthy", False)
            )
            total_services = len(services_status)

            services_healthy = healthy_services >= total_services // 2

            return {
                "healthy": services_healthy,
                "services": services_status,
                "healthy_count": healthy_services,
                "total_count": total_services,
                "warnings": (
                    ["Some services not configured"] if not services_healthy else []
                ),
            }

        except Exception as e:
            logger.error(f"Services health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    def _check_telegram_config(self) -> Dict[str, Any]:
        """Check Telegram bot configuration"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        return {
            "healthy": bool(token),
            "configured": bool(token),
            "token_length": len(token) if token else 0,
        }

    def _check_exchange_config(self) -> Dict[str, Any]:
        """Check exchange API configuration"""
        mexc_key = os.getenv("MEXC_API_KEY")
        mexc_secret = os.getenv("MEXC_API_SECRET")
        bybit_key = os.getenv("BYBIT_API_KEY")
        bybit_secret = os.getenv("BYBIT_API_SECRET")

        mexc_configured = bool(mexc_key and mexc_secret)
        bybit_configured = bool(bybit_key and bybit_secret)

        return {
            "healthy": mexc_configured or bybit_configured,
            "mexc_configured": mexc_configured,
            "bybit_configured": bybit_configured,
            "total_exchanges": sum([mexc_configured, bybit_configured]),
        }

    def _check_environment_config(self) -> Dict[str, Any]:
        """Check environment configuration"""
        required_vars = ["TELEGRAM_BOT_TOKEN"]
        optional_vars = ["REDIS_URL", "SENTRY_DSN", "DATABASE_URL"]

        required_configured = all(os.getenv(var) for var in required_vars)
        optional_configured = sum(1 for var in optional_vars if os.getenv(var))

        return {
            "healthy": required_configured,
            "required_configured": required_configured,
            "optional_configured": optional_configured,
            "total_optional": len(optional_vars),
            "environment": os.getenv("ENVIRONMENT", "development"),
        }

    def _store_health_result(self, result: Dict[str, Any]):
        """Store health check result in history"""
        self.health_history.append(
            {
                "timestamp": result["timestamp"],
                "status": result["status"],
                "uptime": result["uptime_seconds"],
            }
        )

        # Keep only recent history
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history :]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary and trends"""
        if not self.health_history:
            return {"status": "no_data"}

        recent_checks = self.health_history[-10:]  # Last 10 checks
        healthy_count = sum(
            1 for check in recent_checks if check["status"] == "healthy"
        )

        return {
            "current_status": (
                self.health_history[-1]["status"] if self.health_history else "unknown"
            ),
            "last_check": (
                self.last_health_check.isoformat() if self.last_health_check else None
            ),
            "recent_health_rate": healthy_count / len(recent_checks) * 100,
            "total_checks": len(self.health_history),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
        }


# Global health checker instance
health_checker = RenderHealthChecker()
