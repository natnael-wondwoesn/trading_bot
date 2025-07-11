#!/usr/bin/env python3
"""
External Services Configuration for Render Deployment
Handles Redis, Database, and other external service configurations
"""

import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ExternalServicesConfig:
    """Configuration manager for external services"""

    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.redis_config = self._configure_redis()
        self.database_config = self._configure_database()
        self.monitoring_config = self._configure_monitoring()

    def _configure_redis(self) -> Dict[str, Any]:
        """Configure Redis connection"""
        redis_url = os.getenv("REDIS_URL")

        if redis_url:
            # Parse Redis URL
            parsed = urlparse(redis_url)
            config = {
                "enabled": True,
                "url": redis_url,
                "host": parsed.hostname,
                "port": parsed.port or 6379,
                "password": parsed.password,
                "db": int(parsed.path.lstrip("/")) if parsed.path else 0,
                "ssl": parsed.scheme == "rediss",
                "connection_pool": {
                    "max_connections": 20,
                    "retry_on_timeout": True,
                    "health_check_interval": 30,
                },
            }
        else:
            # Fallback to in-memory cache for development
            config = {
                "enabled": False,
                "fallback": "memory",
                "warning": "Redis not configured, using in-memory fallback",
            }

        logger.info(
            f"Redis configuration: {'Enabled' if config['enabled'] else 'Disabled (fallback)'}"
        )
        return config

    def _configure_database(self) -> Dict[str, Any]:
        """Configure database connection"""
        database_url = os.getenv("DATABASE_URL", "/tmp/trading_data/trading_service.db")

        if database_url.startswith("postgresql://") or database_url.startswith(
            "postgres://"
        ):
            # PostgreSQL configuration for production
            config = {
                "type": "postgresql",
                "url": database_url,
                "pool_size": int(os.getenv("DB_POOL_SIZE", 20)),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 10)),
                "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", 30)),
                "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", 3600)),
                "ssl_required": os.getenv("DB_SSL_REQUIRED", "true").lower() == "true",
            }
        else:
            # SQLite configuration for development/testing
            # Ensure directory exists
            db_dir = os.path.dirname(database_url)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            config = {
                "type": "sqlite",
                "url": database_url,
                "path": database_url,
                "wal_mode": True,  # Better for concurrent access
                "synchronous": "NORMAL",  # Better performance
                "journal_mode": "WAL",
                "cache_size": 10000,
                "temp_store": "memory",
            }

        logger.info(f"Database configuration: {config['type']} at {database_url}")
        return config

    def _configure_monitoring(self) -> Dict[str, Any]:
        """Configure monitoring and alerting services"""
        config = {
            "sentry": {
                "enabled": bool(os.getenv("SENTRY_DSN")),
                "dsn": os.getenv("SENTRY_DSN"),
                "environment": self.environment,
                "sample_rate": float(os.getenv("SENTRY_SAMPLE_RATE", 0.1)),
                "debug": self.environment != "production",
            },
            "webhook": {
                "enabled": bool(os.getenv("WEBHOOK_URL")),
                "url": os.getenv("WEBHOOK_URL"),
                "timeout": int(os.getenv("WEBHOOK_TIMEOUT", 10)),
                "retry_attempts": int(os.getenv("WEBHOOK_RETRY_ATTEMPTS", 3)),
            },
            "health_checks": {
                "interval_seconds": int(os.getenv("HEALTH_CHECK_INTERVAL", 60)),
                "timeout_seconds": int(os.getenv("HEALTH_CHECK_TIMEOUT", 10)),
                "failure_threshold": int(
                    os.getenv("HEALTH_CHECK_FAILURE_THRESHOLD", 3)
                ),
            },
        }

        logger.info(
            f"Monitoring: Sentry={'enabled' if config['sentry']['enabled'] else 'disabled'}, "
            f"Webhook={'enabled' if config['webhook']['enabled'] else 'disabled'}"
        )
        return config

    def get_redis_client(self):
        """Get configured Redis client"""
        if not self.redis_config["enabled"]:
            logger.warning("Redis not configured, using in-memory fallback")
            return None

        try:
            import redis

            if self.redis_config.get("url"):
                # Use URL-based connection
                client = redis.from_url(
                    self.redis_config["url"],
                    decode_responses=True,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=self.redis_config["connection_pool"][
                        "health_check_interval"
                    ],
                )
            else:
                # Use individual parameters
                client = redis.Redis(
                    host=self.redis_config["host"],
                    port=self.redis_config["port"],
                    password=self.redis_config["password"],
                    db=self.redis_config["db"],
                    ssl=self.redis_config["ssl"],
                    decode_responses=True,
                    socket_keepalive=True,
                    health_check_interval=self.redis_config["connection_pool"][
                        "health_check_interval"
                    ],
                )

            # Test connection
            client.ping()
            logger.info("Redis connection established successfully")
            return client

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None

    def get_database_engine(self):
        """Get configured database engine"""
        if self.database_config["type"] == "postgresql":
            try:
                from sqlalchemy import create_engine

                engine = create_engine(
                    self.database_config["url"],
                    pool_size=self.database_config["pool_size"],
                    max_overflow=self.database_config["max_overflow"],
                    pool_timeout=self.database_config["pool_timeout"],
                    pool_recycle=self.database_config["pool_recycle"],
                    echo=self.environment == "development",
                )

                logger.info("PostgreSQL database engine configured")
                return engine

            except Exception as e:
                logger.error(f"Failed to configure PostgreSQL engine: {e}")
                raise

        else:  # SQLite
            try:
                import aiosqlite

                # For SQLite, we'll use aiosqlite directly
                logger.info(
                    f"SQLite database configured at: {self.database_config['path']}"
                )
                return self.database_config["path"]

            except Exception as e:
                logger.error(f"Failed to configure SQLite: {e}")
                raise

    def setup_sentry_monitoring(self):
        """Setup Sentry error tracking"""
        if not self.monitoring_config["sentry"]["enabled"]:
            logger.info("Sentry monitoring disabled")
            return

        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.asyncio import AsyncioIntegration
            from sentry_sdk.integrations.aiohttp import AioHttpIntegration

            sentry_sdk.init(
                dsn=self.monitoring_config["sentry"]["dsn"],
                integrations=[
                    FastApiIntegration(auto_enabling_integrations=False),
                    AsyncioIntegration(),
                    AioHttpIntegration(),
                ],
                traces_sample_rate=self.monitoring_config["sentry"]["sample_rate"],
                environment=self.monitoring_config["sentry"]["environment"],
                debug=self.monitoring_config["sentry"]["debug"],
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send personally identifiable information
            )

            logger.info("Sentry monitoring initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")

    async def test_external_services(self) -> Dict[str, bool]:
        """Test connectivity to all external services"""
        results = {}

        # Test Redis
        redis_client = self.get_redis_client()
        if redis_client:
            try:
                redis_client.ping()
                results["redis"] = True
            except:
                results["redis"] = False
        else:
            results["redis"] = False

        # Test Database
        try:
            if self.database_config["type"] == "postgresql":
                engine = self.get_database_engine()
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                results["database"] = True
            else:  # SQLite
                import aiosqlite

                async with aiosqlite.connect(self.database_config["path"]) as db:
                    await db.execute("SELECT 1")
                results["database"] = True
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            results["database"] = False

        # Test Webhook
        if self.monitoring_config["webhook"]["enabled"]:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    test_payload = {"test": "connection", "timestamp": "now"}
                    async with session.post(
                        self.monitoring_config["webhook"]["url"],
                        json=test_payload,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        results["webhook"] = response.status < 400
            except:
                results["webhook"] = False
        else:
            results["webhook"] = None  # Not configured

        return results

    def get_service_urls(self) -> Dict[str, str]:
        """Get URLs for external services (for monitoring)"""
        urls = {}

        if self.redis_config["enabled"]:
            urls["redis"] = self.redis_config["url"]

        if self.database_config["type"] == "postgresql":
            # Mask password in URL for logging
            url = self.database_config["url"]
            if "@" in url:
                scheme, rest = url.split("://", 1)
                if "@" in rest:
                    credentials, host_part = rest.split("@", 1)
                    username = credentials.split(":")[0]
                    urls["database"] = f"{scheme}://{username}:***@{host_part}"
                else:
                    urls["database"] = url
            else:
                urls["database"] = url
        else:
            urls["database"] = f"sqlite://{self.database_config['path']}"

        if self.monitoring_config["webhook"]["enabled"]:
            urls["webhook"] = self.monitoring_config["webhook"]["url"]

        return urls


# Global configuration instance
external_services = ExternalServicesConfig()
