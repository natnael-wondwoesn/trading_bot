#!/usr/bin/env python3
"""
Monitoring and Health Check Service for 24/7 Trading Platform
Comprehensive system monitoring, alerting, and performance tracking
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import aiohttp
from pathlib import Path

from db.multi_user_db import multi_user_db
from services.user_service import user_service
from services.trading_orchestrator import trading_orchestrator

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    name: str
    value: float
    status: str  # 'healthy', 'warning', 'critical'
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""


@dataclass
class SystemAlert:
    alert_id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    component: str
    message: str
    details: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class PerformanceTracker:
    """Track and analyze system performance"""

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.performance_baselines: Dict[str, float] = {}

    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        timestamp = datetime.now()

        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []

        self.metrics_history[metric_name].append((timestamp, value))

        # Clean old data
        cutoff = timestamp - timedelta(days=self.retention_days)
        self.metrics_history[metric_name] = [
            (ts, val) for ts, val in self.metrics_history[metric_name] if ts > cutoff
        ]

    def get_metric_stats(self, metric_name: str, hours: int = 24) -> Dict:
        """Get statistics for a metric over specified hours"""
        if metric_name not in self.metrics_history:
            return {}

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_data = [
            val for ts, val in self.metrics_history[metric_name] if ts > cutoff
        ]

        if not recent_data:
            return {}

        return {
            "count": len(recent_data),
            "min": min(recent_data),
            "max": max(recent_data),
            "avg": sum(recent_data) / len(recent_data),
            "latest": recent_data[-1] if recent_data else 0,
        }

    def detect_anomalies(self, metric_name: str) -> List[str]:
        """Detect performance anomalies"""
        anomalies = []
        stats = self.get_metric_stats(metric_name, hours=24)

        if not stats:
            return anomalies

        # Check for baseline deviation
        baseline = self.performance_baselines.get(metric_name, stats["avg"])
        deviation = abs(stats["latest"] - baseline) / baseline if baseline > 0 else 0

        if deviation > 0.5:  # 50% deviation
            anomalies.append(f"{metric_name} deviating {deviation:.1%} from baseline")

        # Check for sudden spikes
        if stats["latest"] > stats["avg"] * 2:
            anomalies.append(
                f"{metric_name} showing sudden spike: {stats['latest']:.2f}"
            )

        return anomalies


class AlertManager:
    """Manage system alerts and notifications"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        self.max_history = 1000

        # Alert thresholds
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=15)  # Prevent spam

    async def create_alert(
        self,
        severity: str,
        component: str,
        message: str,
        details: Dict = None,
        alert_key: str = None,
    ) -> SystemAlert:
        """Create a new system alert"""

        # Generate alert ID
        if not alert_key:
            alert_key = f"{component}_{int(time.time())}"

        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if datetime.now() < self.alert_cooldowns[alert_key]:
                return None  # Still in cooldown

        alert = SystemAlert(
            alert_id=alert_key,
            severity=severity,
            component=component,
            message=message,
            details=details or {},
        )

        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Set cooldown
        self.alert_cooldowns[alert_key] = datetime.now() + self.cooldown_period

        # Clean history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

        # Send notifications
        await self._send_alert_notification(alert)

        logger.warning(f"Alert created [{severity}] {component}: {message}")
        return alert

    async def resolve_alert(self, alert_key: str):
        """Resolve an active alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            del self.active_alerts[alert_key]

            logger.info(f"Alert resolved: {alert_key}")

    async def _send_alert_notification(self, alert: SystemAlert):
        """Send alert notification via webhook"""
        if not self.webhook_url:
            return

        try:
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "component": alert.component,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Alert notification sent: {alert.alert_id}")
                    else:
                        logger.error(
                            f"Failed to send alert notification: {response.status}"
                        )

        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")

    def get_active_alerts(self, severity: str = None) -> List[SystemAlert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)


class HealthChecker:
    """Comprehensive system health monitoring"""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.health_metrics: Dict[str, HealthMetric] = {}

    async def check_system_health(self) -> Dict[str, HealthMetric]:
        """Perform comprehensive system health check"""

        # Check all health metrics
        checks = [
            self._check_cpu_usage(),
            self._check_memory_usage(),
            self._check_disk_usage(),
            self._check_database_health(),
            self._check_bot_health(),
            self._check_trading_orchestrator_health(),
            self._check_user_service_health(),
            self._check_network_connectivity(),
            self._check_queue_health(),
        ]

        # Run all checks
        await asyncio.gather(*checks)

        # Check for critical issues
        await self._evaluate_overall_health()

        return self.health_metrics

    async def _check_cpu_usage(self):
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)

        metric = HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            status="healthy",
            threshold_warning=70.0,
            threshold_critical=90.0,
        )

        if cpu_percent > metric.threshold_critical:
            metric.status = "critical"
            metric.message = f"CPU usage critical: {cpu_percent:.1f}%"
            await self.alert_manager.create_alert(
                "critical", "system", metric.message, {"cpu_percent": cpu_percent}
            )
        elif cpu_percent > metric.threshold_warning:
            metric.status = "warning"
            metric.message = f"CPU usage high: {cpu_percent:.1f}%"
            await self.alert_manager.create_alert(
                "warning", "system", metric.message, {"cpu_percent": cpu_percent}
            )

        self.health_metrics["cpu_usage"] = metric

    async def _check_memory_usage(self):
        """Check memory usage"""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        metric = HealthMetric(
            name="memory_usage",
            value=memory_percent,
            status="healthy",
            threshold_warning=80.0,
            threshold_critical=95.0,
        )

        if memory_percent > metric.threshold_critical:
            metric.status = "critical"
            metric.message = f"Memory usage critical: {memory_percent:.1f}%"
            await self.alert_manager.create_alert(
                "critical",
                "system",
                metric.message,
                {
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                },
            )
        elif memory_percent > metric.threshold_warning:
            metric.status = "warning"
            metric.message = f"Memory usage high: {memory_percent:.1f}%"

        self.health_metrics["memory_usage"] = metric

    async def _check_disk_usage(self):
        """Check disk usage"""
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        metric = HealthMetric(
            name="disk_usage",
            value=disk_percent,
            status="healthy",
            threshold_warning=80.0,
            threshold_critical=95.0,
        )

        if disk_percent > metric.threshold_critical:
            metric.status = "critical"
            metric.message = f"Disk usage critical: {disk_percent:.1f}%"
            await self.alert_manager.create_alert(
                "critical",
                "system",
                metric.message,
                {"disk_percent": disk_percent, "free_gb": disk.free / (1024**3)},
            )
        elif disk_percent > metric.threshold_warning:
            metric.status = "warning"
            metric.message = f"Disk usage high: {disk_percent:.1f}%"

        self.health_metrics["disk_usage"] = metric

    async def _check_database_health(self):
        """Check database connectivity and performance"""
        try:
            start_time = time.time()

            # Test database connection
            async with multi_user_db.get_connection() as db:
                await db.execute("SELECT 1")

            response_time = (time.time() - start_time) * 1000  # ms

            metric = HealthMetric(
                name="database_response_time",
                value=response_time,
                status="healthy",
                threshold_warning=100.0,  # 100ms
                threshold_critical=500.0,  # 500ms
            )

            if response_time > metric.threshold_critical:
                metric.status = "critical"
                metric.message = (
                    f"Database response time critical: {response_time:.1f}ms"
                )
                await self.alert_manager.create_alert(
                    "critical",
                    "database",
                    metric.message,
                    {"response_time_ms": response_time},
                )
            elif response_time > metric.threshold_warning:
                metric.status = "warning"
                metric.message = f"Database response time high: {response_time:.1f}ms"

            self.health_metrics["database_response_time"] = metric

        except Exception as e:
            metric = HealthMetric(
                name="database_response_time",
                value=float("inf"),
                status="critical",
                threshold_warning=100.0,
                threshold_critical=500.0,
                message=f"Database connection failed: {str(e)}",
            )

            await self.alert_manager.create_alert(
                "critical", "database", metric.message, {"error": str(e)}
            )

            self.health_metrics["database_response_time"] = metric

    async def _check_bot_health(self):
        """Check Telegram bot health"""
        try:
            from services.multi_user_bot import multi_user_bot

            # Check if bot is initialized and running
            if hasattr(multi_user_bot, "application") and multi_user_bot.application:
                stats = multi_user_bot.get_system_stats()

                metric = HealthMetric(
                    name="bot_active_users",
                    value=stats.get("active_contexts", 0),
                    status="healthy",
                    threshold_warning=500,
                    threshold_critical=900,
                )

                if stats.get("active_contexts", 0) > metric.threshold_critical:
                    metric.status = "critical"
                    metric.message = (
                        f"Bot user load critical: {stats['active_contexts']} users"
                    )
                elif stats.get("active_contexts", 0) > metric.threshold_warning:
                    metric.status = "warning"
                    metric.message = (
                        f"Bot user load high: {stats['active_contexts']} users"
                    )

                self.health_metrics["bot_active_users"] = metric

                # Check message queue size
                queue_size = stats.get("queue_size", 0)
                queue_metric = HealthMetric(
                    name="bot_queue_size",
                    value=queue_size,
                    status="healthy",
                    threshold_warning=1000,
                    threshold_critical=5000,
                )

                if queue_size > queue_metric.threshold_critical:
                    queue_metric.status = "critical"
                    queue_metric.message = f"Bot queue size critical: {queue_size}"
                    await self.alert_manager.create_alert(
                        "critical",
                        "bot",
                        queue_metric.message,
                        {"queue_size": queue_size},
                    )
                elif queue_size > queue_metric.threshold_warning:
                    queue_metric.status = "warning"
                    queue_metric.message = f"Bot queue size high: {queue_size}"

                self.health_metrics["bot_queue_size"] = queue_metric
            else:
                raise Exception("Bot not initialized")

        except Exception as e:
            metric = HealthMetric(
                name="bot_health",
                value=0,
                status="critical",
                threshold_warning=1,
                threshold_critical=1,
                message=f"Bot health check failed: {str(e)}",
            )

            await self.alert_manager.create_alert(
                "critical", "bot", metric.message, {"error": str(e)}
            )

            self.health_metrics["bot_health"] = metric

    async def _check_trading_orchestrator_health(self):
        """Check trading orchestrator health"""
        try:
            stats = trading_orchestrator.get_system_stats()

            # Check active sessions
            active_sessions = stats.get("active_sessions", 0)
            session_metric = HealthMetric(
                name="orchestrator_active_sessions",
                value=active_sessions,
                status="healthy",
                threshold_warning=800,
                threshold_critical=950,
            )

            if active_sessions > session_metric.threshold_critical:
                session_metric.status = "critical"
                session_metric.message = (
                    f"Orchestrator sessions critical: {active_sessions}"
                )
                await self.alert_manager.create_alert(
                    "critical",
                    "orchestrator",
                    session_metric.message,
                    {"active_sessions": active_sessions},
                )
            elif active_sessions > session_metric.threshold_warning:
                session_metric.status = "warning"
                session_metric.message = (
                    f"Orchestrator sessions high: {active_sessions}"
                )

            self.health_metrics["orchestrator_active_sessions"] = session_metric

            # Check signal queue
            queue_size = stats.get("signal_queue_size", 0)
            queue_metric = HealthMetric(
                name="orchestrator_signal_queue",
                value=queue_size,
                status="healthy",
                threshold_warning=1000,
                threshold_critical=5000,
            )

            if queue_size > queue_metric.threshold_critical:
                queue_metric.status = "critical"
                queue_metric.message = f"Signal queue critical: {queue_size}"
                await self.alert_manager.create_alert(
                    "critical",
                    "orchestrator",
                    queue_metric.message,
                    {"signal_queue_size": queue_size},
                )

            self.health_metrics["orchestrator_signal_queue"] = queue_metric

        except Exception as e:
            metric = HealthMetric(
                name="orchestrator_health",
                value=0,
                status="critical",
                threshold_warning=1,
                threshold_critical=1,
                message=f"Orchestrator health check failed: {str(e)}",
            )

            self.health_metrics["orchestrator_health"] = metric

    async def _check_user_service_health(self):
        """Check user service health"""
        try:
            # Test user service functionality
            start_time = time.time()
            user_count = await multi_user_db.get_active_users_count()
            response_time = (time.time() - start_time) * 1000

            metric = HealthMetric(
                name="user_service_response_time",
                value=response_time,
                status="healthy",
                threshold_warning=50.0,
                threshold_critical=200.0,
            )

            if response_time > metric.threshold_critical:
                metric.status = "critical"
                metric.message = (
                    f"User service response critical: {response_time:.1f}ms"
                )
                await self.alert_manager.create_alert(
                    "critical",
                    "user_service",
                    metric.message,
                    {"response_time_ms": response_time},
                )
            elif response_time > metric.threshold_warning:
                metric.status = "warning"
                metric.message = f"User service response high: {response_time:.1f}ms"

            self.health_metrics["user_service_response_time"] = metric

        except Exception as e:
            metric = HealthMetric(
                name="user_service_health",
                value=0,
                status="critical",
                threshold_warning=1,
                threshold_critical=1,
                message=f"User service check failed: {str(e)}",
            )

            self.health_metrics["user_service_health"] = metric

    async def _check_network_connectivity(self):
        """Check external network connectivity"""
        try:
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://httpbin.org/status/200", timeout=5
                ) as response:
                    if response.status == 200:
                        response_time = (time.time() - start_time) * 1000

                        metric = HealthMetric(
                            name="network_connectivity",
                            value=response_time,
                            status="healthy",
                            threshold_warning=2000.0,
                            threshold_critical=5000.0,
                        )

                        if response_time > metric.threshold_warning:
                            metric.status = "warning"
                            metric.message = (
                                f"Network latency high: {response_time:.1f}ms"
                            )

                        self.health_metrics["network_connectivity"] = metric
                    else:
                        raise Exception(f"HTTP {response.status}")

        except Exception as e:
            metric = HealthMetric(
                name="network_connectivity",
                value=0,
                status="critical",
                threshold_warning=1,
                threshold_critical=1,
                message=f"Network connectivity failed: {str(e)}",
            )

            await self.alert_manager.create_alert(
                "warning", "network", metric.message, {"error": str(e)}
            )

            self.health_metrics["network_connectivity"] = metric

    async def _check_queue_health(self):
        """Check various queue health metrics"""
        try:
            # This would check your message queues, task queues, etc.
            # Implementation depends on your specific queue systems

            metric = HealthMetric(
                name="queue_health",
                value=100,  # Placeholder - implement actual queue monitoring
                status="healthy",
                threshold_warning=80,
                threshold_critical=60,
            )

            self.health_metrics["queue_health"] = metric

        except Exception as e:
            logger.error(f"Queue health check error: {e}")

    async def _evaluate_overall_health(self):
        """Evaluate overall system health and create alerts if needed"""
        critical_count = sum(
            1 for metric in self.health_metrics.values() if metric.status == "critical"
        )
        warning_count = sum(
            1 for metric in self.health_metrics.values() if metric.status == "warning"
        )

        if critical_count > 0:
            await self.alert_manager.create_alert(
                "critical",
                "system",
                f"System health critical: {critical_count} critical issues",
                {"critical_count": critical_count, "warning_count": warning_count},
                "system_health_critical",
            )
        elif warning_count > 3:
            await self.alert_manager.create_alert(
                "warning",
                "system",
                f"System health degraded: {warning_count} warnings",
                {"warning_count": warning_count},
                "system_health_degraded",
            )


class MonitoringService:
    """Main monitoring service orchestrating all monitoring components"""

    def __init__(self, webhook_url: str = None, check_interval: int = 60):
        self.check_interval = check_interval
        self.performance_tracker = PerformanceTracker()
        self.alert_manager = AlertManager(webhook_url)
        self.health_checker = HealthChecker(self.alert_manager)

        self.is_running = False
        self.monitoring_tasks = []

    async def start(self):
        """Start the monitoring service"""
        logger.info("Starting monitoring service...")

        self.is_running = True

        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._performance_monitoring_loop()),
            asyncio.create_task(self._alert_cleanup_task()),
        ]

        logger.info("Monitoring service started")

    async def stop(self):
        """Stop the monitoring service"""
        logger.info("Stopping monitoring service...")

        self.is_running = False

        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        logger.info("Monitoring service stopped")

    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()

                # Perform health checks
                await self.health_checker.check_system_health()

                # Record monitoring performance
                check_duration = time.time() - start_time
                self.performance_tracker.record_metric(
                    "health_check_duration", check_duration
                )

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Shorter sleep on error

    async def _performance_monitoring_loop(self):
        """Performance metrics collection loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self.performance_tracker.record_metric(
                    "cpu_usage", psutil.cpu_percent()
                )
                self.performance_tracker.record_metric(
                    "memory_usage", psutil.virtual_memory().percent
                )

                # Collect application metrics
                if hasattr(multi_user_db, "get_system_health"):
                    db_health = await multi_user_db.get_system_health()
                    self.performance_tracker.record_metric(
                        "active_users", db_health.get("active_users_24h", 0)
                    )
                    self.performance_tracker.record_metric(
                        "trades_today", db_health.get("trades_today", 0)
                    )

                # Check for anomalies
                for metric_name in self.performance_tracker.metrics_history.keys():
                    anomalies = self.performance_tracker.detect_anomalies(metric_name)
                    for anomaly in anomalies:
                        await self.alert_manager.create_alert(
                            "warning",
                            "performance",
                            anomaly,
                            {"metric": metric_name},
                            f"anomaly_{metric_name}",
                        )

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

    async def _alert_cleanup_task(self):
        """Clean up resolved alerts and old history"""
        while self.is_running:
            try:
                # Auto-resolve old alerts
                cutoff_time = datetime.now() - timedelta(hours=24)

                alerts_to_resolve = []
                for alert_key, alert in self.alert_manager.active_alerts.items():
                    if alert.timestamp < cutoff_time and alert.severity in [
                        "info",
                        "warning",
                    ]:
                        alerts_to_resolve.append(alert_key)

                for alert_key in alerts_to_resolve:
                    await self.alert_manager.resolve_alert(alert_key)

                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Error in alert cleanup: {e}")
                await asyncio.sleep(1800)  # 30 minutes on error

    # Public API
    def get_health_status(self) -> Dict:
        """Get current system health status"""
        metrics = self.health_checker.health_metrics

        overall_status = "healthy"
        if any(m.status == "critical" for m in metrics.values()):
            overall_status = "critical"
        elif any(m.status == "warning" for m in metrics.values()):
            overall_status = "warning"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: {
                    "value": metric.value,
                    "status": metric.status,
                    "message": metric.message,
                }
                for name, metric in metrics.items()
            },
            "active_alerts": len(self.alert_manager.active_alerts),
            "critical_alerts": len(self.alert_manager.get_active_alerts("critical")),
        }

    def get_performance_stats(self, metric_name: str = None, hours: int = 24) -> Dict:
        """Get performance statistics"""
        if metric_name:
            return self.performance_tracker.get_metric_stats(metric_name, hours)

        # Return all metrics
        all_stats = {}
        for name in self.performance_tracker.metrics_history.keys():
            all_stats[name] = self.performance_tracker.get_metric_stats(name, hours)

        return all_stats

    def get_alerts(self, severity: str = None, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        alerts = self.alert_manager.get_active_alerts(severity)
        alerts.extend(self.alert_manager.alert_history[-limit:])

        # Sort by timestamp and limit
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

        return [
            {
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "component": alert.component,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "details": alert.details,
            }
            for alert in alerts
        ]


# Global monitoring service
monitoring_service = MonitoringService()
