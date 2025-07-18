# Multi-User Trading Bot - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tradingbot
RUN chown -R tradingbot:tradingbot /app
USER tradingbot

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs

# Expose port for health checks
EXPOSE 8080

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; from services.monitoring_service import monitoring_service; print('healthy')" || exit 1

# Run the application
CMD ["python", "production_main.py"] 