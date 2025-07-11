# 🚀 24/7 Trading Bot - Render Deployment Guide

This guide will help you deploy your trading bot to Render for 24/7 operation with zero downtime.

## 📋 Prerequisites

Before deploying to Render, make sure you have:

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Telegram Bot Token**: Create a bot via [@BotFather](https://t.me/botfather)
4. **Exchange API Keys**: Configure your trading exchange APIs (MEXC, Bybit, etc.)

## 🔧 Step 1: Repository Setup

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Verify these files are in your repository**:
   - `Dockerfile.render` - Optimized Dockerfile for Render
   - `requirements_render.txt` - Streamlined dependencies
   - `render_entrypoint.py` - Custom entry point for Render
   - `render.yaml` - Render configuration
   - `health_check_render.py` - Health monitoring
   - `external_services_config.py` - External services config

## 🚀 Step 2: Deploy to Render

### Option A: Using render.yaml (Recommended)

1. **Connect Repository**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`

2. **Review Configuration**:
   - Service name: `trading-bot-24-7`
   - Region: Choose closest to your users
   - Plan: Start with `Standard` (can upgrade later)

### Option B: Manual Setup

1. **Create Web Service**:
   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure settings:

   ```
   Name: trading-bot-24-7
   Region: Oregon (or closest to you)
   Branch: main
   Runtime: Docker
   Dockerfile Path: ./Dockerfile.render
   ```

2. **Build Configuration**:
   ```
   Build Command: (leave empty - Docker handles it)
   Start Command: python render_entrypoint.py
   ```

## ⚙️ Step 3: Environment Variables

In your Render service settings, add these environment variables:

### Required Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `TELEGRAM_BOT_TOKEN` | `your_bot_token` | Your Telegram bot token |
| `ENVIRONMENT` | `production` | Deployment environment |
| `LOG_LEVEL` | `INFO` | Logging level |

### Optional Variables (for enhanced features)

| Variable | Example | Description |
|----------|---------|-------------|
| `MEXC_API_KEY` | `your_mexc_key` | MEXC exchange API key |
| `MEXC_API_SECRET` | `your_mexc_secret` | MEXC exchange API secret |
| `BYBIT_API_KEY` | `your_bybit_key` | Bybit exchange API key |
| `BYBIT_API_SECRET` | `your_bybit_secret` | Bybit exchange API secret |
| `SENTRY_DSN` | `https://...` | Error tracking (optional) |
| `REDIS_URL` | `redis://...` | External Redis (optional) |
| `WEBHOOK_URL` | `https://...` | Notification webhook |

### Performance Tuning Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_USERS` | `1000` | Maximum concurrent users |
| `HEALTH_CHECK_INTERVAL` | `60` | Health check interval (seconds) |
| `DB_POOL_SIZE` | `20` | Database connection pool size |

## 💾 Step 4: Persistent Storage

Render provides persistent disks for data storage:

1. **Add Persistent Disk**:
   - In service settings, go to "Disks"
   - Add disk: `trading-data` (5GB recommended)
   - Mount path: `/tmp/trading_data`

2. **Database Storage**:
   - SQLite: Stored on persistent disk
   - PostgreSQL: Use Render's PostgreSQL add-on (recommended for production)

## 🔄 Step 5: External Services (Optional but Recommended)

### Redis for Caching

1. **Add Redis Service**:
   - In Render Dashboard: "New" → "Redis"
   - Plan: `Starter` (upgrade as needed)
   - Note the Redis URL

2. **Update Environment Variable**:
   ```
   REDIS_URL=redis://red-xxx:6379
   ```

### PostgreSQL Database

1. **Add PostgreSQL Service**:
   - In Render Dashboard: "New" → "PostgreSQL"
   - Plan: `Starter` (upgrade as needed)
   - Note the Database URL

2. **Update Environment Variable**:
   ```
   DATABASE_URL=postgresql://user:pass@host:5432/db
   ```

## 🏥 Step 6: Health Monitoring

Your bot includes comprehensive health monitoring:

### Health Check Endpoints

- **Primary**: `https://your-service.onrender.com/health`
- **Metrics**: `https://your-service.onrender.com/metrics`
- **Status**: `https://your-service.onrender.com/status`

### Monitoring Features

- ✅ System resource monitoring
- ✅ Database connectivity checks
- ✅ Memory usage tracking
- ✅ Network connectivity tests
- ✅ Service health validation

## 🚨 Step 7: Error Tracking (Optional)

### Setup Sentry

1. **Create Sentry Account**: [sentry.io](https://sentry.io)
2. **Create Project**: Choose Python/FastAPI
3. **Get DSN**: Copy your project DSN
4. **Add Environment Variable**:
   ```
   SENTRY_DSN=https://xxx@sentry.io/xxx
   ```

## 📊 Step 8: Verification

After deployment, verify everything is working:

1. **Check Service Status**:
   ```bash
   curl https://your-service.onrender.com/health
   ```

2. **Verify Bot Response**:
   - Send `/start` to your Telegram bot
   - Check for proper response

3. **Monitor Logs**:
   - In Render Dashboard → Your Service → Logs
   - Look for "✅ All services started successfully!"

## 🔧 Troubleshooting

### Common Issues

1. **Bot Not Responding**:
   - Check `TELEGRAM_BOT_TOKEN` environment variable
   - Verify bot token with [@BotFather](https://t.me/botfather)

2. **Memory Issues**:
   - Upgrade to `Pro` plan (more RAM)
   - Check memory usage: `/metrics` endpoint

3. **Database Errors**:
   - Check disk space: `/status` endpoint
   - Consider upgrading to PostgreSQL

4. **Build Failures**:
   - Check `requirements_render.txt` for conflicts
   - Review build logs in Render Dashboard

### Debug Commands

```bash
# Check service health
curl https://your-service.onrender.com/health

# View metrics
curl https://your-service.onrender.com/metrics

# Check detailed status
curl https://your-service.onrender.com/status
```

## 🎯 Performance Optimization

### Resource Allocation

- **Starter Plan**: Good for testing, limited users
- **Standard Plan**: Recommended for production, up to 1000 users
- **Pro Plan**: High traffic, enterprise features

### Scaling Recommendations

| Users | Plan | RAM | CPU | Storage |
|-------|------|-----|-----|---------|
| 1-100 | Starter | 512MB | 0.1 CPU | 1GB |
| 100-1000 | Standard | 2GB | 1 CPU | 5GB |
| 1000+ | Pro | 4GB+ | 2+ CPU | 10GB+ |

## 🔄 Maintenance

### Auto-Deployment

Your service will automatically redeploy when you push to GitHub:

```bash
git add .
git commit -m "Update trading bot"
git push origin main
# Render automatically deploys in ~3-5 minutes
```

### Monitoring

- **Health Checks**: Automatic every 30 seconds
- **Error Tracking**: Via Sentry (if configured)
- **Performance**: Via `/metrics` endpoint

### Backup Strategy

1. **Database Backups**: 
   - SQLite: Downloaded from persistent disk
   - PostgreSQL: Render provides automatic backups

2. **Configuration Backup**:
   - Environment variables: Document in secure location
   - Code: Stored in GitHub repository

## 🛡️ Security

### Best Practices

1. **Environment Variables**: Never commit secrets to GitHub
2. **API Keys**: Use read-only permissions when possible
3. **Database**: Use PostgreSQL for production
4. **Monitoring**: Enable Sentry for error tracking

### Security Features

- ✅ Non-root container user
- ✅ Minimal Docker image
- ✅ Environment-based secrets
- ✅ Health check validation
- ✅ Error tracking integration

## 💰 Cost Estimation

### Monthly Costs (USD)

| Component | Starter | Standard | Pro |
|-----------|---------|----------|-----|
| Web Service | $7 | $25 | $85 |
| PostgreSQL | $7 | $25 | $90 |
| Redis | $3 | $15 | $50 |
| **Total** | **$17** | **$65** | **$225** |

*Prices may vary - check Render pricing page*

## 📞 Support

### Getting Help

1. **Render Support**: [render.com/support](https://render.com/support)
2. **Documentation**: [render.com/docs](https://render.com/docs)
3. **Health Monitoring**: Use built-in endpoints
4. **Logs**: Available in Render Dashboard

### Community

- **Render Discord**: Community support
- **GitHub Issues**: For bot-specific problems

---

## 🎉 Congratulations!

Your 24/7 trading bot is now deployed on Render! 

**Next Steps**:
1. Monitor performance via health endpoints
2. Set up alerting via webhooks
3. Scale resources as user base grows
4. Consider upgrading to Pro plan for production

**Your bot is now running 24/7 with:**
- ✅ Automatic scaling
- ✅ Health monitoring
- ✅ Error tracking
- ✅ Zero-downtime deployments
- ✅ Global CDN
- ✅ SSL certificates

Happy trading! 🚀📈 