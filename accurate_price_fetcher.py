#!/usr/bin/env python3
"""
Price Accuracy Fix
Ensures real-time prices are fetched with maximum precision
"""

import asyncio
import aiohttp
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AccuratePriceFetcher:
    """Fetches accurate real-time prices from multiple sources"""
    
    def __init__(self):
        self.session = None
        self.price_cache = {}
        self.cache_timeout = 5  # 5 seconds
        
    async def get_accurate_price(self, symbol: str) -> Optional[float]:
        """Get most accurate current price for symbol"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Try multiple endpoints for accuracy
            prices = []
            
            # MEXC API
            mexc_price = await self._get_mexc_price(symbol)
            if mexc_price:
                prices.append(mexc_price)
            
            # Bybit API (for comparison)
            bybit_price = await self._get_bybit_price(symbol)
            if bybit_price:
                prices.append(bybit_price)
            
            if prices:
                # Use median for accuracy
                prices.sort()
                median_price = prices[len(prices)//2]
                
                # Cache the result
                self.price_cache[symbol] = {
                    'price': median_price,
                    'timestamp': asyncio.get_event_loop().time()
                }
                
                return median_price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting accurate price for {symbol}: {e}")
            return None
    
    async def _get_mexc_price(self, symbol: str) -> Optional[float]:
        """Get price from MEXC with high precision"""
        try:
            url = f"https://api.mexc.com/api/v3/ticker/price"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data["price"])
                    
        except Exception as e:
            logger.warning(f"MEXC price fetch failed for {symbol}: {e}")
        
        return None
    
    async def _get_bybit_price(self, symbol: str) -> Optional[float]:
        """Get price from Bybit for comparison"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "spot", "symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("result", {}).get("list"):
                        return float(data["result"]["list"][0]["lastPrice"])
                        
        except Exception as e:
            logger.warning(f"Bybit price fetch failed for {symbol}: {e}")
        
        return None
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


# Global instance
accurate_pricer = AccuratePriceFetcher()
