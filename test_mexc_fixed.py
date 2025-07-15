#!/usr/bin/env python3
"""
Test MEXC API after fixes
"""

import asyncio
import logging
from mexc.mexc_client import MEXCClient

logging.basicConfig(level=logging.INFO)

async def test_mexc_endpoints():
    """Test all MEXC public endpoints"""
    print("🧪 TESTING MEXC PUBLIC ENDPOINTS")
    print("=" * 40)
    
    client = MEXCClient("test", "test")
    
    try:
        # Test 1: Exchange info (should work)
        print("Testing exchange info...")
        try:
            info = await client.get_exchange_info()
            print("✅ Exchange info: OK")
        except Exception as e:
            print(f"❌ Exchange info: {e}")
        
        # Test 2: Klines (the problematic one)
        print("\nTesting klines...")
        try:
            klines = await client.get_klines("BTCUSDT", "1h", 5)
            print(f"✅ Klines: OK ({len(klines)} candles)")
        except Exception as e:
            print(f"❌ Klines: {e}")
        
        # Test 3: Ticker
        print("\nTesting ticker...")
        try:
            ticker = await client.get_ticker("BTCUSDT")
            print(f"✅ Ticker: OK (Price: ${ticker['lastPrice']})")
        except Exception as e:
            print(f"❌ Ticker: {e}")
        
        # Test 4: Accurate price method
        print("\nTesting accurate price...")
        try:
            if hasattr(client, 'get_accurate_price'):
                price = await client.get_accurate_price("AVAXUSDT")
                print(f"✅ Accurate price: OK (AVAXUSDT: ${price:.4f})")
            else:
                print("⚠️ get_accurate_price method not found")
        except Exception as e:
            print(f"❌ Accurate price: {e}")
            
    finally:
        await client.close()
    
    print("\n" + "=" * 40)
    print("🎯 If klines test passes, your signal service will work!")


if __name__ == "__main__":
    asyncio.run(test_mexc_endpoints())
