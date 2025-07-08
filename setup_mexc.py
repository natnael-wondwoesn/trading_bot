"""
MEXC API Setup Guide

1. Go to https://www.mexc.com/user/openapi
2. Click "Create API"
3. Set API name and permissions:
   - Enable "Read" permission
   - Enable "Trade" permission
   - Do NOT enable "Withdraw" permission
4. Complete security verification
5. Save your API Key and Secret Key
6. Add them to your .env file

Security Tips:
- Never share your API keys
- Use IP whitelist if possible
- Regularly rotate your API keys
- Monitor API usage
"""

import asyncio
from mexc.mexc_client import MEXCClient
from config import Config


async def test_connection():
    """Test MEXC API connection"""
    print("Testing MEXC API connection...")

    try:
        client = MEXCClient(Config.MEXC_API_KEY, Config.MEXC_API_SECRET)

        # Test public endpoint
        print("\n1. Testing public endpoint...")
        ticker = await client.get_ticker("BTCUSDT")
        print(f"✓ BTC/USDT Price: ${ticker['lastPrice']}")

        # Test authenticated endpoint
        print("\n2. Testing authenticated endpoint...")
        account = await client.get_account()
        print("✓ Account connected successfully")

        # Show balances
        print("\n3. Account balances:")
        balances = await client.get_balance()
        for asset, balance in balances.items():
            if balance["free"] > 0 or balance["locked"] > 0:
                print(
                    f"   {asset}: {balance['free']} (free) + {balance['locked']} (locked)"
                )

        # Test order book
        print("\n4. Testing order book...")
        orderbook = await client.get_orderbook("BTCUSDT", 5)
        print(f"✓ Order book data received")
        print(f"   Best bid: ${orderbook['bids'][0][0]}")
        print(f"   Best ask: ${orderbook['asks'][0][0]}")

        await client.close()
        print("\n✅ All tests passed! MEXC API is properly configured.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check:")
        print("1. Your API keys are correctly set in .env")
        print("2. Your API has proper permissions")
        print("3. Your IP is whitelisted (if applicable)")


if __name__ == "__main__":
    Config.validate()
    asyncio.run(test_connection())
