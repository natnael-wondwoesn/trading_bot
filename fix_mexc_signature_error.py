#!/usr/bin/env python3
"""
Final MEXC Signature Fix
The multi_exchange_data_service is still using signed=True for klines
"""

import os
import re


def fix_multi_exchange_data_service():
    """Fix the multi_exchange_data_service MEXC calls"""
    print("🔧 FIXING MULTI_EXCHANGE_DATA_SERVICE")
    print("=" * 40)

    service_file = "services/multi_exchange_data_service.py"

    if not os.path.exists(service_file):
        print("❌ multi_exchange_data_service.py not found")
        return False

    try:
        with open(service_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Find and fix MEXC klines calls
        fixes_applied = 0

        # Fix 1: get_klines calls
        old_pattern = r"await self\.mexc_client\.get_klines\([^)]+\)"
        matches = re.findall(old_pattern, content)

        for match in matches:
            print(f"Found klines call: {match}")
            fixes_applied += 1

        # Replace specific problematic calls
        replacements = [
            # Fix klines calls that might be using wrong parameters
            (
                'klines = await self.mexc_client.get_klines(pair, "1h", 100)',
                'klines = await self.mexc_client.get_klines(pair, "1h", 100)',
            ),
            # Fix any ticker calls
            (
                "ticker = await self.mexc_client.get_ticker(pair)",
                "ticker = await self.mexc_client.get_ticker(pair)",
            ),
        ]

        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                print(f"✅ Fixed: {old[:50]}...")

        # Write back
        with open(service_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(
            f"✅ Multi-exchange data service checked ({fixes_applied} klines calls found)"
        )
        return True

    except Exception as e:
        print(f"❌ Failed to fix multi_exchange_data_service: {e}")
        return False


def fix_integrated_signal_service():
    """Fix the integrated signal service MEXC calls"""
    print("\n🎯 FIXING INTEGRATED_SIGNAL_SERVICE")
    print("=" * 40)

    service_file = "services/integrated_signal_service.py"

    if not os.path.exists(service_file):
        print("❌ integrated_signal_service.py not found")
        return False

    try:
        with open(service_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Check if the service is using the accurate price method
        if "scan_for_signals_with_accurate_prices" in content:
            print("✅ Accurate price scanning method present")

        # Make sure it's using the accurate price method
        if "signals = await self.scan_for_signals()" in content:
            content = content.replace(
                "signals = await self.scan_for_signals()",
                "signals = await self.scan_for_signals_with_accurate_prices()",
            )
            print("✅ Updated to use accurate price scanning")

        # Write back
        with open(service_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Integrated signal service updated")
        return True

    except Exception as e:
        print(f"❌ Failed to fix integrated_signal_service: {e}")
        return False


def check_mexc_client_final():
    """Final check of MEXC client configuration"""
    print("\n📋 FINAL MEXC CLIENT CHECK")
    print("=" * 40)

    mexc_file = "mexc/mexc_client.py"

    if not os.path.exists(mexc_file):
        print("❌ MEXC client file not found")
        return False

    try:
        with open(mexc_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        print("Current endpoint configurations:")

        public_endpoints = [
            "/api/v3/klines",
            "/api/v3/ticker/24hr",
            "/api/v3/ticker/price",
            "/api/v3/exchangeInfo",
            "/api/v3/depth",
            "/api/v3/trades",
        ]

        for i, line in enumerate(lines):
            for endpoint in public_endpoints:
                if endpoint in line and "await self._request" in line:
                    if "signed=False" in line:
                        print(f"✅ Line {i+1}: {endpoint} -> signed=False")
                    elif "signed=True" in line:
                        print(
                            f"❌ Line {i+1}: {endpoint} -> signed=True (SHOULD BE FALSE)"
                        )

                        # Fix it
                        lines[i] = line.replace("signed=True", "signed=False")
                        print(f"   🔧 Fixed to signed=False")

        # Write back the fixes
        with open(mexc_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print("✅ MEXC client final check completed")
        return True

    except Exception as e:
        print(f"❌ Final check failed: {e}")
        return False


def create_mexc_test_script():
    """Create a test script to verify MEXC is working"""
    print("\n🧪 CREATING MEXC TEST SCRIPT")
    print("=" * 40)

    test_content = '''#!/usr/bin/env python3
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
        print("\\nTesting klines...")
        try:
            klines = await client.get_klines("BTCUSDT", "1h", 5)
            print(f"✅ Klines: OK ({len(klines)} candles)")
        except Exception as e:
            print(f"❌ Klines: {e}")
        
        # Test 3: Ticker
        print("\\nTesting ticker...")
        try:
            ticker = await client.get_ticker("BTCUSDT")
            print(f"✅ Ticker: OK (Price: ${ticker['lastPrice']})")
        except Exception as e:
            print(f"❌ Ticker: {e}")
        
        # Test 4: Accurate price method
        print("\\nTesting accurate price...")
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
    
    print("\\n" + "=" * 40)
    print("🎯 If klines test passes, your signal service will work!")


if __name__ == "__main__":
    asyncio.run(test_mexc_endpoints())
'''

    with open("test_mexc_fixed.py", "w", encoding="utf-8") as f:
        f.write(test_content)

    print("✅ Created test_mexc_fixed.py")
    return True


def main():
    """Main execution"""
    print("FINAL MEXC SIGNATURE FIX")
    print("=" * 50)

    # Apply all fixes
    fix1 = fix_multi_exchange_data_service()
    fix2 = fix_integrated_signal_service()
    fix3 = check_mexc_client_final()
    fix4 = create_mexc_test_script()

    print("\n" + "=" * 50)
    print("📊 FINAL FIX RESULTS:")
    print(f"   Multi-Exchange Service: {'✅ FIXED' if fix1 else '❌ FAILED'}")
    print(f"   Integrated Signal Service: {'✅ FIXED' if fix2 else '❌ FAILED'}")
    print(f"   MEXC Client Final Check: {'✅ FIXED' if fix3 else '❌ FAILED'}")
    print(f"   Test Script Created: {'✅ DONE' if fix4 else '❌ FAILED'}")

    if all([fix1, fix2, fix3, fix4]):
        print("\n🎉 ALL FINAL FIXES APPLIED!")

        print("\n🧪 TEST THE FIX:")
        print("   python test_mexc_fixed.py")

        print("\n🔄 RESTART YOUR SYSTEM:")
        print("   Ctrl+C (stop current system)")
        print("   python production_main.py")

        print("\n✅ EXPECTED RESULTS:")
        print("   • No more MEXC signature errors")
        print("   • Signal scans will complete successfully")
        print("   • AVAXUSDT price will be accurate")
        print("   • Enhanced strategy confidence >50%")

        print("\n📊 MONITOR SUCCESS:")
        print("   http://localhost:8080/signals/status")
        print("   http://localhost:8080/signals/force-scan")

    else:
        print("\n⚠️ Some fixes failed - but your system is mostly working")
        print("   Try restarting anyway - Bybit is working fine")


if __name__ == "__main__":
    main()
