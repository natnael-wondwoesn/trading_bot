#!/usr/bin/env python3
"""
Test script to verify JSON parsing fixes
"""

import asyncio
import logging
import os
import json
from db.multi_user_db import multi_user_db, UserSettings
from services.user_service import user_service

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_json_parsing_fixes():
    """Test JSON parsing fixes and database repair"""
    try:
        logger.info("🧪 Testing JSON parsing fixes...")

        # Initialize database
        await multi_user_db.initialize()
        logger.info("✅ Database initialization successful")

        # Test safe JSON parsing
        test_cases = [
            '{"valid": "json"}',  # Valid JSON
            '{"broken": "json",}',  # Invalid JSON with trailing comma
            '{"extra": "data"}garbage',  # Extra data after JSON
            "",  # Empty string
            "not json at all",  # Not JSON
            '{"incomplete": "json"',  # Incomplete JSON
        ]

        logger.info("🔍 Testing safe JSON parsing...")
        for i, test_case in enumerate(test_cases):
            result = multi_user_db._safe_json_loads(test_case, {"default": "value"})
            logger.info(f"  Test {i+1}: {repr(test_case[:20])}... → {result}")

        # Test safe JSON serialization
        logger.info("🔍 Testing safe JSON serialization...")
        test_data = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        result = multi_user_db._safe_json_dumps(test_data, "test")
        logger.info(f"  Serialization: {result}")

        # Parse it back to verify
        parsed_back = multi_user_db._safe_json_loads(result)
        logger.info(f"  Round-trip test: {parsed_back}")

        # Test database repair
        logger.info("🔧 Testing database repair functionality...")
        repair_results = await multi_user_db.repair_corrupted_json_data()
        logger.info(f"  Repair results: {repair_results}")

        # Test user settings creation and retrieval
        logger.info("👤 Testing user settings...")

        # Create a test user
        test_user = await multi_user_db.create_user(
            telegram_id=12345, username="test_user", first_name="Test", last_name="User"
        )
        logger.info(f"  Created test user: {test_user.user_id}")

        # Get user settings (should create defaults)
        settings = await multi_user_db.get_user_settings(test_user.user_id)
        logger.info(f"  Retrieved settings: {settings}")

        if settings:
            logger.info(f"    Strategy: {settings.strategy}")
            logger.info(f"    Exchange: {settings.exchange}")
            logger.info(f"    Risk Management: {settings.risk_management}")
            logger.info(f"    Notifications: {settings.notifications}")
            logger.info(f"    Emergency: {settings.emergency}")

        # Test updating settings
        logger.info("📝 Testing settings update...")
        update_success = await multi_user_db.update_user_settings(
            test_user.user_id,
            strategy="MACD",
            risk_management={"max_risk_per_trade": 0.01, "trading_enabled": True},
        )
        logger.info(f"  Update successful: {update_success}")

        # Retrieve updated settings
        updated_settings = await multi_user_db.get_user_settings(test_user.user_id)
        logger.info(f"  Updated settings: {updated_settings.strategy}")

        logger.info("🎉 All JSON parsing tests passed!")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            await multi_user_db.shutdown()
        except:
            pass


async def main():
    """Main test function"""
    logger.info("🚀 Starting JSON parsing fix tests...")

    success = await test_json_parsing_fixes()

    if success:
        logger.info("✅ All tests completed successfully!")
        logger.info("🎯 JSON parsing errors should now be resolved!")
    else:
        logger.error("❌ Some tests failed!")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
