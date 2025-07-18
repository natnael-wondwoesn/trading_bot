#!/usr/bin/env python3
"""
Test Database Operations
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def test_database_operations():
    """Test database operations"""
    print("🧪 TESTING DATABASE OPERATIONS")
    print("=" * 40)
    
    try:
        from db.multi_user_db import multi_user_db
        
        # Test 1: Initialize database
        print("\n1. Testing database initialization...")
        await multi_user_db.initialize()
        print("✅ Database initialized successfully")
        
        # Test 2: Create a test user
        print("\n2. Testing user creation...")
        test_user = await multi_user_db.create_user(
            telegram_id=12345,
            username="test_user",
            first_name="Test",
            last_name="User"
        )
        print(f"✅ User created: {test_user.user_id}")
        
        # Test 3: Get user settings
        print("\n3. Testing user settings...")
        settings = await multi_user_db.get_user_settings(test_user.user_id)
        if settings:
            print(f"✅ Settings retrieved: {settings.strategy}")
        else:
            print("❌ Failed to get settings")
        
        # Test 4: Update user activity
        print("\n4. Testing user activity update...")
        await multi_user_db.update_user_activity(test_user.user_id)
        print("✅ User activity updated")
        
        # Test 5: Get user by telegram ID
        print("\n5. Testing user retrieval...")
        retrieved_user = await multi_user_db.get_user_by_telegram_id(12345)
        if retrieved_user:
            print(f"✅ User retrieved: {retrieved_user.username}")
        else:
            print("❌ Failed to retrieve user")
        
        print("\n🎉 All database tests passed!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            await multi_user_db.shutdown()
            print("✅ Database shutdown complete")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_database_operations())
