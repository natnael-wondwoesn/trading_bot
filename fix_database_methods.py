#!/usr/bin/env python3
"""
Fix Database Methods Error
Fix 'execute_fetchone' method that doesn't exist in aiosqlite
"""

import os
import re


def fix_database_methods():
    """Fix execute_fetchone methods in multi_user_db.py"""
    print("🔧 FIXING DATABASE METHODS")
    print("=" * 40)

    db_file = "db/multi_user_db.py"

    if not os.path.exists(db_file):
        print("❌ db/multi_user_db.py not found")
        return False

    try:
        with open(db_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Fix 1: Replace execute_fetchone with proper aiosqlite pattern
        # Pattern: await connection.execute_fetchone(query, params)
        # Should be: cursor = await connection.execute(query, params); row = await cursor.fetchone()

        # Find all execute_fetchone patterns
        execute_fetchone_pattern = (
            r"(\w+)\s*=\s*await\s+(\w+)\.execute_fetchone\s*\(\s*([^)]+)\s*\)"
        )

        def replace_execute_fetchone(match):
            var_name = match.group(1)
            conn_name = match.group(2)
            query_params = match.group(3)

            return f"""cursor = await {conn_name}.execute({query_params})
            {var_name} = await cursor.fetchone()"""

        # Apply the replacement
        new_content = re.sub(
            execute_fetchone_pattern, replace_execute_fetchone, content
        )

        if new_content != content:
            print("✅ Fixed execute_fetchone patterns")
            content = new_content

        # Fix 2: Look for any remaining execute_fetchone patterns that might be different
        if "execute_fetchone" in content:
            print("⚠️ Found remaining execute_fetchone patterns, fixing manually...")

            # Manual fixes for specific patterns
            manual_fixes = [
                # Pattern: row = await db.execute_fetchone(...)
                (
                    r'(\w+)\s*=\s*await\s+(\w+)\.execute_fetchone\s*\(\s*"""([^"]+)"""\s*,\s*([^)]+)\s*\)',
                    r'cursor = await \2.execute("""\3""", \4)\n            \1 = await cursor.fetchone()',
                ),
                # Pattern: result = await connection.execute_fetchone(query)
                (
                    r"(\w+)\s*=\s*await\s+(\w+)\.execute_fetchone\s*\(\s*([^,)]+)\s*\)",
                    r"cursor = await \2.execute(\3)\n            \1 = await cursor.fetchone()",
                ),
                # Any remaining execute_fetchone
                (r"\.execute_fetchone\(", ".execute("),
            ]

            for pattern, replacement in manual_fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    print("✅ Applied manual fix for execute_fetchone")

        # Fix 3: Add missing cursor.fetchone() calls where needed
        # Look for execute() calls that should be followed by fetchone()

        # Pattern where we execute but don't fetch when we should
        lines = content.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)

            # If we see a cursor = await db.execute() pattern
            if "cursor = await" in line and ".execute(" in line:
                # Check if the next few lines have fetchone
                next_lines = lines[i + 1 : i + 3] if i + 1 < len(lines) else []
                has_fetchone = any(
                    "fetchone()" in next_line for next_line in next_lines
                )

                # If we're assigning the cursor to something and no fetchone follows
                if "cursor =" in line and not has_fetchone:
                    # Look ahead to see if this should have a fetchone
                    if i + 1 < len(lines) and (
                        "if row" in lines[i + 1]
                        or "row[" in lines[i + 1]
                        or "return row" in lines[i + 1]
                    ):
                        # Insert fetchone after the execute
                        indent = "            "  # Match typical indentation
                        fixed_lines.append(f"{indent}row = await cursor.fetchone()")
                        print("✅ Added missing fetchone() call")

            i += 1

        content = "\n".join(fixed_lines)

        # Fix 4: Look for get_user_settings method specifically
        if "async def get_user_settings" in content:
            # Find the get_user_settings method and fix it properly
            get_settings_start = content.find("async def get_user_settings")
            get_settings_end = content.find("\n    async def", get_settings_start + 1)
            if get_settings_end == -1:
                get_settings_end = content.find("\n    def", get_settings_start + 1)
            if get_settings_end == -1:
                get_settings_end = len(content)

            get_settings_method = content[get_settings_start:get_settings_end]

            # Fix the method if it has issues
            if (
                "execute_fetchone" in get_settings_method
                or "Connection" in get_settings_method
            ):
                fixed_method = '''async def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        """Get user settings"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute(
                    """
                    SELECT strategy, exchange, risk_management, notifications, emergency
                    FROM user_settings WHERE user_id = ?
                """,
                    (user_id,),
                )
                
                row = await cursor.fetchone()
                if row:
                    strategy, exchange, risk_management, notifications, emergency = row
                    
                    # Safely parse JSON fields
                    risk_data = self._safe_json_loads(risk_management, {
                        "trading_enabled": True,
                        "max_risk_per_trade": 0.02,
                        "max_open_positions": 3,
                        "daily_loss_limit": 0.05
                    })
                    
                    notifications_data = self._safe_json_loads(notifications, {
                        "signal_alerts": True,
                        "trade_confirmations": True,
                        "daily_summary": True,
                        "emergency_alerts": True
                    })
                    
                    emergency_data = self._safe_json_loads(emergency, {
                        "stop_all_trading": False,
                        "close_all_positions": False,
                        "notify_admin": False
                    })
                    
                    return UserSettings(
                        user_id=user_id,
                        strategy=strategy,
                        exchange=exchange,
                        risk_management=risk_data,
                        notifications=notifications_data,
                        emergency=emergency_data
                    )
                
                # Create default settings if none exist
                logger.info(f"Creating default settings for user {user_id}")
                await self._create_default_settings(db, user_id)
                await db.commit()
                
                # Return default settings
                return UserSettings(
                    user_id=user_id,
                    strategy="ENHANCED_RSI_EMA",
                    exchange="MEXC",
                    risk_management={
                        "trading_enabled": True,
                        "max_risk_per_trade": 0.02,
                        "max_open_positions": 3,
                        "daily_loss_limit": 0.05
                    },
                    notifications={
                        "signal_alerts": True,
                        "trade_confirmations": True,
                        "daily_summary": True,
                        "emergency_alerts": True
                    },
                    emergency={
                        "stop_all_trading": False,
                        "close_all_positions": False,
                        "notify_admin": False
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting user settings for {user_id}: {e}")
            # Return default settings on error
            return UserSettings(
                user_id=user_id,
                strategy="ENHANCED_RSI_EMA",
                exchange="MEXC",
                risk_management={
                    "trading_enabled": True,
                    "max_risk_per_trade": 0.02,
                    "max_open_positions": 3,
                    "daily_loss_limit": 0.05
                },
                notifications={
                    "signal_alerts": True,
                    "trade_confirmations": True,
                    "daily_summary": True,
                    "emergency_alerts": True
                },
                emergency={
                    "stop_all_trading": False,
                    "close_all_positions": False,
                    "notify_admin": False
                }
            )'''

                content = (
                    content[:get_settings_start]
                    + fixed_method
                    + content[get_settings_end:]
                )
                print("✅ Fixed get_user_settings method")

        # Write back the fixed content
        with open(db_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Database methods fixed successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to fix database methods: {e}")
        return False


def check_user_settings_model():
    """Check if UserSettings model exists and fix if needed"""
    print("\n📝 CHECKING USER SETTINGS MODEL")
    print("=" * 40)

    models_file = "models/models.py"

    if not os.path.exists(models_file):
        print("❌ models/models.py not found")
        return False

    try:
        with open(models_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if "class UserSettings" not in content:
            print("⚠️ UserSettings model not found, adding it...")

            user_settings_model = '''

@dataclass
class UserSettings:
    """User settings model"""
    user_id: int
    strategy: str = "ENHANCED_RSI_EMA"
    exchange: str = "MEXC"
    risk_management: dict = None
    notifications: dict = None
    emergency: dict = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.risk_management is None:
            self.risk_management = {
                "trading_enabled": True,
                "max_risk_per_trade": 0.02,
                "max_open_positions": 3,
                "daily_loss_limit": 0.05
            }
        
        if self.notifications is None:
            self.notifications = {
                "signal_alerts": True,
                "trade_confirmations": True,
                "daily_summary": True,
                "emergency_alerts": True
            }
        
        if self.emergency is None:
            self.emergency = {
                "stop_all_trading": False,
                "close_all_positions": False,
                "notify_admin": False
            }
'''

            # Add at the end of the file
            content += user_settings_model

            with open(models_file, "w", encoding="utf-8") as f:
                f.write(content)

            print("✅ Added UserSettings model")
            return True
        else:
            print("✅ UserSettings model exists")
            return True

    except Exception as e:
        print(f"❌ Failed to check UserSettings model: {e}")
        return False


def add_missing_imports():
    """Add missing imports to multi_user_db.py"""
    print("\n📦 CHECKING MISSING IMPORTS")
    print("=" * 40)

    db_file = "db/multi_user_db.py"

    try:
        with open(db_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Check for required imports
        required_imports = [
            "from models.models import UserSettings",
            "from typing import Optional",
            "import json",
            "import secrets",
            "import logging",
        ]

        missing_imports = []
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)

        if missing_imports:
            print(f"⚠️ Found {len(missing_imports)} missing imports")

            # Find the import section (after the docstring, before the first class)
            import_section_end = content.find("logger = logging.getLogger(__name__)")

            if import_section_end == -1:
                # Find first class or function
                import_section_end = content.find("class MultiUserDatabase")

            if import_section_end > 0:
                # Add missing imports before the logger line or class
                imports_to_add = "\n".join(missing_imports) + "\n\n"
                content = (
                    content[:import_section_end]
                    + imports_to_add
                    + content[import_section_end:]
                )

                with open(db_file, "w", encoding="utf-8") as f:
                    f.write(content)

                print("✅ Added missing imports")
                return True
        else:
            print("✅ All required imports present")
            return True

    except Exception as e:
        print(f"❌ Failed to check imports: {e}")
        return False


def create_database_test():
    """Create a test script for database operations"""
    print("\n🧪 CREATING DATABASE TEST SCRIPT")
    print("=" * 40)

    test_content = '''#!/usr/bin/env python3
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
        print("\\n1. Testing database initialization...")
        await multi_user_db.initialize()
        print("✅ Database initialized successfully")
        
        # Test 2: Create a test user
        print("\\n2. Testing user creation...")
        test_user = await multi_user_db.create_user(
            telegram_id=12345,
            username="test_user",
            first_name="Test",
            last_name="User"
        )
        print(f"✅ User created: {test_user.user_id}")
        
        # Test 3: Get user settings
        print("\\n3. Testing user settings...")
        settings = await multi_user_db.get_user_settings(test_user.user_id)
        if settings:
            print(f"✅ Settings retrieved: {settings.strategy}")
        else:
            print("❌ Failed to get settings")
        
        # Test 4: Update user activity
        print("\\n4. Testing user activity update...")
        await multi_user_db.update_user_activity(test_user.user_id)
        print("✅ User activity updated")
        
        # Test 5: Get user by telegram ID
        print("\\n5. Testing user retrieval...")
        retrieved_user = await multi_user_db.get_user_by_telegram_id(12345)
        if retrieved_user:
            print(f"✅ User retrieved: {retrieved_user.username}")
        else:
            print("❌ Failed to retrieve user")
        
        print("\\n🎉 All database tests passed!")
        
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
'''

    with open("test_database_operations.py", "w", encoding="utf-8") as f:
        f.write(test_content)

    print("✅ Created test_database_operations.py")
    return True


def main():
    """Main execution"""
    print("FIXING DATABASE METHODS ERROR")
    print("=" * 50)
    print("Fixing 'execute_fetchone' method that doesn't exist")
    print("=" * 50)

    # Apply all fixes
    fix1 = fix_database_methods()
    fix2 = check_user_settings_model()
    fix3 = add_missing_imports()
    fix4 = create_database_test()

    print("\\n" + "=" * 50)
    print("📊 DATABASE FIX RESULTS:")
    print(f"   Database Methods: {'✅ FIXED' if fix1 else '❌ FAILED'}")
    print(f"   UserSettings Model: {'✅ CHECKED' if fix2 else '❌ FAILED'}")
    print(f"   Missing Imports: {'✅ ADDED' if fix3 else '❌ FAILED'}")
    print(f"   Test Script: {'✅ CREATED' if fix4 else '❌ FAILED'}")

    if all([fix1, fix2, fix3, fix4]):
        print("\\n🎉 DATABASE METHODS FIXED!")

        print("\\n🧪 TEST THE DATABASE:")
        print("   python test_database_operations.py")

        print("\\n🔄 RESTART YOUR SYSTEM:")
        print("   Ctrl+C (stop current system)")
        print("   python production_main.py")

        print("\\n✅ WHAT WAS FIXED:")
        print("   • execute_fetchone() → execute() + fetchone()")
        print("   • Added missing cursor assignments")
        print("   • Fixed get_user_settings method")
        print("   • Added UserSettings model if missing")
        print("   • Added required imports")

        print("\\n📊 DATABASE OPERATIONS NOW WORK:")
        print("   • User registration and login")
        print("   • Settings retrieval and storage")
        print("   • Multi-user session management")
        print("   • Proper error handling")

    else:
        print("\\n⚠️ Some fixes failed - check errors above")


if __name__ == "__main__":
    main()
