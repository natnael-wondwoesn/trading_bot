#!/usr/bin/env python3
"""
VishvaAlgo Integration Validation Script
Validate that the ML strategy is properly integrated into your trading bot
"""

import asyncio
import os
import logging
from datetime import datetime
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VishvaIntegrationValidator:
    """Validate VishvaAlgo ML Strategy integration"""

    def __init__(self):
        self.checks_passed = 0
        self.total_checks = 0

    def check_dependencies(self):
        """Check if all required ML dependencies are installed"""
        print("🔍 Checking ML Dependencies")
        print("-" * 30)

        dependencies = [
            ("catboost", "CatBoost"),
            ("tensorflow", "TensorFlow"),
            ("sklearn", "Scikit-learn"),
            ("joblib", "Joblib"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy"),
        ]

        for module_name, display_name in dependencies:
            self.total_checks += 1
            try:
                __import__(module_name)
                print(f"✅ {display_name}: Installed")
                self.checks_passed += 1
            except ImportError:
                print(
                    f"❌ {display_name}: Missing - Install with: pip install {module_name}"
                )

    def check_file_structure(self):
        """Check if all required files are in place"""
        print("\n📁 Checking File Structure")
        print("-" * 30)

        required_files = [
            "strategy/strategies/vishva_ml_strategy.py",
            "config/ml_config.py",
            "ml_utils/__init__.py",
            "ml_utils/feature_engineering.py",
            "ml_utils/model_trainer.py",
            "train_vishva_models.py",
            "test_vishva_strategy.py",
        ]

        for file_path in required_files:
            self.total_checks += 1
            if os.path.exists(file_path):
                print(f"✅ {file_path}: Found")
                self.checks_passed += 1
            else:
                print(f"❌ {file_path}: Missing")

    def check_directory_structure(self):
        """Check if required directories exist"""
        print("\n📂 Checking Directory Structure")
        print("-" * 30)

        required_dirs = ["models", "models/vishva_ml", "ml_utils"]

        for dir_path in required_dirs:
            self.total_checks += 1
            if os.path.exists(dir_path):
                print(f"✅ {dir_path}/: Exists")
                self.checks_passed += 1
            else:
                print(f"⚠️ {dir_path}/: Creating...")
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"✅ {dir_path}/: Created")
                    self.checks_passed += 1
                except Exception as e:
                    print(f"❌ {dir_path}/: Failed to create - {e}")

    async def check_strategy_import(self):
        """Check if VishvaML strategy can be imported"""
        print("\n🧪 Checking Strategy Import")
        print("-" * 30)

        self.total_checks += 1
        try:
            from strategy.strategies.vishva_ml_strategy import VishvaMLStrategy

            print("✅ VishvaMLStrategy: Import successful")
            self.checks_passed += 1

            # Test strategy creation
            self.total_checks += 1
            try:
                strategy = VishvaMLStrategy(symbol="BTCUSDT")
                info = strategy.get_strategy_info()
                print(
                    f"✅ Strategy creation: Success ({info['type']} v{info['version']})"
                )
                self.checks_passed += 1
            except Exception as e:
                print(f"❌ Strategy creation: Failed - {e}")

        except ImportError as e:
            print(f"❌ VishvaMLStrategy: Import failed - {e}")

    async def check_orchestrator_integration(self):
        """Check if strategy is integrated into trading orchestrator"""
        print("\n🎭 Checking Orchestrator Integration")
        print("-" * 30)

        self.total_checks += 1
        try:
            from services.trading_orchestrator import StrategyFactory

            available_strategies = StrategyFactory.get_available_strategies()

            if "VISHVA_ML" in available_strategies:
                print("✅ StrategyFactory: VISHVA_ML found")
                print(f"   Description: {available_strategies['VISHVA_ML']}")
                self.checks_passed += 1

                # Test strategy creation through factory
                self.total_checks += 1
                try:
                    strategy = StrategyFactory.create_strategy(
                        "VISHVA_ML", {"trading_symbol": "BTCUSDT"}
                    )
                    print(f"✅ Factory creation: Success ({strategy.name})")
                    self.checks_passed += 1
                except Exception as e:
                    print(f"❌ Factory creation: Failed - {e}")
            else:
                print("❌ StrategyFactory: VISHVA_ML not found")
                print("   Available strategies:", list(available_strategies.keys()))

        except ImportError as e:
            print(f"❌ StrategyFactory import failed: {e}")

    async def check_config_integration(self):
        """Check if ML strategy is in configuration"""
        print("\n⚙️ Checking Configuration Integration")
        print("-" * 30)

        self.total_checks += 1
        try:
            from config.config import Config

            if hasattr(Config, "SUPPORTED_STRATEGIES"):
                supported = Config.SUPPORTED_STRATEGIES
                if "VISHVA_ML" in supported:
                    print("✅ Config: VISHVA_ML in SUPPORTED_STRATEGIES")
                    self.checks_passed += 1
                else:
                    print("❌ Config: VISHVA_ML not in SUPPORTED_STRATEGIES")
                    print(f"   Current strategies: {supported}")
            else:
                print("⚠️ Config: SUPPORTED_STRATEGIES not found")

        except ImportError as e:
            print(f"❌ Config import failed: {e}")

    def check_ml_config(self):
        """Check ML configuration file"""
        print("\n🔧 Checking ML Configuration")
        print("-" * 30)

        self.total_checks += 1
        try:
            from config.ml_config import ML_CONFIG

            required_keys = ["model_path", "catboost", "neural", "asset_risk_params"]
            missing_keys = [key for key in required_keys if key not in ML_CONFIG]

            if not missing_keys:
                print("✅ ML_CONFIG: All required keys present")
                print(f"   Model path: {ML_CONFIG['model_path']}")
                print(f"   Asset configs: {len(ML_CONFIG['asset_risk_params'])} assets")
                self.checks_passed += 1
            else:
                print(f"❌ ML_CONFIG: Missing keys - {missing_keys}")

        except ImportError as e:
            print(f"❌ ML_CONFIG import failed: {e}")

    def check_feature_engineering(self):
        """Check feature engineering module"""
        print("\n🔧 Checking Feature Engineering")
        print("-" * 30)

        self.total_checks += 1
        try:
            from ml_utils.feature_engineering import calculate_all_features

            print("✅ Feature engineering: Import successful")
            self.checks_passed += 1

            # Test with sample data
            self.total_checks += 1
            try:
                import pandas as pd
                import numpy as np

                # Create simple test data
                data = pd.DataFrame(
                    {
                        "open": [100, 101, 102, 103, 104],
                        "high": [101, 102, 103, 104, 105],
                        "low": [99, 100, 101, 102, 103],
                        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                        "volume": [1000, 1100, 1200, 1300, 1400],
                    }
                )

                features = calculate_all_features(data)
                if len(features) > 10:  # Should have many features
                    print(f"✅ Feature calculation: Success ({len(features)} features)")
                    self.checks_passed += 1
                else:
                    print(f"⚠️ Feature calculation: Limited features ({len(features)})")

            except Exception as e:
                print(f"❌ Feature calculation: Failed - {e}")

        except ImportError as e:
            print(f"❌ Feature engineering import: Failed - {e}")

    def check_model_trainer(self):
        """Check model trainer module"""
        print("\n🤖 Checking Model Trainer")
        print("-" * 30)

        self.total_checks += 1
        try:
            from ml_utils.model_trainer import VishvaModelTrainer, train_vishva_models

            print("✅ Model trainer: Import successful")
            self.checks_passed += 1

            # Test trainer initialization
            self.total_checks += 1
            try:
                from config.ml_config import ML_CONFIG

                trainer = VishvaModelTrainer(ML_CONFIG)
                print("✅ Trainer initialization: Success")
                self.checks_passed += 1
            except Exception as e:
                print(f"❌ Trainer initialization: Failed - {e}")

        except ImportError as e:
            print(f"❌ Model trainer import: Failed - {e}")

    def check_training_script(self):
        """Check training script functionality"""
        print("\n📊 Checking Training Script")
        print("-" * 30)

        self.total_checks += 1
        if os.path.exists("train_vishva_models.py"):
            print("✅ Training script: File exists")
            self.checks_passed += 1

            # Test script syntax
            self.total_checks += 1
            try:
                with open("train_vishva_models.py", "r") as f:
                    content = f.read()

                # Basic syntax check
                compile(content, "train_vishva_models.py", "exec")
                print("✅ Training script: Syntax valid")
                self.checks_passed += 1
            except Exception as e:
                print(f"❌ Training script: Syntax error - {e}")
        else:
            print("❌ Training script: File missing")

    def check_test_script(self):
        """Check test script functionality"""
        print("\n🧪 Checking Test Script")
        print("-" * 30)

        self.total_checks += 1
        if os.path.exists("test_vishva_strategy.py"):
            print("✅ Test script: File exists")
            self.checks_passed += 1

            # Test script syntax
            self.total_checks += 1
            try:
                with open("test_vishva_strategy.py", "r") as f:
                    content = f.read()

                # Basic syntax check
                compile(content, "test_vishva_strategy.py", "exec")
                print("✅ Test script: Syntax valid")
                self.checks_passed += 1
            except Exception as e:
                print(f"❌ Test script: Syntax error - {e}")
        else:
            print("❌ Test script: File missing")

    def generate_integration_report(self):
        """Generate final integration report"""
        print("\n" + "=" * 60)
        print("📊 INTEGRATION VALIDATION REPORT")
        print("=" * 60)

        success_rate = (
            (self.checks_passed / self.total_checks) * 100
            if self.total_checks > 0
            else 0
        )

        print(
            f"Checks Passed: {self.checks_passed}/{self.total_checks} ({success_rate:.1f}%)"
        )

        if success_rate >= 90:
            print("\n🎉 INTEGRATION STATUS: EXCELLENT")
            print("✅ VishvaAlgo ML Strategy is fully integrated!")
            print("\n📋 Ready for:")
            print("   • Model training: python train_vishva_models.py")
            print("   • Strategy testing: python test_vishva_strategy.py")
            print("   • Live trading with ML strategy")

        elif success_rate >= 70:
            print("\n⚠️ INTEGRATION STATUS: GOOD")
            print("Most components are integrated. Fix remaining issues.")

        elif success_rate >= 50:
            print("\n🔶 INTEGRATION STATUS: PARTIAL")
            print("Significant issues found. Review failed checks.")

        else:
            print("\n❌ INTEGRATION STATUS: INCOMPLETE")
            print("Major integration issues. Follow integration guide.")

        print("\n📖 For help:")
        print("   • Review integration guide")
        print("   • Check error messages for specific issues")
        print("   • Ensure all dependencies are installed")

        # Additional recommendations
        if success_rate >= 80:
            print("\n🚀 Next Steps:")
            print("   1. python train_vishva_models.py BTCUSDT  # Train for Bitcoin")
            print("   2. python test_vishva_strategy.py BTCUSDT  # Test the strategy")
            print("   3. Add VISHVA_ML to your user settings")
            print("   4. Start with paper trading")
            print("   5. Monitor performance and retrain weekly")

    async def run_validation(self):
        """Run complete integration validation"""
        print("🧠 VISHVAALGO ML STRATEGY - INTEGRATION VALIDATION")
        print("=" * 65)
        print(f"⏰ Validation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run all validation checks
        self.check_dependencies()
        self.check_file_structure()
        self.check_directory_structure()
        await self.check_strategy_import()
        await self.check_orchestrator_integration()
        await self.check_config_integration()
        self.check_ml_config()
        self.check_feature_engineering()
        self.check_model_trainer()
        self.check_training_script()
        self.check_test_script()

        # Generate final report
        self.generate_integration_report()


async def quick_validation():
    """Run a quick validation for essential components"""
    print("🧠 VISHVAALGO ML STRATEGY - QUICK VALIDATION")
    print("=" * 55)

    essential_checks = [
        (
            "Strategy Import",
            lambda: __import__("strategy.strategies.vishva_ml_strategy"),
        ),
        ("ML Config", lambda: __import__("config.ml_config")),
        ("Feature Engineering", lambda: __import__("ml_utils.feature_engineering")),
        ("Model Trainer", lambda: __import__("ml_utils.model_trainer")),
        (
            "Orchestrator Integration",
            lambda: __import__("services.trading_orchestrator"),
        ),
    ]

    passed = 0
    total = len(essential_checks)

    for check_name, check_func in essential_checks:
        try:
            check_func()
            print(f"✅ {check_name}: OK")
            passed += 1
        except Exception as e:
            print(f"❌ {check_name}: Failed - {e}")

    success_rate = (passed / total) * 100
    print(f"\n📊 Quick validation: {passed}/{total} ({success_rate:.1f}%)")

    if success_rate == 100:
        print("🎉 All essential components are working!")
        print(
            "Run: python validate_vishva_integration.py --full for detailed validation"
        )
    else:
        print("⚠️ Some essential components have issues")
        print("Run: python validate_vishva_integration.py --full for detailed analysis")


async def main():
    """Main validation execution"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        await quick_validation()
    else:
        validator = VishvaIntegrationValidator()
        await validator.run_validation()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Validation stopped by user")
    except Exception as e:
        logger.error(f"❌ Validation script error: {e}")
        print(f"\n❌ Error: {e}")
        print("Check the logs above for details")
