#!/usr/bin/env python3
"""
ML Configuration for VishvaAlgo Strategy
"""

# Model Configuration
ML_CONFIG = {
    "model_path": "models/vishva_ml",
    "retrain_interval_days": 7,
    "min_training_samples": 1000,
    "confidence_threshold": 0.6,
    "feature_lookback": 200,
    # CatBoost Configuration
    "catboost": {
        "iterations": 800,
        "learning_rate": 0.2,
        "depth": 6,
        "class_weights": [1, 2, 2],  # [Neutral, Long, Short]
    },
    # Neural Network Configuration
    "neural": {
        "lstm_units": [64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
    },
    # Risk Management per Asset
    "asset_risk_params": {
        "BTCUSDT": {"stop_loss": 0.08, "take_profit": 0.25, "leverage": 3},
        "ETHUSDT": {"stop_loss": 0.10, "take_profit": 0.30, "leverage": 3},
        "ADAUSDT": {"stop_loss": 0.12, "take_profit": 0.35, "leverage": 2},
        "SOLUSDT": {"stop_loss": 0.15, "take_profit": 0.40, "leverage": 2},
        "DOGEUSDT": {"stop_loss": 0.18, "take_profit": 0.45, "leverage": 2},
        "XRPUSDT": {"stop_loss": 0.12, "take_profit": 0.35, "leverage": 2},
        "LINKUSDT": {"stop_loss": 0.15, "take_profit": 0.40, "leverage": 2},
        "AVAXUSDT": {"stop_loss": 0.15, "take_profit": 0.40, "leverage": 2},
    },
    # Default risk params for unlisted assets
    "default_risk_params": {"stop_loss": 0.10, "take_profit": 0.30, "leverage": 2},
}
