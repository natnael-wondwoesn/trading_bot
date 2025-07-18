#!/usr/bin/env python3
"""
Model Training Utilities for VishvaAlgo ML Strategy
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from datetime import datetime
import os
import pickle
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)

# Try to import CatBoost and TensorFlow
CATBOOST_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
    logger.info("CatBoost is available")
except ImportError:
    logger.warning("CatBoost not available, will use RandomForest instead")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError:
    logger.warning("TensorFlow not available, neural network training disabled")


class VishvaModelTrainer:
    """Advanced model trainer for VishvaAlgo ML Strategy"""

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = MinMaxScaler()

    def prepare_training_data(
        self, data: pd.DataFrame, symbol: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with comprehensive feature engineering"""
        from ml_utils.feature_engineering import calculate_all_features

        logger.info(f"Preparing training data for {symbol}...")

        # Calculate features for all historical periods
        lookback = self.config["feature_lookback"]
        all_features = []

        for i in range(lookback, len(data) - 2):
            hist_data = data.iloc[i - lookback : i + 1]
            try:
                features = calculate_all_features(hist_data)
                feature_vector = self._dict_to_vector(features)
                all_features.append(feature_vector)
            except Exception as e:
                logger.warning(f"Error calculating features at index {i}: {e}")
                continue

        if len(all_features) < self.config["min_training_samples"]:
            raise ValueError(
                f"Insufficient training samples: {len(all_features)} < {self.config['min_training_samples']}"
            )

        X = np.array(all_features)

        # Create target variables
        y = self._create_targets(data, symbol, lookback)

        # Ensure X and y have same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        logger.info(
            f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features"
        )
        return X, y

    def _dict_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dictionary to numpy vector"""
        # Remove non-numeric features
        numeric_features = []
        for key in sorted(features.keys()):
            if key != "current_price" and isinstance(
                features[key], (int, float, np.number)
            ):
                value = features[key]
                if pd.isna(value) or np.isinf(value):
                    value = 0.0
                numeric_features.append(float(value))

        return np.array(numeric_features)

    def _create_targets(
        self, data: pd.DataFrame, symbol: str, lookback: int
    ) -> np.ndarray:
        """Create 3-class targets based on future price movements"""
        close = data["close"].iloc[lookback:]

        # Get risk parameters for this symbol
        risk_params = self.config["asset_risk_params"].get(
            symbol, self.config["default_risk_params"]
        )

        # Calculate future returns (2 periods ahead)
        future_returns = close.shift(-2) / close - 1

        targets = []
        stop_loss_pct = risk_params["stop_loss"]
        take_profit_pct = risk_params["take_profit"]

        for ret in future_returns:
            if pd.isna(ret):
                targets.append(0)  # Neutral
            elif ret > take_profit_pct:
                targets.append(1)  # Long
            elif ret < -stop_loss_pct:
                targets.append(2)  # Short
            else:
                targets.append(0)  # Neutral

        return np.array(targets[:-2])  # Remove last 2 periods without future data

    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> VotingClassifier:
        """Train ensemble model (CatBoost + Random Forest + Gradient Boosting)"""
        logger.info("Training ensemble model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split (chronological)
        split_idx = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create base models
        models = []

        # CatBoost (if available)
        if CATBOOST_AVAILABLE:
            catboost_model = CatBoostClassifier(
                iterations=self.config["catboost"]["iterations"],
                learning_rate=self.config["catboost"]["learning_rate"],
                depth=self.config["catboost"]["depth"],
                class_weights=self.config["catboost"]["class_weights"],
                random_seed=42,
                verbose=False,
            )
            models.append(("catboost", catboost_model))

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight={0: 1, 1: 2, 2: 2},
        )
        models.append(("random_forest", rf_model))

        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        models.append(("gradient_boost", gb_model))

        # Create ensemble
        ensemble = VotingClassifier(estimators=models, voting="soft")

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Evaluate
        if len(X_test) > 0:
            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Ensemble accuracy: {accuracy:.3f}")

            # Detailed classification report
            try:
                report = classification_report(
                    y_test, y_pred, target_names=["Neutral", "Long", "Short"]
                )
                logger.info(f"Classification Report:\n{report}")
            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")

        return ensemble

    def train_neural_network(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM neural network (if TensorFlow is available)"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network training")
            return None

        logger.info("Training LSTM neural network...")

        # Scale features if not already scaled
        if not hasattr(self.scaler, "scale_"):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Train-test split
        split_idx = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Reshape for LSTM
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Create model
        model = Sequential(
            [
                LSTM(
                    self.config["neural"]["lstm_units"][0],
                    return_sequences=True,
                    input_shape=(1, X_scaled.shape[1]),
                ),
                BatchNormalization(),
                LSTM(self.config["neural"]["lstm_units"][1], return_sequences=False),
                Dropout(self.config["neural"]["dropout_rate"]),
                Dense(16, activation="relu"),
                Dense(3, activation="softmax"),  # 3 classes
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=self.config["neural"]["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train with early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        try:
            history = model.fit(
                X_train_lstm,
                y_train,
                epochs=self.config["neural"]["epochs"],
                batch_size=self.config["neural"]["batch_size"],
                validation_data=(X_test_lstm, y_test),
                callbacks=[early_stopping],
                verbose=0,
            )

            # Evaluate
            if len(X_test) > 0:
                y_pred = np.argmax(model.predict(X_test_lstm, verbose=0), axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Neural network accuracy: {accuracy:.3f}")

        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            return None

        return model


def train_vishva_models(data: pd.DataFrame, symbol: str, config: Dict) -> Dict:
    """Main function to train all VishvaAlgo models"""
    trainer = VishvaModelTrainer(config)

    try:
        # Prepare training data
        X, y = trainer.prepare_training_data(data, symbol)

        # Train ensemble model
        ensemble_model = trainer.train_ensemble_model(X, y)

        # Train neural network (if available)
        neural_model = None
        if TENSORFLOW_AVAILABLE:
            neural_model = trainer.train_neural_network(X, y)

        # Return trained components
        return {
            "ensemble_model": ensemble_model,
            "neural_model": neural_model,
            "scaler": trainer.scaler,
            "training_samples": len(X),
            "feature_count": X.shape[1],
            "class_distribution": np.bincount(y),
            "training_date": datetime.now(),
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "catboost_available": CATBOOST_AVAILABLE,
        }

    except Exception as e:
        logger.error(f"Error training models for {symbol}: {e}")
        raise


def save_models(models: Dict, symbol: str, model_path: str):
    """Save trained models to disk"""
    try:
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Save ensemble model
        if models["ensemble_model"] is not None:
            ensemble_path = os.path.join(model_path, f"ensemble_{symbol}.pkl")
            joblib.dump(models["ensemble_model"], ensemble_path)
            logger.info(f"Saved ensemble model to {ensemble_path}")

        # Save neural network model
        if models["neural_model"] is not None and TENSORFLOW_AVAILABLE:
            neural_path = os.path.join(model_path, f"neural_{symbol}.h5")
            models["neural_model"].save(neural_path)
            logger.info(f"Saved neural network to {neural_path}")

        # Save scaler
        scaler_path = os.path.join(model_path, f"scaler_{symbol}.pkl")
        joblib.dump(models["scaler"], scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            "training_samples": models["training_samples"],
            "feature_count": models["feature_count"],
            "class_distribution": models["class_distribution"].tolist(),
            "training_date": models["training_date"].isoformat(),
            "tensorflow_available": models["tensorflow_available"],
            "catboost_available": models["catboost_available"],
        }

        metadata_path = os.path.join(model_path, f"metadata_{symbol}.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"Error saving models for {symbol}: {e}")
        raise


def load_models(symbol: str, model_path: str) -> Dict:
    """Load trained models from disk"""
    try:
        models = {}

        # Load ensemble model
        ensemble_path = os.path.join(model_path, f"ensemble_{symbol}.pkl")
        if os.path.exists(ensemble_path):
            models["ensemble_model"] = joblib.load(ensemble_path)
            logger.info(f"Loaded ensemble model from {ensemble_path}")
        else:
            models["ensemble_model"] = None

        # Load neural network model
        neural_path = os.path.join(model_path, f"neural_{symbol}.h5")
        if os.path.exists(neural_path) and TENSORFLOW_AVAILABLE:
            try:
                models["neural_model"] = tf.keras.models.load_model(neural_path)
                logger.info(f"Loaded neural network from {neural_path}")
            except Exception as e:
                logger.warning(f"Could not load neural network: {e}")
                models["neural_model"] = None
        else:
            models["neural_model"] = None

        # Load scaler
        scaler_path = os.path.join(model_path, f"scaler_{symbol}.pkl")
        if os.path.exists(scaler_path):
            models["scaler"] = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            models["scaler"] = MinMaxScaler()

        # Load metadata
        metadata_path = os.path.join(model_path, f"metadata_{symbol}.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            models.update(metadata)
            logger.info(f"Loaded metadata from {metadata_path}")

        return models

    except Exception as e:
        logger.error(f"Error loading models for {symbol}: {e}")
        return {
            "ensemble_model": None,
            "neural_model": None,
            "scaler": MinMaxScaler(),
            "training_samples": 0,
            "feature_count": 0,
        }
