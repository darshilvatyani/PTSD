"""
XGBoost Risk Prediction Model for PTSD Trigger Detection System

Trains an XGBoost model to predict PTSD episode risk
from multi-modal feature inputs (emotion, object, audio, stress scores).

This is an optional enhancement over the weighted fusion in engine.py.
It learns patterns from historical trigger data for smarter predictions.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("RiskModel")

MODEL_PATH = os.path.join(MODELS_DIR, "risk_predictor.pkl")


class RiskPredictor:
    """
    XGBoost model that predicts PTSD episode risk from module scores.
    
    Features:
        - emotion_score (0-100)
        - object_score (0-100)
        - audio_score (0-100)
        - stress_score (0-100)
    
    Output:
        - risk_score (0-100)
        - risk_level (LOW / MEDIUM / HIGH)
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        self._load_model()

    def generate_training_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic training data for the risk predictor.
        Simulates realistic multi-modal trigger scenarios.
        """
        import random
        random.seed(42)
        np.random.seed(42)

        data = []
        for _ in range(n_samples):
            scenario = random.choice([
                "safe", "safe", "safe",  # 3x more safe scenarios
                "mild_single", "mild_multi",
                "high_single", "high_multi", "panic"
            ])

            if scenario == "safe":
                emotion = random.uniform(0, 20)
                obj = random.uniform(0, 15)
                audio = random.uniform(0, 15)
                stress = random.uniform(0, 25)
                risk = random.uniform(0, 20)

            elif scenario == "mild_single":
                # One module slightly elevated
                base = [random.uniform(0, 15) for _ in range(4)]
                idx = random.randint(0, 3)
                base[idx] = random.uniform(30, 55)
                emotion, obj, audio, stress = base
                risk = random.uniform(25, 45)

            elif scenario == "mild_multi":
                # Two modules elevated
                emotion = random.uniform(25, 50)
                obj = random.uniform(20, 45)
                audio = random.uniform(0, 20)
                stress = random.uniform(30, 55)
                risk = random.uniform(40, 65)

            elif scenario == "high_single":
                # One module very high
                base = [random.uniform(0, 20) for _ in range(4)]
                idx = random.randint(0, 3)
                base[idx] = random.uniform(65, 95)
                emotion, obj, audio, stress = base
                risk = random.uniform(50, 75)

            elif scenario == "high_multi":
                # Multiple modules high
                emotion = random.uniform(50, 85)
                obj = random.uniform(40, 70)
                audio = random.uniform(45, 80)
                stress = random.uniform(55, 85)
                risk = random.uniform(70, 90)

            else:  # panic
                emotion = random.uniform(70, 100)
                obj = random.uniform(60, 100)
                audio = random.uniform(65, 100)
                stress = random.uniform(75, 100)
                risk = random.uniform(85, 100)

            # Add noise
            emotion = max(0, min(100, emotion + random.gauss(0, 3)))
            obj = max(0, min(100, obj + random.gauss(0, 3)))
            audio = max(0, min(100, audio + random.gauss(0, 3)))
            stress = max(0, min(100, stress + random.gauss(0, 3)))
            risk = max(0, min(100, risk + random.gauss(0, 2)))

            data.append({
                "emotion_score": round(emotion, 1),
                "object_score": round(obj, 1),
                "audio_score": round(audio, 1),
                "stress_score": round(stress, 1),
                "risk_score": round(risk, 1),
            })

        return pd.DataFrame(data)

    def train(self, data: pd.DataFrame = None):
        """Train the XGBoost risk predictor."""
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        if data is None:
            logger.info("Generating 5000 training samples...")
            data = self.generate_training_data(5000)

        features = ["emotion_score", "object_score", "audio_score", "stress_score"]
        X = data[features].values
        y = data["risk_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("Training XGBoost risk predictor...")
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Training complete! MAE: {mae:.2f}, R²: {r2:.4f}")

        # Save
        self._save_model()

        return {"mae": round(mae, 2), "r2": round(r2, 4)}

    def predict(self, emotion_score: float, object_score: float,
                audio_score: float, stress_score: float) -> dict:
        """
        Predict risk from module scores.

        Returns:
            {"risk_score": float, "risk_level": str}
        """
        if not self.is_trained:
            self.train()

        features = np.array([[emotion_score, object_score, audio_score, stress_score]])
        risk = float(self.model.predict(features)[0])
        risk = max(0, min(100, risk))

        if risk < 30:
            level = "LOW"
        elif risk < 60:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {"risk_score": round(risk, 1), "risk_level": level}

    def _save_model(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Risk model saved to {MODEL_PATH}")

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Loaded pre-trained risk model from {MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load risk model: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  PTSD - XGBoost Risk Predictor Training")
    print("=" * 60)

    predictor = RiskPredictor()
    results = predictor.train()
    print(f"\nResults: MAE = {results['mae']}, R² = {results['r2']}")

    # Test predictions
    print("\nTest Predictions:")
    tests = [
        (5, 0, 0, 10, "All calm"),
        (60, 10, 5, 30, "Emotion elevated"),
        (40, 50, 60, 70, "Multi-modal stress"),
        (90, 80, 85, 95, "Full panic"),
    ]
    for emo, obj, aud, stress, label in tests:
        r = predictor.predict(emo, obj, aud, stress)
        print(f"  {label:25s} → Risk: {r['risk_score']:5.1f}% [{r['risk_level']}]")
