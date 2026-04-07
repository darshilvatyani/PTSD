"""
Stress Classifier Module for PTSD Trigger Detection System

Trains a Random Forest classifier on simulated physiological data
(heart rate, GSR, HRV, skin temperature) to predict stress levels.

Model: scikit-learn Random Forest
Classes: calm, mild_stress, high_stress
Features: heart_rate, gsr, hrv, skin_temp
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import HEART_RATE_SOURCE, SENSOR_SOURCES, MODELS_DIR
from src.utils.logger import setup_logger

logger = setup_logger("StressClassifier")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "stress_classifier.pkl")
FEATURE_COLUMNS = ["heart_rate", "gsr", "hrv", "skin_temp"]
LABEL_COLUMN = "stress_label"

# Stress label → numeric risk score
STRESS_SCORES = {
    "calm": 10.0,
    "mild_stress": 55.0,
    "high_stress": 90.0,
}


class StressClassifier:
    """
    Predicts stress level from physiological signals.
    Uses Random Forest trained on dummy/real sensor data.
    Supports per-sensor source selection (DUMMY/SERIAL/AUTO/NEUTRAL).
    """

    def __init__(self):
        """Initialize the stress classifier."""
        self.model = None
        self.is_trained = False
        self.sensor_stream = None

        # Set up data source based on per-sensor config
        self._setup_data_source()

        # Try to load pre-trained model
        self._load_model()

        logger.info(f"StressClassifier initialized (mode: {HEART_RATE_SOURCE})")
        logger.info(f"  Sensor sources: {SENSOR_SOURCES}")

    def _setup_data_source(self):
        """
        Set up the sensor data source.
        Uses hybrid SerialSensorStream if ANY sensor is SERIAL,
        otherwise uses DummySensorStream.
        """
        has_serial = any(v == "SERIAL" for v in SENSOR_SOURCES.values())

        if has_serial:
            from src.stress.serial_reader import SerialSensorStream
            self.sensor_stream = SerialSensorStream()
            logger.info("Using HYBRID sensor stream (mix of real + simulated)")
        else:
            from src.stress.dummy_data import DummySensorStream
            self.sensor_stream = DummySensorStream()
            logger.info("Using DUMMY sensor data (all simulated)")

    def train(self, csv_path: str = None, n_samples: int = 3000):
        """
        Train the stress classifier.

        Args:
            csv_path: Path to CSV dataset. If None, generates dummy data.
            n_samples: Number of samples to generate if no CSV provided.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score

        # Load or generate data
        if csv_path and os.path.exists(csv_path):
            logger.info(f"Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            logger.info(f"Generating {n_samples} training samples...")
            from src.stress.dummy_data import generate_dataset
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(project_root, "data", "stress_dataset.csv")
            df = generate_dataset(n_samples=n_samples, save_path=csv_path)

        # Prepare features and labels
        X = df[FEATURE_COLUMNS].values
        y = df[LABEL_COLUMN].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest
        logger.info("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Training complete! Accuracy: {accuracy:.2%}")
        logger.info(f"\nClassification Report:\n{report}")

        # Feature importance
        importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
        logger.info(f"Feature importances: {importances}")

        # Save model
        self._save_model()

        return {
            "accuracy": round(accuracy, 4),
            "report": report,
            "feature_importances": importances,
        }

    def predict(self, reading: dict = None) -> dict:
        """
        Predict stress level from a sensor reading.

        Args:
            reading: dict with heart_rate, gsr, hrv, skin_temp.
                     If None, gets reading from sensor stream.

        Returns:
            dict with:
            {
                "stress_level": str ("calm" / "mild_stress" / "high_stress"),
                "confidence": float (0-1),
                "probabilities": {"calm": float, "mild_stress": float, "high_stress": float},
                "trigger_score": float (0-100),
                "reading": dict (raw sensor values)
            }
        """
        if not self.is_trained:
            # Auto-train if no model loaded
            logger.info("No trained model found. Training now...")
            self.train()

        # Get reading from sensor if none provided
        if reading is None:
            reading = self.sensor_stream.get_reading()

        # Extract features
        features = np.array([[
            reading.get("heart_rate", 70),
            reading.get("gsr", 3.0),
            reading.get("hrv", 50),
            reading.get("skin_temp", 34.0),
        ]])

        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(max(probabilities))

        # Map to class probabilities
        prob_dict = {}
        for cls, prob in zip(self.model.classes_, probabilities):
            prob_dict[cls] = round(float(prob), 3)

        # Calculate trigger score
        trigger_score = STRESS_SCORES.get(prediction, 50.0)
        # Adjust by confidence
        trigger_score = trigger_score * confidence

        return {
            "stress_level": prediction,
            "confidence": round(confidence, 3),
            "probabilities": prob_dict,
            "trigger_score": round(trigger_score, 1),
            "reading": reading,
        }

    def set_sensor_state(self, state: str):
        """Change the dummy sensor state (for testing)."""
        if self.sensor_stream and hasattr(self.sensor_stream, "set_state"):
            self.sensor_stream.set_state(state)

    def _save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    def _load_model(self):
        """Load pre-trained model from disk."""
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                with open(MODEL_SAVE_PATH, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Loaded pre-trained model from {MODEL_SAVE_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")


def run_stress_demo():
    """
    Demo: Train the classifier and run live predictions.
    Press Ctrl+C to stop.
    """
    import time

    print("=" * 60)
    print("  PTSD - Stress Level Detection Demo")
    print("  Using Random Forest on simulated sensor data")
    print("=" * 60)

    classifier = StressClassifier()

    # Train if needed
    if not classifier.is_trained:
        print("\n[Step 1] Training classifier on 3000 samples...")
        results = classifier.train()
        print(f"\nAccuracy: {results['accuracy']:.2%}")
        print(f"\n{results['report']}")
    else:
        print("\nPre-trained model loaded!")

    # Live predictions
    print(f"\n{'='*60}")
    print("  Live Stress Predictions (state changes every 8 readings)")
    print("  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    states = ["calm", "mild_stress", "high_stress"]
    count = 0

    try:
        while True:
            # Cycle through states
            if count % 8 == 0:
                state = states[(count // 8) % 3]
                classifier.set_sensor_state(state)
                print(f"\n  --- Simulating: {state.upper()} ---")

            result = classifier.predict()
            reading = result["reading"]

            # Status indicator
            level = result["stress_level"]
            if level == "calm":
                indicator = "🟢"
            elif level == "mild_stress":
                indicator = "🟡"
            else:
                indicator = "🔴"

            print(f"  {indicator} HR: {reading['heart_rate']:6.1f} | "
                  f"GSR: {reading['gsr']:5.2f} | "
                  f"HRV: {reading['hrv']:5.1f} | "
                  f"Predicted: {level:>14s} ({result['confidence']:.0%}) | "
                  f"Risk: {result['trigger_score']:.0f}%")

            count += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")


if __name__ == "__main__":
    run_stress_demo()
