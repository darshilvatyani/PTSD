"""
Dummy Physiological Data Generator for PTSD Trigger Detection System

Simulates realistic heart rate (HR) and Galvanic Skin Response (GSR) data
for testing the stress classifier before real hardware is connected.

Simulates 3 states:
  - CALM: Normal resting state (HR ~60-80 bpm, GSR ~1-5 µS)
  - MILD_STRESS: Slight anxiety (HR ~85-110 bpm, GSR ~5-12 µS)
  - HIGH_STRESS: Panic / PTSD trigger response (HR ~110-160 bpm, GSR ~12-25 µS)

When hardware is ready, swap this for serial_reader.py via config.py
"""

import sys
import os
import time
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import NORMAL_HR_MIN, NORMAL_HR_MAX, STRESS_HR_THRESHOLD, STRESS_HR_CRITICAL
from src.utils.logger import setup_logger

logger = setup_logger("DummyData")


# ============================================================
# Physiological parameter ranges for each stress state
# ============================================================
STRESS_PROFILES = {
    "calm": {
        "heart_rate": (60, 80),       # bpm
        "gsr": (1.0, 5.0),           # µS (microsiemens)
        "hrv": (40, 80),             # ms (heart rate variability — higher = calmer)
        "skin_temp": (33.0, 35.0),   # °C
    },
    "mild_stress": {
        "heart_rate": (85, 110),
        "gsr": (5.0, 12.0),
        "hrv": (20, 40),
        "skin_temp": (31.0, 33.0),
    },
    "high_stress": {
        "heart_rate": (110, 160),
        "gsr": (12.0, 25.0),
        "hrv": (5, 20),
        "skin_temp": (28.0, 31.0),
    },
}


def generate_single_reading(state: str = "calm") -> dict:
    """
    Generate a single physiological reading for a given stress state.

    Args:
        state: One of "calm", "mild_stress", "high_stress"

    Returns:
        dict with heart_rate, gsr, hrv, skin_temp, and stress_label
    """
    profile = STRESS_PROFILES.get(state, STRESS_PROFILES["calm"])

    reading = {
        "heart_rate": round(random.uniform(*profile["heart_rate"]), 1),
        "gsr": round(random.uniform(*profile["gsr"]), 2),
        "hrv": round(random.uniform(*profile["hrv"]), 1),
        "skin_temp": round(random.uniform(*profile["skin_temp"]), 1),
        "stress_label": state,
    }
    return reading


def generate_dataset(n_samples: int = 3000, save_path: str = None) -> pd.DataFrame:
    """
    Generate a full training dataset with balanced stress states.

    Args:
        n_samples: Total number of samples (split evenly across 3 states)
        save_path: Optional CSV file path to save the dataset

    Returns:
        pandas DataFrame with columns: heart_rate, gsr, hrv, skin_temp, stress_label
    """
    samples_per_state = n_samples // 3
    data = []

    for state in ["calm", "mild_stress", "high_stress"]:
        for _ in range(samples_per_state):
            reading = generate_single_reading(state)
            # Add some realistic noise
            reading["heart_rate"] += random.gauss(0, 3)
            reading["gsr"] += random.gauss(0, 0.5)
            reading["hrv"] += random.gauss(0, 3)
            reading["skin_temp"] += random.gauss(0, 0.3)
            # Clamp to positive values
            reading["heart_rate"] = max(40, round(reading["heart_rate"], 1))
            reading["gsr"] = max(0.1, round(reading["gsr"], 2))
            reading["hrv"] = max(1, round(reading["hrv"], 1))
            reading["skin_temp"] = max(25, round(reading["skin_temp"], 1))
            data.append(reading)

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Dataset saved to {save_path} ({len(df)} samples)")

    logger.info(f"Generated {len(df)} samples:")
    logger.info(f"  Calm: {len(df[df['stress_label'] == 'calm'])}")
    logger.info(f"  Mild stress: {len(df[df['stress_label'] == 'mild_stress'])}")
    logger.info(f"  High stress: {len(df[df['stress_label'] == 'high_stress'])}")

    return df


def generate_realtime_stream(state: str = "calm", interval: float = 1.0):
    """
    Generator that yields physiological readings in real-time.
    Simulates a live sensor stream.

    Args:
        state: Stress state to simulate
        interval: Seconds between readings

    Yields:
        dict with heart_rate, gsr, hrv, skin_temp, timestamp
    """
    while True:
        reading = generate_single_reading(state)
        reading["timestamp"] = time.time()
        yield reading
        time.sleep(interval)


class DummySensorStream:
    """
    Simulates a live sensor stream that can change states dynamically.
    Used by the fusion engine to get current physiological data.
    """

    def __init__(self):
        self.current_state = "calm"
        self._transition_speed = 0.1  # How fast readings change between states
        self._current_hr = 70.0
        self._current_gsr = 3.0
        self._current_hrv = 60.0
        self._current_temp = 34.0
        logger.info("DummySensorStream initialized (state: calm)")

    def set_state(self, state: str):
        """Change the simulated stress state."""
        if state in STRESS_PROFILES:
            self.current_state = state
            logger.info(f"Sensor state changed to: {state}")

    def get_reading(self) -> dict:
        """
        Get the current sensor reading.
        Values smoothly transition toward the target state.
        """
        target = STRESS_PROFILES[self.current_state]

        # Smoothly move current values toward target range
        target_hr = random.uniform(*target["heart_rate"])
        target_gsr = random.uniform(*target["gsr"])
        target_hrv = random.uniform(*target["hrv"])
        target_temp = random.uniform(*target["skin_temp"])

        # Smooth transition (weighted average with current)
        speed = self._transition_speed
        self._current_hr = self._current_hr * (1 - speed) + target_hr * speed
        self._current_gsr = self._current_gsr * (1 - speed) + target_gsr * speed
        self._current_hrv = self._current_hrv * (1 - speed) + target_hrv * speed
        self._current_temp = self._current_temp * (1 - speed) + target_temp * speed

        # Add slight noise
        return {
            "heart_rate": round(self._current_hr + random.gauss(0, 1), 1),
            "gsr": round(max(0.1, self._current_gsr + random.gauss(0, 0.2)), 2),
            "hrv": round(max(1, self._current_hrv + random.gauss(0, 1)), 1),
            "skin_temp": round(self._current_temp + random.gauss(0, 0.1), 1),
            "state": self.current_state,
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("  PTSD - Dummy Physiological Data Generator")
    print("=" * 60)

    # Generate and save training dataset
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(project_root, "data", "stress_dataset.csv")

    print(f"\nGenerating 3000-sample training dataset...")
    df = generate_dataset(n_samples=3000, save_path=save_path)

    print(f"\nDataset saved to: {save_path}")
    print(f"\nSample data:")
    print(df.head(10).to_string(index=False))
    print(f"\nStats per class:")
    print(df.groupby("stress_label").describe().round(1))

    # Demo live stream
    print(f"\n{'='*60}")
    print("  Live Sensor Stream Demo (changes state every 5 readings)")
    print("  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    stream = DummySensorStream()
    states = ["calm", "mild_stress", "high_stress"]
    count = 0

    try:
        while True:
            # Cycle through states
            if count % 5 == 0:
                state = states[(count // 5) % 3]
                stream.set_state(state)

            reading = stream.get_reading()
            print(f"  HR: {reading['heart_rate']:6.1f} bpm | "
                  f"GSR: {reading['gsr']:5.2f} µS | "
                  f"HRV: {reading['hrv']:5.1f} ms | "
                  f"Temp: {reading['skin_temp']:5.1f}°C | "
                  f"State: {reading['state']}")

            count += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStream stopped.")
