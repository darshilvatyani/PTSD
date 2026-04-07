"""
Audio Classification Module for PTSD Trigger Detection System
Uses Google YAMNet (pre-trained on AudioSet) for real-time sound classification.

YAMNet: Yet Another Mel-Spectrogram Network
- 521 audio event classes (gunshots, sirens, screaming, etc.)
- Pre-trained on AudioSet-YouTube corpus
- Uses MobileNetV1 architecture
- Reference: https://tfhub.dev/google/yamnet/1
"""

import sys
import os
import time
import threading
import queue
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import (
    YAMNET_CONFIDENCE_THRESHOLD,
    AUDIO_SAMPLE_RATE,
    AUDIO_DURATION,
)
from src.utils.logger import setup_logger

logger = setup_logger("AudioClassifier")

# Path to trigger sounds config
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
TRIGGER_SOUNDS_PATH = os.path.join(CONFIG_DIR, "trigger_sounds.yaml")


class AudioClassifier:
    """
    Real-time audio trigger classifier using Google YAMNet.

    Listens to microphone input and classifies sounds into 521 categories,
    then filters through PTSD trigger configuration to calculate risk score.
    """

    def __init__(self):
        """Initialize the audio classifier."""
        self.model = None
        self.class_names = None
        self.confidence_threshold = YAMNET_CONFIDENCE_THRESHOLD
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.duration = AUDIO_DURATION

        # Load trigger config
        self.trigger_config = self._load_trigger_config()

        # Build lookup dicts
        self.trigger_lookup = {}
        for sound in self.trigger_config.get("trigger_sounds", []):
            self.trigger_lookup[sound["name"].lower()] = {
                "risk_weight": sound["risk_weight"],
                "category": sound.get("category", "unknown"),
            }

        self.safe_lookup = {}
        for sound in self.trigger_config.get("safe_sounds", []):
            self.safe_lookup[sound["name"].lower()] = sound["calm_weight"]

        # Audio queue for threaded recording
        self.audio_queue = queue.Queue()
        self.is_running = False

        logger.info("AudioClassifier initialized")
        logger.info(f"  Trigger sounds: {len(self.trigger_lookup)} configured")
        logger.info(f"  Safe sounds: {len(self.safe_lookup)} configured")
        logger.info(f"  Sample rate: {self.sample_rate} Hz")

    def _load_trigger_config(self) -> dict:
        """Load trigger sounds configuration from YAML."""
        try:
            with open(TRIGGER_SOUNDS_PATH, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded trigger sounds config from {TRIGGER_SOUNDS_PATH}")
            return config
        except FileNotFoundError:
            logger.warning(f"Trigger sounds config not found, using defaults")
            return {"trigger_sounds": [], "safe_sounds": []}

    def _load_model(self):
        """Lazy-load YAMNet model from TensorFlow Hub."""
        if self.model is None:
            logger.info("Loading YAMNet model from TensorFlow Hub (first time downloads ~200MB)...")
            import tensorflow_hub as hub

            yamnet_url = "https://tfhub.dev/google/yamnet/1"

            try:
                self.model = hub.load(yamnet_url)
            except (ValueError, OSError) as e:
                # Corrupted cache — clear and retry
                logger.warning(f"Model cache corrupted: {e}")
                logger.info("Clearing TF Hub cache and re-downloading...")
                import shutil
                import tempfile
                cache_dir = os.path.join(tempfile.gettempdir(), "tfhub_modules")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    logger.info(f"Cleared cache: {cache_dir}")
                self.model = hub.load(yamnet_url)

            # Load class names
            import csv
            class_map_path = self.model.class_map_path().numpy().decode("utf-8")
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self.class_names = [row["display_name"] for row in reader]

            logger.info(f"YAMNet loaded! {len(self.class_names)} audio classes available.")
        return self.model

    def classify_audio(self, audio_data: np.ndarray) -> dict:
        """
        Classify an audio chunk.

        Args:
            audio_data: 1D numpy array of audio samples (float32, 16kHz mono)

        Returns:
            dict with:
            {
                "top_sounds": [{"name": str, "confidence": float}],
                "trigger_sounds": [{"name": str, "category": str, "risk_weight": float, "confidence": float}],
                "safe_sounds": [{"name": str, "calm_weight": float}],
                "trigger_score": float  # 0-100
            }
        """
        model = self._load_model()

        # Ensure audio is float32 and mono
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Run YAMNet inference
        scores, embeddings, spectrogram = model(audio_data)
        scores = scores.numpy()

        # Average scores across time frames
        mean_scores = scores.mean(axis=0)

        # Get top predictions
        top_indices = np.argsort(mean_scores)[::-1][:10]
        top_sounds = []
        trigger_sounds = []
        safe_sounds = []

        for idx in top_indices:
            name = self.class_names[idx]
            confidence = float(mean_scores[idx])

            if confidence < self.confidence_threshold:
                continue

            top_sounds.append({"name": name, "confidence": round(confidence, 3)})

            # Check if trigger
            name_lower = name.lower()
            if name_lower in self.trigger_lookup:
                trigger_sounds.append({
                    "name": name,
                    "category": self.trigger_lookup[name_lower]["category"],
                    "risk_weight": self.trigger_lookup[name_lower]["risk_weight"],
                    "confidence": round(confidence, 3),
                })

            # Check if safe
            if name_lower in self.safe_lookup:
                safe_sounds.append({
                    "name": name,
                    "calm_weight": self.safe_lookup[name_lower],
                })

        # Calculate trigger score
        trigger_score = self._calculate_trigger_score(trigger_sounds, safe_sounds)

        return {
            "top_sounds": top_sounds,
            "trigger_sounds": trigger_sounds,
            "safe_sounds": safe_sounds,
            "trigger_score": round(trigger_score, 1),
        }

    def _calculate_trigger_score(self, triggers, safe_sounds) -> float:
        """Calculate 0-100 trigger risk score from detected sounds."""
        score = 0.0

        for trigger in triggers:
            # Weight by both risk_weight and confidence
            score += trigger["risk_weight"] * trigger["confidence"] * 80

        for safe in safe_sounds:
            score -= safe["calm_weight"] * 15

        return max(0, min(score, 100))

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream — puts audio into queue."""
        if status:
            logger.debug(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def get_volume_level(self, audio_data: np.ndarray) -> float:
        """Get the volume level (RMS) of audio data, 0-100 scale."""
        rms = np.sqrt(np.mean(audio_data ** 2))
        # Convert to a 0-100 scale (roughly)
        volume = min(rms * 500, 100)
        return round(volume, 1)


def run_microphone_demo():
    """
    Run real-time audio classification from microphone.
    Press Ctrl+C to stop.
    """
    import sounddevice as sd

    print("=" * 60)
    print("  PTSD Trigger Detection - Audio Classification Demo")
    print("  Using Google YAMNet (521 audio classes)")
    print("=" * 60)
    print("Listening to microphone... (first run downloads YAMNet ~200MB)")
    print("Press Ctrl+C to stop\n")

    classifier = AudioClassifier()
    sample_rate = classifier.sample_rate
    duration = classifier.duration
    samples_per_chunk = int(sample_rate * duration)

    print(f"Recording at {sample_rate} Hz, {duration}s chunks\n")
    print("-" * 60)

    try:
        while True:
            # Record audio chunk
            audio = sd.rec(
                samples_per_chunk,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()  # Wait for recording to finish

            audio_1d = audio.flatten()

            # Get volume
            volume = classifier.get_volume_level(audio_1d)

            # Classify
            results = classifier.classify_audio(audio_1d)

            # Display results
            timestamp = time.strftime("%H:%M:%S")
            score = results["trigger_score"]

            # Status color (for terminal)
            if score < 30:
                status = "LOW RISK"
            elif score < 60:
                status = "MEDIUM RISK"
            else:
                status = "!! HIGH RISK !!"

            print(f"\n[{timestamp}] Volume: {volume:.0f}% | {status} | Trigger Score: {score:.0f}%")

            # Show top sounds
            if results["top_sounds"]:
                sounds_str = ", ".join(f"{s['name']} ({s['confidence']:.1%})" for s in results["top_sounds"][:5])
                print(f"  Sounds: {sounds_str}")

            # Show triggers
            if results["trigger_sounds"]:
                print(f"  >> TRIGGERS DETECTED:")
                for t in results["trigger_sounds"]:
                    print(f"     - {t['name']} [{t['category']}] (risk: {t['risk_weight']}, conf: {t['confidence']:.1%})")

            # Show safe sounds
            if results["safe_sounds"]:
                safe_str = ", ".join(s["name"] for s in results["safe_sounds"])
                print(f"  Safe sounds: {safe_str}")

    except KeyboardInterrupt:
        print("\n\nStopped listening.")
        logger.info("Audio demo ended.")


if __name__ == "__main__":
    run_microphone_demo()
