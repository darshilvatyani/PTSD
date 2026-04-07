"""
Multi-Modal Fusion Engine for PTSD Trigger Detection System

This is the CORE module that combines ALL detection modules into
one unified PTSD risk assessment:

  📷 Emotion (DeepFace)     → emotion_score
  📷 Objects (YOLO26)       → object_score
  🎤 Audio (YAMNet)         → audio_score
  💓 Stress (Random Forest) → stress_score
       ↓
  ⚡ FUSION ENGINE → Overall Risk Score (0-100%)

The fusion uses weighted combination + escalation logic.
"""

import sys
import os
import time
import threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import FUSION_WEIGHTS, RISK_LOW, RISK_MEDIUM, RISK_HIGH
from src.utils.logger import setup_logger

logger = setup_logger("FusionEngine")


class FusionEngine:
    """
    Combines all PTSD detection modules into a single risk assessment.

    Runs emotion + object detection on webcam frames,
    audio classification on mic input, and stress prediction
    on physiological data — then fuses all scores.
    """

    def __init__(self, enable_emotion=True, enable_objects=True,
                 enable_audio=True, enable_stress=True):
        """
        Initialize the fusion engine with selected modules.

        Args:
            enable_emotion: Use DeepFace emotion detection
            enable_objects: Use YOLO26 object detection
            enable_audio: Use YAMNet audio classification
            enable_stress: Use stress classifier
        """
        self.weights = FUSION_WEIGHTS
        self.modules_enabled = {
            "emotion": enable_emotion,
            "object": enable_objects,
            "audio": enable_audio,
            "stress": enable_stress,
        }

        # Module instances (lazy-loaded)
        self._emotion_detector = None
        self._object_detector = None
        self._audio_classifier = None
        self._stress_classifier = None

        # Latest results from each module
        self.latest_results = {
            "emotion": {"trigger_score": 0, "data": {}},
            "object": {"trigger_score": 0, "data": {}},
            "audio": {"trigger_score": 0, "data": {}},
            "stress": {"trigger_score": 0, "data": {}},
        }

        # Overall risk
        self.overall_risk = 0.0
        self.risk_level = "LOW"
        self.risk_history = []  # Track risk over time

        # Audio runs in background thread
        self._audio_thread = None
        self._audio_running = False

        logger.info("FusionEngine initialized")
        logger.info(f"  Modules: {self.modules_enabled}")
        logger.info(f"  Weights: {self.weights}")

    # ============================================================
    # Module Loaders (lazy — only load when first used)
    # ============================================================

    def _get_emotion_detector(self):
        if self._emotion_detector is None:
            from src.emotion.detector import EmotionDetector
            self._emotion_detector = EmotionDetector()
        return self._emotion_detector

    def _get_object_detector(self):
        if self._object_detector is None:
            from src.object_detection.detector import ObjectDetector
            self._object_detector = ObjectDetector()
        return self._object_detector

    def _get_audio_classifier(self):
        if self._audio_classifier is None:
            from src.audio.classifier import AudioClassifier
            self._audio_classifier = AudioClassifier()
        return self._audio_classifier

    def _get_stress_classifier(self):
        if self._stress_classifier is None:
            from src.stress.classifier import StressClassifier
            self._stress_classifier = StressClassifier()
        return self._stress_classifier

    # ============================================================
    # Per-Module Analysis
    # ============================================================

    def analyze_emotion(self, frame: np.ndarray) -> dict:
        """Run emotion detection on a webcam frame."""
        if not self.modules_enabled["emotion"]:
            return {"trigger_score": 0, "data": {}}

        detector = self._get_emotion_detector()
        results = detector.analyze_frame(frame)

        score = 0
        if results:
            score = max(r["trigger_score"] for r in results)

        self.latest_results["emotion"] = {
            "trigger_score": score,
            "data": results,
        }
        return self.latest_results["emotion"]

    def analyze_objects(self, frame: np.ndarray) -> dict:
        """Run object detection on a webcam frame."""
        if not self.modules_enabled["object"]:
            return {"trigger_score": 0, "data": {}}

        detector = self._get_object_detector()
        results = detector.detect_frame(frame)

        self.latest_results["object"] = {
            "trigger_score": results["trigger_score"],
            "data": results,
        }
        return self.latest_results["object"]

    def analyze_audio(self, audio_data: np.ndarray) -> dict:
        """Run audio classification on an audio chunk."""
        if not self.modules_enabled["audio"]:
            return {"trigger_score": 0, "data": {}}

        classifier = self._get_audio_classifier()
        results = classifier.classify_audio(audio_data)

        self.latest_results["audio"] = {
            "trigger_score": results["trigger_score"],
            "data": results,
        }
        return self.latest_results["audio"]

    def analyze_stress(self, reading: dict = None) -> dict:
        """Run stress prediction on physiological data."""
        if not self.modules_enabled["stress"]:
            return {"trigger_score": 0, "data": {}}

        classifier = self._get_stress_classifier()
        results = classifier.predict(reading)

        self.latest_results["stress"] = {
            "trigger_score": results["trigger_score"],
            "data": results,
        }
        return self.latest_results["stress"]

    # ============================================================
    # Fusion — Combine All Scores
    # ============================================================

    def calculate_risk(self) -> dict:
        """
        Calculate the overall PTSD risk score by fusing all module scores.

        Returns:
            dict with:
            {
                "overall_risk": float (0-100),
                "risk_level": str ("LOW" / "MEDIUM" / "HIGH"),
                "module_scores": {module: score},
                "module_contributions": {module: weighted_score},
                "timestamp": float
            }
        """
        module_scores = {}
        module_contributions = {}
        total_weight = 0

        for module, enabled in self.modules_enabled.items():
            if enabled:
                score = self.latest_results[module]["trigger_score"]
                weight = self.weights.get(module, 0.25)
                module_scores[module] = score
                module_contributions[module] = round(score * weight, 1)
                total_weight += weight

        # Weighted average (normalize by total active weight)
        if total_weight > 0:
            raw_risk = sum(module_contributions.values()) / total_weight
        else:
            raw_risk = 0

        # Escalation: if ANY single module is very high, boost overall risk
        max_single = max(module_scores.values()) if module_scores else 0
        if max_single > 80:
            raw_risk = max(raw_risk, max_single * 0.8)  # Don't let fusion dilute a severe trigger

        # Multi-trigger escalation: if 2+ modules are elevated, boost
        elevated_count = sum(1 for s in module_scores.values() if s > 40)
        if elevated_count >= 2:
            raw_risk *= 1.15  # 15% boost for multi-modal triggers
        if elevated_count >= 3:
            raw_risk *= 1.1   # Additional 10% for 3+ simultaneous triggers

        # Clamp
        self.overall_risk = round(max(0, min(raw_risk, 100)), 1)

        # Risk level
        if self.overall_risk < RISK_LOW:
            self.risk_level = "LOW"
        elif self.overall_risk < RISK_MEDIUM:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "HIGH"

        # Track history
        result = {
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level,
            "module_scores": module_scores,
            "module_contributions": module_contributions,
            "timestamp": time.time(),
        }
        self.risk_history.append(result)

        # Keep last 100 entries
        if len(self.risk_history) > 100:
            self.risk_history = self.risk_history[-100:]

        return result

    # ============================================================
    # Audio Background Thread
    # ============================================================

    def _audio_loop(self):
        """Background loop that continuously classifies mic audio."""
        import sounddevice as sd

        classifier = self._get_audio_classifier()
        sample_rate = classifier.sample_rate
        duration = classifier.duration
        samples = int(sample_rate * duration)

        logger.info("Audio background thread started")

        while self._audio_running:
            try:
                audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype="float32")
                sd.wait()
                audio_1d = audio.flatten()
                self.analyze_audio(audio_1d)
            except Exception as e:
                logger.debug(f"Audio thread error: {e}")
                time.sleep(1)

        logger.info("Audio background thread stopped")

    def start_audio_background(self):
        """Start audio classification in a background thread."""
        if not self.modules_enabled["audio"]:
            return
        if self._audio_thread and self._audio_thread.is_alive():
            return

        self._audio_running = True
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._audio_thread.start()
        logger.info("Audio background thread launched")

    def stop_audio_background(self):
        """Stop the audio background thread."""
        self._audio_running = False
        if self._audio_thread:
            self._audio_thread.join(timeout=3)
        logger.info("Audio background thread stopped")

    # ============================================================
    # Drawing / Visualization
    # ============================================================

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all detection results on a webcam frame.
        Combines emotion boxes, object boxes, and overall risk bar.
        """
        annotated = frame.copy()

        # Draw emotion results
        if self.modules_enabled["emotion"] and self.latest_results["emotion"]["data"]:
            emotion_detector = self._get_emotion_detector()
            annotated = emotion_detector.draw_results(annotated, self.latest_results["emotion"]["data"])

        # Draw object results
        if self.modules_enabled["object"] and self.latest_results["object"]["data"]:
            object_detector = self._get_object_detector()
            annotated = object_detector.draw_results(annotated, self.latest_results["object"]["data"])

        # Draw overall risk bar at top
        risk = self.overall_risk
        level = self.risk_level
        if level == "LOW":
            bar_color = (0, 200, 0)
        elif level == "MEDIUM":
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 0, 255)

        # Background bar
        bar_height = 40
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], bar_height), (30, 30, 30), -1)

        # Risk fill bar
        fill_width = int((risk / 100) * annotated.shape[1])
        cv2.rectangle(annotated, (0, 0), (fill_width, bar_height), bar_color, -1)

        # Text
        cv2.putText(annotated,
                    f"PTSD RISK: {risk:.0f}% [{level}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Module scores on the right
        scores_text = " | ".join(
            f"{m[:3].upper()}: {self.latest_results[m]['trigger_score']:.0f}%"
            for m in ["emotion", "object", "audio", "stress"]
            if self.modules_enabled[m]
        )
        cv2.putText(annotated, scores_text,
                    (annotated.shape[1] - 400, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Stress info at bottom
        if self.modules_enabled["stress"] and self.latest_results["stress"]["data"]:
            stress_data = self.latest_results["stress"]["data"]
            reading = stress_data.get("reading", {})
            hr = reading.get("heart_rate", 0)
            gsr = reading.get("gsr", 0)
            stress_level = stress_data.get("stress_level", "unknown")

            cv2.rectangle(annotated, (0, annotated.shape[0] - 30),
                         (annotated.shape[1], annotated.shape[0]), (30, 30, 30), -1)
            cv2.putText(annotated,
                       f"HR: {hr:.0f} bpm | GSR: {gsr:.1f} uS | Stress: {stress_level}",
                       (10, annotated.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated


def run_fusion_demo():
    """
    Run the full multi-modal fusion demo.
    Combines webcam (emotion + objects) + mic (audio) + dummy stress data.
    Press 'q' to quit.
    """
    print("=" * 60)
    print("  PTSD Trigger Detection - FULL SYSTEM DEMO")
    print("  Fusing: Emotion + Objects + Audio + Stress")
    print("=" * 60)
    print("Loading all modules (this may take a moment)...")
    print("Press 'q' to quit\n")

    # Initialize fusion engine
    engine = FusionEngine(
        enable_emotion=True,
        enable_objects=True,
        enable_audio=True,
        enable_stress=True,
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Start audio in background
    engine.start_audio_background()

    frame_count = 0
    emotion_interval = 5    # Analyze emotions every 5 frames
    object_interval = 3     # Analyze objects every 3 frames
    stress_interval = 10    # Analyze stress every 10 frames

    logger.info("Full system running! All 4 modules active.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run modules at different intervals for performance
            if frame_count % emotion_interval == 0:
                engine.analyze_emotion(frame)

            if frame_count % object_interval == 0:
                engine.analyze_objects(frame)

            if frame_count % stress_interval == 0:
                engine.analyze_stress()  # Uses dummy/serial data automatically

            # Audio runs in background thread (continuously)

            # Calculate fused risk
            risk_result = engine.calculate_risk()

            # Draw everything
            display = engine.draw_on_frame(frame)

            cv2.imshow("PTSD Trigger Detection - Full System", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        engine.stop_audio_background()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Full system demo ended.")


if __name__ == "__main__":
    run_fusion_demo()
