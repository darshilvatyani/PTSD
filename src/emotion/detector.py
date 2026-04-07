"""
Emotion Detection Module for PTSD Trigger Detection System
Uses DeepFace library for real-time facial emotion recognition.

Library: https://github.com/serengil/deepface
Models available: VGG-Face, Facenet, OpenFace, DeepFace, ArcFace

Detects 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
"""

import sys
import os
import time
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import (
    EMOTION_DETECTOR_BACKEND,
    EMOTION_CONFIDENCE_THRESHOLD,
    TRIGGER_EMOTIONS,
    SAFE_EMOTIONS,
)
from src.utils.logger import setup_logger

logger = setup_logger("EmotionDetector")


class EmotionDetector:
    """
    Real-time facial emotion detector using DeepFace.
    
    Wraps the DeepFace library to provide:
    - Face detection from webcam frames
    - Emotion classification (7 classes)
    - PTSD trigger emotion scoring
    """

    def __init__(self):
        """Initialize the emotion detector."""
        self.detector_backend = EMOTION_DETECTOR_BACKEND
        self.confidence_threshold = EMOTION_CONFIDENCE_THRESHOLD
        self._deepface = None
        logger.info("EmotionDetector initialized")
        logger.info(f"  Backend: {self.detector_backend}")
        logger.info(f"  Trigger emotions: {TRIGGER_EMOTIONS}")

    def _load_deepface(self):
        """Lazy-load DeepFace to avoid slow import at startup."""
        if self._deepface is None:
            logger.info("Loading DeepFace library (first time may download models)...")
            from deepface import DeepFace
            self._deepface = DeepFace
            logger.info("DeepFace loaded successfully!")
        return self._deepface

    def analyze_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Analyze a video frame for facial emotions.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            List of detected faces with emotions:
            [
                {
                    "region": {"x": int, "y": int, "w": int, "h": int},
                    "dominant_emotion": str,
                    "emotions": {"angry": float, "fear": float, ...},
                    "trigger_score": float  # 0-100, how much this is a PTSD trigger
                },
                ...
            ]
        """
        DeepFace = self._load_deepface()

        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )

            # DeepFace returns a list of dicts (one per face)
            if not isinstance(results, list):
                results = [results]

            processed = []
            for face in results:
                # Skip if no face detected
                if face.get("face_confidence", 1.0) < self.confidence_threshold:
                    continue

                emotions = face.get("emotion", {})
                dominant = face.get("dominant_emotion", "neutral")
                region = face.get("region", {})

                # Calculate PTSD trigger score
                # Sum up trigger emotion percentages
                trigger_score = sum(
                    emotions.get(emo, 0) for emo in TRIGGER_EMOTIONS
                )
                # Clamp to 0-100
                trigger_score = min(max(trigger_score, 0), 100)

                processed.append({
                    "region": region,
                    "dominant_emotion": dominant,
                    "emotions": emotions,
                    "trigger_score": round(trigger_score, 1),
                })

            return processed

        except Exception as e:
            logger.debug(f"Frame analysis skipped: {e}")
            return []

    def get_emotion_color(self, emotion: str) -> tuple:
        """Get a BGR color for an emotion (for drawing on frame)."""
        colors = {
            "angry": (0, 0, 255),       # Red
            "disgust": (0, 128, 128),    # Dark yellow
            "fear": (0, 0, 200),         # Dark red
            "happy": (0, 255, 0),        # Green
            "sad": (255, 128, 0),        # Blue-orange
            "surprise": (0, 255, 255),   # Yellow
            "neutral": (200, 200, 200),  # Gray
        }
        return colors.get(emotion, (255, 255, 255))

    def draw_results(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        """
        Draw emotion detection results on the frame.

        Args:
            frame: Original BGR frame
            results: Output from analyze_frame()

        Returns:
            Annotated frame with face boxes and emotion labels
        """
        annotated = frame.copy()

        for face in results:
            region = face["region"]
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            emotion = face["dominant_emotion"]
            trigger_score = face["trigger_score"]
            color = self.get_emotion_color(emotion)

            # Draw face bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw emotion label
            label = f"{emotion} ({trigger_score:.0f}% risk)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x, y - 25), (x + label_size[0], y), color, -1)
            cv2.putText(annotated, label, (x, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw emotion bars (small bar chart on the right side of face)
            bar_x = x + w + 10
            bar_y = y
            for emo_name, emo_val in sorted(face["emotions"].items(), key=lambda x: -x[1]):
                bar_width = int(emo_val * 1.5)  # Scale for visibility
                emo_color = self.get_emotion_color(emo_name)

                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + 12), emo_color, -1)
                cv2.putText(annotated, f"{emo_name[:3]} {emo_val:.0f}%", (bar_x + bar_width + 5, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                bar_y += 16

        # Draw overall status bar at top
        if results:
            max_risk = max(f["trigger_score"] for f in results)
            status = "LOW RISK" if max_risk < 30 else "MEDIUM RISK" if max_risk < 60 else "HIGH RISK"
            status_color = (0, 200, 0) if max_risk < 30 else (0, 200, 255) if max_risk < 60 else (0, 0, 255)

            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), (40, 40, 40), -1)
            cv2.putText(annotated, f"PTSD Monitor | {status} | Trigger Score: {max_risk:.0f}%",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return annotated


def run_webcam_demo():
    """
    Run real-time emotion detection on webcam feed.
    Press 'q' to quit.
    """
    print("=" * 60)
    print("  PTSD Trigger Detection - Emotion Recognition Demo")
    print("=" * 60)
    print("Starting webcam... (first run downloads models ~500MB)")
    print("Press 'q' to quit\n")

    detector = EmotionDetector()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam! Make sure camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    analyze_every_n = 5  # Analyze every Nth frame (for performance)
    last_results = []

    logger.info("Webcam started. Analyzing emotions...")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read from webcam")
            break

        frame_count += 1

        # Analyze every Nth frame to keep things smooth
        if frame_count % analyze_every_n == 0:
            last_results = detector.analyze_frame(frame)

        # Always draw the latest results
        display_frame = detector.draw_results(frame, last_results)

        # Show FPS
        cv2.putText(display_frame, f"Frame: {frame_count}",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("PTSD Trigger Detection - Emotion Monitor", display_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam demo ended.")


if __name__ == "__main__":
    run_webcam_demo()
