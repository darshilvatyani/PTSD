"""
Object Detection Module for PTSD Trigger Detection System
Uses Ultralytics YOLO26 (pre-trained on COCO) for real-time object detection.

Library: https://github.com/ultralytics/ultralytics
Pre-trained on 80 COCO classes including: person, car, truck, knife, scissors, etc.

We filter detections through trigger_config.yaml to identify PTSD-relevant objects.
"""

import sys
import os
import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import YOLO_MODEL, YOLO_CONFIDENCE_THRESHOLD
from src.utils.logger import setup_logger

logger = setup_logger("ObjectDetector")

# Path to trigger config
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
TRIGGER_CONFIG_PATH = os.path.join(CONFIG_DIR, "trigger_config.yaml")


class ObjectDetector:
    """
    Real-time PTSD trigger object detector using YOLO26.
    
    Detects objects via YOLO26, then filters them through
    a PTSD trigger configuration to calculate a risk score.
    """

    def __init__(self):
        """Initialize the object detector and load trigger config."""
        self.model = None
        self.trigger_config = self._load_trigger_config()
        self.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD

        # Build lookup dict: object_name → risk_weight
        self.trigger_lookup = {}
        for obj in self.trigger_config.get("trigger_objects", []):
            self.trigger_lookup[obj["name"]] = {
                "risk_weight": obj["risk_weight"],
                "category": obj["category"],
                "description": obj.get("description", ""),
            }

        # Safe objects
        self.safe_lookup = {}
        for obj in self.trigger_config.get("safe_objects", []):
            self.safe_lookup[obj["name"]] = obj["calm_weight"]

        # Crowd config
        self.crowd_config = self.trigger_config.get("crowd_detection", {})

        logger.info("ObjectDetector initialized")
        logger.info(f"  Trigger objects: {list(self.trigger_lookup.keys())}")
        logger.info(f"  Crowd threshold: {self.crowd_config.get('person_threshold', 5)}")

    def _load_trigger_config(self) -> dict:
        """Load the trigger configuration from YAML file."""
        try:
            with open(TRIGGER_CONFIG_PATH, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded trigger config from {TRIGGER_CONFIG_PATH}")
            return config
        except FileNotFoundError:
            logger.warning(f"Trigger config not found at {TRIGGER_CONFIG_PATH}, using defaults")
            return {"trigger_objects": [], "crowd_detection": {"enabled": True, "person_threshold": 5}}

    def _load_model(self):
        """Lazy-load YOLO26 model (downloads on first use)."""
        if self.model is None:
            logger.info(f"Loading YOLO26 model: {YOLO_MODEL} (first time downloads ~6MB)...")
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL)
            logger.info("YOLO26 model loaded successfully!")
        return self.model

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Detect objects in a video frame and evaluate PTSD triggers.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            dict with:
            {
                "all_objects": [{"name": str, "confidence": float, "box": [x1,y1,x2,y2]}],
                "trigger_objects": [{"name": str, "category": str, "risk_weight": float, "box": ...}],
                "safe_objects": [{"name": str, "calm_weight": float}],
                "person_count": int,
                "is_crowded": bool,
                "trigger_score": float  # 0-100
            }
        """
        model = self._load_model()

        # Run YOLO26 inference
        results = model(frame, conf=self.confidence_threshold, verbose=False)

        all_objects = []
        trigger_objects = []
        safe_objects = []
        person_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get detection info
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                name = model.names[cls_id]
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]

                obj_info = {
                    "name": name,
                    "confidence": round(confidence, 2),
                    "box": coords,
                }
                all_objects.append(obj_info)

                # Count persons for crowd detection
                if name == "person":
                    person_count += 1

                # Check if it's a trigger object
                if name in self.trigger_lookup:
                    trigger_info = {
                        **obj_info,
                        "category": self.trigger_lookup[name]["category"],
                        "risk_weight": self.trigger_lookup[name]["risk_weight"],
                        "description": self.trigger_lookup[name]["description"],
                    }
                    trigger_objects.append(trigger_info)

                # Check if it's a safe object
                if name in self.safe_lookup:
                    safe_objects.append({
                        "name": name,
                        "calm_weight": self.safe_lookup[name],
                    })

        # Crowd detection
        crowd_threshold = self.crowd_config.get("person_threshold", 5)
        is_crowded = person_count >= crowd_threshold

        # Calculate trigger score (0-100)
        trigger_score = self._calculate_trigger_score(
            trigger_objects, safe_objects, person_count, is_crowded
        )

        return {
            "all_objects": all_objects,
            "trigger_objects": trigger_objects,
            "safe_objects": safe_objects,
            "person_count": person_count,
            "is_crowded": is_crowded,
            "trigger_score": round(trigger_score, 1),
        }

    def _calculate_trigger_score(self, triggers, safe_objects, person_count, is_crowded) -> float:
        """Calculate a 0-100 trigger risk score from detected objects."""
        score = 0.0

        # Add risk from trigger objects
        for trigger in triggers:
            # Skip "person" from trigger objects since we handle crowd separately
            if trigger["name"] == "person":
                continue
            score += trigger["risk_weight"] * 30  # Scale: 0.9 weight → 27 points

        # Add crowd risk
        if is_crowded:
            crowd_weight = self.crowd_config.get("risk_weight", 0.7)
            # More people = more risk (up to a cap)
            crowd_factor = min(person_count / 10.0, 1.5)
            score += crowd_weight * 30 * crowd_factor

        # Subtract safe object calming effect
        for safe in safe_objects:
            score -= safe["calm_weight"] * 15

        return max(0, min(score, 100))

    def draw_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """
        Draw detection results on the frame with color-coded boxes.
        
        - Red boxes: trigger objects
        - Green boxes: safe objects
        - Yellow boxes: neutral objects
        - Orange highlight: crowd warning
        """
        annotated = frame.copy()
        trigger_names = set(self.trigger_lookup.keys())
        safe_names = set(self.safe_lookup.keys())

        for obj in results["all_objects"]:
            x1, y1, x2, y2 = obj["box"]
            name = obj["name"]
            conf = obj["confidence"]

            # Color based on trigger status
            if name in trigger_names:
                color = (0, 0, 255)       # Red — danger
                label_prefix = "⚠ TRIGGER"
            elif name in safe_names:
                color = (0, 200, 0)       # Green — safe
                label_prefix = "✓ SAFE"
            else:
                color = (200, 200, 0)     # Yellow — neutral
                label_prefix = ""

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{label_prefix} {name} {conf:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Crowd warning
        if results["is_crowded"]:
            cv2.putText(annotated, f"CROWD DETECTED ({results['person_count']} people)",
                        (10, annotated.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Status bar at top
        score = results["trigger_score"]
        status = "LOW RISK" if score < 30 else "MEDIUM RISK" if score < 60 else "HIGH RISK"
        status_color = (0, 200, 0) if score < 30 else (0, 200, 255) if score < 60 else (0, 0, 255)

        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), (40, 40, 40), -1)
        cv2.putText(annotated, f"PTSD Object Monitor | {status} | Score: {score:.0f}% | Objects: {len(results['all_objects'])}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return annotated


def run_webcam_demo():
    """
    Run real-time object detection on webcam.
    Press 'q' to quit.
    """
    print("=" * 60)
    print("  PTSD Trigger Detection - Object Detection Demo")
    print("=" * 60)
    print("Starting webcam with YOLO26...")
    print("Red boxes = TRIGGER objects | Green = SAFE | Yellow = Neutral")
    print("Press 'q' to quit\n")

    detector = ObjectDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam! Make sure camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    last_results = {
        "all_objects": [], "trigger_objects": [], "safe_objects": [],
        "person_count": 0, "is_crowded": False, "trigger_score": 0
    }

    logger.info("Webcam started for object detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect every 3rd frame for performance
        if frame_count % 3 == 0:
            last_results = detector.detect_frame(frame)

        # Draw results
        display = detector.draw_results(frame, last_results)

        cv2.imshow("PTSD Trigger Detection - Object Monitor", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Object detection demo ended.")


if __name__ == "__main__":
    run_webcam_demo()
