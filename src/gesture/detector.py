"""
Gesture Detection Module for PTSD Trigger Detection System

Uses Google MediaPipe Tasks API (PoseLandmarker) for real-time
body gesture and posture detection from webcam frames.

Detects:
  - Fighting stance (fists raised)
  - Head covering (hands on head — distress)
  - Hand rubbing / fidgeting (nervous gesture)
  - Defensive posture (arms crossed)
  - Trembling (rapid small movements)
  - Crouching / cowering
  - Face touching (anxiety)
"""

import sys
import os
import time
import math
import numpy as np
import cv2
import yaml
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger

logger = setup_logger("GestureDetector")

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "gesture_config.yaml")
PROJECT_ROOT = os.path.dirname(os.path.dirname(CONFIG_DIR))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker.task")


class GestureDetector:
    """
    Detects PTSD-relevant body gestures using MediaPipe PoseLandmarker.
    Analyzes 33 body landmarks per frame to detect stress-related postures.
    """

    def __init__(self):
        """Initialize MediaPipe PoseLandmarker (Tasks API)."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Pose model not found at {MODEL_PATH}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )

        # Create PoseLandmarker
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.mp = mp

        # Pose landmark indices (same as the old PoseLandmark enum)
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_INDEX = 19
        self.RIGHT_INDEX = 20

        # Load config
        self.config = self._load_config()
        self.thresholds = self.config.get("thresholds", {})

        # Movement history for trembling detection
        self._landmark_history = deque(maxlen=10)

        logger.info("GestureDetector initialized (MediaPipe Tasks PoseLandmarker)")

    def _load_config(self) -> dict:
        try:
            with open(CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Gesture config not found, using defaults")
            return {"trigger_gestures": [], "safe_gestures": [], "thresholds": {}}

    def _dist_2d(self, a, b) -> float:
        """2D distance between two landmarks."""
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _angle(self, a, b, c) -> float:
        """Angle at point b formed by a-b-c (degrees)."""
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if mag_ba * mag_bc == 0:
            return 0
        cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Analyze a webcam frame for PTSD-related gestures.

        Args:
            frame: BGR image from OpenCV

        Returns:
            dict with gestures, safe_gestures, trigger_score, pose_landmarks
        """
        # Convert BGR → RGB and create mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

        # Detect pose
        result = self.landmarker.detect(mp_image)

        detected_gestures = []
        safe_gestures = []
        pose_lm_list = None

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]  # First person
            pose_lm_list = lm

            # Store for trembling detection
            lm_snapshot = [(l.x, l.y, l.z) for l in lm]
            self._landmark_history.append(lm_snapshot)

            # ---- 1. Head Covering (hands near head) ----
            nose = lm[self.NOSE]
            l_wrist = lm[self.LEFT_WRIST]
            r_wrist = lm[self.RIGHT_WRIST]
            head_thresh = self.thresholds.get("head_hand_proximity", 0.12)

            l_head_dist = self._dist_2d(l_wrist, nose)
            r_head_dist = self._dist_2d(r_wrist, nose)

            if l_head_dist < head_thresh and r_head_dist < head_thresh:
                conf = 1.0 - ((l_head_dist + r_head_dist) / (2 * head_thresh))
                detected_gestures.append({
                    "name": "Head Covering",
                    "confidence": round(min(conf, 1.0), 2),
                    "category": "distress",
                    "risk_weight": 0.85,
                })
            elif l_head_dist < head_thresh or r_head_dist < head_thresh:
                min_dist = min(l_head_dist, r_head_dist)
                conf = 1.0 - (min_dist / head_thresh)
                detected_gestures.append({
                    "name": "Face Touching",
                    "confidence": round(min(conf, 1.0), 2),
                    "category": "anxiety",
                    "risk_weight": 0.4,
                })

            # ---- 2. Hand Rubbing / Fidgeting ----
            hand_thresh = self.thresholds.get("hand_proximity", 0.08)
            wrist_dist = self._dist_2d(l_wrist, r_wrist)

            if wrist_dist < hand_thresh:
                conf = 1.0 - (wrist_dist / hand_thresh)
                detected_gestures.append({
                    "name": "Hand Rubbing",
                    "confidence": round(min(conf, 1.0), 2),
                    "category": "anxiety",
                    "risk_weight": 0.5,
                })

            # ---- 3. Fighting Stance (fists raised above shoulders) ----
            l_shoulder = lm[self.LEFT_SHOULDER]
            r_shoulder = lm[self.RIGHT_SHOULDER]

            l_fist_raised = l_wrist.y < l_shoulder.y - 0.05
            r_fist_raised = r_wrist.y < r_shoulder.y - 0.05

            if l_fist_raised and r_fist_raised:
                l_elbow = lm[self.LEFT_ELBOW]
                r_elbow = lm[self.RIGHT_ELBOW]
                l_angle = self._angle(l_shoulder, l_elbow, l_wrist)
                r_angle = self._angle(r_shoulder, r_elbow, r_wrist)
                if l_angle < 120 and r_angle < 120:
                    avg_raise = abs(l_shoulder.y - l_wrist.y) + abs(r_shoulder.y - r_wrist.y)
                    conf = min(avg_raise * 5, 1.0)
                    detected_gestures.append({
                        "name": "Fighting Stance",
                        "confidence": round(conf, 2),
                        "category": "aggression",
                        "risk_weight": 0.9,
                    })

            # ---- 4. Defensive Posture (arms crossed) ----
            l_elbow = lm[self.LEFT_ELBOW]
            r_elbow = lm[self.RIGHT_ELBOW]

            if l_wrist.x > r_wrist.x and not (l_fist_raised or r_fist_raised):
                elbow_angle_l = self._angle(l_shoulder, l_elbow, l_wrist)
                elbow_angle_r = self._angle(r_shoulder, r_elbow, r_wrist)
                cross_thresh = self.thresholds.get("crossed_arms_angle", 45)
                if elbow_angle_l < cross_thresh + 30 and elbow_angle_r < cross_thresh + 30:
                    detected_gestures.append({
                        "name": "Defensive Posture",
                        "confidence": 0.7,
                        "category": "defense",
                        "risk_weight": 0.6,
                    })

            # ---- 5. Crouching / Cowering ----
            l_hip = lm[self.LEFT_HIP]
            r_hip = lm[self.RIGHT_HIP]
            shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            hip_y = (l_hip.y + r_hip.y) / 2
            torso_height = abs(hip_y - shoulder_y)
            crouch_ratio = self.thresholds.get("crouch_ratio", 0.6)

            if torso_height < crouch_ratio * 0.3 and shoulder_y > 0.5:
                conf = 1.0 - (torso_height / (crouch_ratio * 0.3))
                detected_gestures.append({
                    "name": "Crouching",
                    "confidence": round(min(conf, 1.0), 2),
                    "category": "fear",
                    "risk_weight": 0.8,
                })

            # ---- 6. Trembling / Shaking ----
            if len(self._landmark_history) >= 5:
                trembling_thresh = self.thresholds.get("trembling_threshold", 0.015)
                recent = list(self._landmark_history)
                deltas = []
                check_indices = [self.LEFT_WRIST, self.RIGHT_WRIST,
                                 self.LEFT_INDEX, self.RIGHT_INDEX]
                for i in range(1, len(recent)):
                    frame_delta = 0
                    for j in check_indices:
                        dx = recent[i][j][0] - recent[i-1][j][0]
                        dy = recent[i][j][1] - recent[i-1][j][1]
                        frame_delta += math.sqrt(dx**2 + dy**2)
                    deltas.append(frame_delta / len(check_indices))

                avg_delta = sum(deltas) / len(deltas)
                if avg_delta > trembling_thresh:
                    conf = min(avg_delta / (trembling_thresh * 3), 1.0)
                    detected_gestures.append({
                        "name": "Trembling",
                        "confidence": round(conf, 2),
                        "category": "fear",
                        "risk_weight": 0.75,
                    })

            # ---- 7. Relaxed Posture (safe) ----
            arms_at_sides = (
                l_wrist.y > l_hip.y - 0.05 and r_wrist.y > r_hip.y - 0.05
                and abs(l_wrist.x - l_hip.x) < 0.1
                and abs(r_wrist.x - r_hip.x) < 0.1
            )
            if arms_at_sides and not detected_gestures:
                safe_gestures.append({"name": "Relaxed Posture", "calm_weight": 0.3})

        # Calculate trigger score
        trigger_score = self._calculate_score(detected_gestures, safe_gestures)

        return {
            "gestures": detected_gestures,
            "safe_gestures": safe_gestures,
            "trigger_score": round(trigger_score, 1),
            "pose_landmarks": pose_lm_list,
        }

    def _calculate_score(self, gestures, safe_gestures) -> float:
        score = 0.0
        for g in gestures:
            score += g["risk_weight"] * g["confidence"] * 80
        for s in safe_gestures:
            score -= s["calm_weight"] * 15
        return max(0, min(score, 100))

    def draw_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw pose skeleton and gesture labels on frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        pose_lm_list = results.get("pose_landmarks")
        if pose_lm_list:
            # Draw skeleton connections
            connections = [
                (self.LEFT_SHOULDER, self.RIGHT_SHOULDER),
                (self.LEFT_SHOULDER, self.LEFT_ELBOW),
                (self.LEFT_ELBOW, self.LEFT_WRIST),
                (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
                (self.RIGHT_ELBOW, self.RIGHT_WRIST),
                (self.LEFT_SHOULDER, self.LEFT_HIP),
                (self.RIGHT_SHOULDER, self.RIGHT_HIP),
                (self.LEFT_HIP, self.RIGHT_HIP),
                (self.NOSE, self.LEFT_SHOULDER),
                (self.NOSE, self.RIGHT_SHOULDER),
            ]

            points = {}
            for lm_idx in set(sum(connections, ())):
                lm = pose_lm_list[lm_idx]
                px, py = int(lm.x * w), int(lm.y * h)
                points[lm_idx] = (px, py)
                cv2.circle(annotated, (px, py), 4, (0, 255, 200), -1)

            for a, b in connections:
                if a in points and b in points:
                    cv2.line(annotated, points[a], points[b], (0, 200, 150), 2)

        # Draw gesture labels
        y_offset = 50
        for g in results.get("gestures", []):
            color = (0, 0, 255) if g["risk_weight"] > 0.6 else (0, 200, 255)
            label = f"{g['name']} ({g['confidence']:.0%})"
            cv2.putText(annotated, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        for s in results.get("safe_gestures", []):
            cv2.putText(annotated, f"OK: {s['name']}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            y_offset += 20

        return annotated


def run_gesture_demo():
    """Run real-time gesture detection demo. Press 'q' to quit."""
    print("=" * 60)
    print("  PTSD Trigger Detection - Gesture Recognition Demo")
    print("  Using Google MediaPipe PoseLandmarker (Tasks API)")
    print("=" * 60)
    print("Starting webcam... Press 'q' to quit\n")

    detector = GestureDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    results = {"gestures": [], "safe_gestures": [], "trigger_score": 0, "pose_landmarks": None}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.detect_frame(frame)
            display = detector.draw_results(frame, results)

            # Print detected gestures
            if results["gestures"]:
                gestures_str = ", ".join(
                    f"{g['name']} ({g['confidence']:.0%})" for g in results["gestures"]
                )
                print(f"  [{time.strftime('%H:%M:%S')}] "
                      f"Gestures: {gestures_str} | "
                      f"Risk: {results['trigger_score']:.0f}%")

            # Draw risk bar
            score = results["trigger_score"]
            color = (0, 200, 0) if score < 30 else (0, 200, 255) if score < 60 else (0, 0, 255)
            cv2.rectangle(display, (0, 0), (display.shape[1], 35), (30, 30, 30), -1)
            cv2.putText(display, f"Gesture Risk: {score:.0f}%",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("PTSD - Gesture Detection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Gesture demo ended.")


if __name__ == "__main__":
    run_gesture_demo()
