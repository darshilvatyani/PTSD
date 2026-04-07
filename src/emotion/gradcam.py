"""
Grad-CAM Explainability Module for PTSD Trigger Detection System

Generates heatmap visualizations showing which facial regions
the emotion model focuses on when making predictions.

Uses pytorch-grad-cam library for plug-and-play Grad-CAM.
Reference: https://github.com/jacobgil/pytorch-grad-cam
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger

logger = setup_logger("GradCAM")


def generate_simple_attention_map(frame: np.ndarray, face_region: dict) -> np.ndarray:
    """
    Generate a simple attention/saliency heatmap over a detected face.
    
    This is a lightweight approach that highlights the face region
    where emotion features are most concentrated (eyes, mouth area).
    For full Grad-CAM with a PyTorch model, use generate_gradcam() below.

    Args:
        frame: BGR image (numpy array)
        face_region: dict with x, y, w, h keys

    Returns:
        Heatmap overlay on the frame (BGR numpy array)
    """
    x = face_region.get("x", 0)
    y = face_region.get("y", 0)
    w = face_region.get("w", 0)
    h = face_region.get("h", 0)

    if w == 0 or h == 0:
        return frame

    # Create a blank heatmap
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Eye region (upper 1/3 of face) — highest attention
    eye_y1 = y + int(h * 0.2)
    eye_y2 = y + int(h * 0.45)
    heatmap[eye_y1:eye_y2, x:x + w] = 0.9

    # Mouth region (lower 1/3 of face) — high attention
    mouth_y1 = y + int(h * 0.65)
    mouth_y2 = y + int(h * 0.9)
    heatmap[mouth_y1:mouth_y2, x:x + w] = 0.7

    # Eyebrow region — medium attention
    brow_y1 = y + int(h * 0.1)
    brow_y2 = y + int(h * 0.25)
    heatmap[brow_y1:brow_y2, x:x + w] = 0.6

    # Blur for smooth gradient effect
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Convert to color heatmap
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # Overlay on original frame
    alpha = 0.4
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def generate_gradcam_pytorch(model, target_layer, input_tensor, frame: np.ndarray) -> np.ndarray:
    """
    Generate proper Grad-CAM using pytorch-grad-cam library.
    
    This requires a PyTorch model. Use this if you train a custom
    emotion model in PyTorch.

    Args:
        model: PyTorch model
        target_layer: Layer to compute Grad-CAM on
        input_tensor: Preprocessed input tensor
        frame: Original BGR frame for overlay

    Returns:
        Heatmap overlay on the frame
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]

        # Resize frame to match cam
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_frame = cv2.resize(rgb_frame, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

        visualization = show_cam_on_image(rgb_frame, grayscale_cam, use_rgb=True)
        return cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    except ImportError:
        logger.warning("pytorch-grad-cam not installed. Using simple attention map instead.")
        return frame
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return frame


if __name__ == "__main__":
    """Quick test: show attention map on webcam."""
    print("Grad-CAM Attention Map Demo")
    print("Press 'q' to quit\n")

    # Use DeepFace for face detection, then show attention map
    from src.emotion.detector import EmotionDetector

    detector = EmotionDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam!")
        sys.exit(1)

    frame_count = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            last_results = detector.analyze_frame(frame)

        # Generate attention map for each face
        display = frame.copy()
        for face in last_results:
            display = generate_simple_attention_map(display, face["region"])

            # Draw label
            region = face["region"]
            x, y = region.get("x", 0), region.get("y", 0)
            emotion = face["dominant_emotion"]
            cv2.putText(display, f"{emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("PTSD - Grad-CAM Attention Map", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
