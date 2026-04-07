# 🧠 PTSD Trigger Detection & Early Warning System

**AI & IoT Based PTSD Trigger Detection and Early Warning System Using Computer Vision**

## 🔹 Overview

A smart system that detects potential PTSD triggers in real-time using:
- **Facial Emotion Recognition** — DeepFace library
- **Object Detection** — YOLOv8 (Ultralytics)
- **Sound Classification** — Google YAMNet
- **Physiological Stress Detection** — Heart rate monitoring (dummy → real sensor)

All signals are fused into a **risk score (0–100%)** and displayed on a **Streamlit dashboard**.

## 🔹 Tech Stack

| Module | Library |
|--------|---------|
| Emotion Detection | DeepFace |
| Object Detection | YOLOv8 (Ultralytics) |
| Sound Classification | YAMNet (TensorFlow Hub) |
| Stress Analysis | scikit-learn |
| Fusion | XGBoost |
| Dashboard | Streamlit |
| Explainability | Grad-CAM |

## 🔹 Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run emotion detection demo
python src/emotion/detector.py

# Run full dashboard
streamlit run dashboard/app.py
```

## 🔹 Project Structure

```
PTSD3/
├── src/
│   ├── emotion/          # DeepFace emotion detection
│   ├── object_detection/ # YOLOv8 trigger detection
│   ├── audio/            # YAMNet sound classification
│   ├── stress/           # Heart rate / stress analysis
│   ├── fusion/           # Multi-modal risk scoring
│   └── utils/            # Config & helpers
├── dashboard/            # Streamlit web dashboard
├── data/                 # Datasets (gitignored)
├── models/               # Saved model weights
├── tests/                # Unit tests
├── docs/                 # Documentation
└── requirements.txt
```

## 🔹 Author

BTech Minor Project — PTSD Trigger Detection System
