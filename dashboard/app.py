"""
PTSD Trigger Detection — Streamlit Dashboard (Cloud-Ready)
==========================================================

Run with:  streamlit run dashboard/app.py
Deploy on: Streamlit Cloud (share.streamlit.io)
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import (
    FUSION_WEIGHTS, RISK_LOW, RISK_MEDIUM, RISK_HIGH,
    HEART_RATE_SOURCE, DASHBOARD_REFRESH_RATE,
    AUDIO_SAMPLE_RATE, AUDIO_DURATION,
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="PTSD Sentinel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Design System & CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0c0e14;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(56,182,255,0.04) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,90,90,0.04) 0%, transparent 60%);
    }

    /* Hide sidebar & branding */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    #MainMenu, footer { display: none !important; }

    /* ── Cards ── */
    .card {
        background: #13161f;
        border: 1px solid #1e2130;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }

    /* ── Score Tiles ── */
    .score-tile {
        background: #13161f;
        border: 1px solid #1e2130;
        border-radius: 12px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        transition: border-color 0.3s;
    }
    .score-tile-low    { border-top: 3px solid #22c55e; }
    .score-tile-medium { border-top: 3px solid #f59e0b; }
    .score-tile-high   { border-top: 3px solid #ef4444; }

    .score-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5a6278;
    }
    .score-value {
        font-family: 'DM Mono', monospace;
        font-size: 32px;
        font-weight: 500;
        line-height: 1;
    }
    .score-bar-track {
        height: 4px;
        background: #1e2130;
        border-radius: 99px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.4s ease;
    }

    /* ── Stat Pills ── */
    .stat-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 4px;
    }
    .stat-pill {
        background: #1a1d28;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 8px 14px;
        flex: 1;
        min-width: 90px;
    }
    .stat-pill-label {
        font-size: 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #5a6278;
        margin-bottom: 2px;
    }
    .stat-pill-value {
        font-family: 'DM Mono', monospace;
        font-size: 18px;
        color: #e2e8f0;
        font-weight: 500;
    }

    /* ── Section Headers ── */
    .sec-header {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5a6278;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .sec-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #1e2130;
    }

    /* ── Alert Banners ── */
    .alert-high {
        background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.3);
        border-left: 3px solid #ef4444;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 14px;
        font-weight: 500;
        color: #fca5a5;
        animation: blink 2.5s ease-in-out infinite;
    }
    .alert-safe {
        background: linear-gradient(90deg, rgba(34,197,94,0.1), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.2);
        border-left: 3px solid #22c55e;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 14px;
        font-weight: 500;
        color: #86efac;
    }
    .alert-medium {
        background: linear-gradient(90deg, rgba(245,158,11,0.1), rgba(245,158,11,0.03));
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 3px solid #f59e0b;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 14px;
        font-weight: 500;
        color: #fcd34d;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.6} }

    /* ── Sound Items ── */
    .sound-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        background: #191c27;
        border: 1px solid #1e2130;
    }
    .sound-row-trigger {
        background: rgba(239,68,68,0.07);
        border-color: rgba(239,68,68,0.25);
    }
    .sound-name {
        flex: 1;
        font-size: 13px;
        color: #c8d0e0;
    }
    .sound-conf {
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        color: #5a6278;
    }
    .sound-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Volume Bar ── */
    .vol-track {
        height: 6px;
        background: #1e2130;
        border-radius: 99px;
        overflow: hidden;
        margin: 6px 0 14px;
    }
    .vol-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.3s ease;
    }

    /* ── Log Items ── */
    .log-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        background: #13161f;
        border: 1px solid #1e2130;
        font-size: 12px;
    }
    .log-risk  { border-left: 2px solid #ef4444; }
    .log-audio { border-left: 2px solid #a78bfa; }
    .log-gesture { border-left: 2px solid #2dd4bf; }
    .log-time {
        font-family: 'DM Mono', monospace;
        color: #5a6278;
        white-space: nowrap;
        font-size: 11px;
        padding-top: 1px;
    }
    .log-text { color: #94a3b8; }

    /* ── Object Tags ── */
    .obj-tag {
        display: inline-block;
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.2);
        color: #fca5a5;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 99px;
        margin: 2px;
    }

    /* ── Dividers ── */
    hr { border-color: #1e2130 !important; margin: 16px 0 !important; }

    /* ── Controls ── */
    .ctrl-bar {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #13161f;
        border: 1px solid #1e2130;
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 20px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .dot-on  { background: #22c55e; box-shadow: 0 0 6px #22c55e; animation: pulse-dot 2s infinite; }
    .dot-off { background: #ef4444; }
    @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* Streamlit widget overrides */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        border: 1px solid #1e2130 !important;
        padding: 6px 18px !important;
    }
    .stButton > button[kind="primary"] {
        background: #22c55e !important;
        border-color: #22c55e !important;
        color: #0c0e14 !important;
    }
    .stMetric { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session_state():
    defaults = {
        "running": False,
        "overall_risk": 0.0,
        "risk_level": "LOW",
        "emotion_score": 0.0,
        "object_score": 0.0,
        "audio_score": 0.0,
        "stress_score": 0.0,
        "gesture_score": 0.0,
        "heart_rate": 70.0,
        "gsr": 3.0,
        "stress_level": "calm",
        "dominant_emotion": "neutral",
        "detected_objects": [],
        "detected_sounds": [],
        "detected_gestures": [],
        "top_sounds": [],
        "audio_volume": 0.0,
        "trigger_log": deque(maxlen=50),
        "risk_history": deque(maxlen=60),
        "hr_history": deque(maxlen=60),
        "frame_bytes": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ============================================================
# Helpers
# ============================================================
def risk_color(score):
    if score < 30: return "#22c55e"
    if score < 60: return "#f59e0b"
    return "#ef4444"

def risk_tier(score):
    if score < 30: return "low"
    if score < 60: return "medium"
    return "high"

def make_gauge(risk):
    c = risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={"suffix": "%", "font": {"color": c, "size": 42, "family": "DM Mono"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#2a2d3e",
                "tickfont": {"color": "#5a6278", "size": 10},
                "tickwidth": 1,
            },
            "bar": {"color": c, "thickness": 0.6},
            "bgcolor": "#13161f",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "rgba(34,197,94,0.06)"},
                {"range": [30, 60],  "color": "rgba(245,158,11,0.06)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.06)"},
            ],
            "threshold": {
                "line": {"color": c, "width": 2},
                "thickness": 0.8,
                "value": risk,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0", "family": "DM Sans"},
        height=210,
        margin=dict(l=24, r=24, t=24, b=0),
    )
    return fig


def make_sparkline(history, color, y_label=""):
    vals = list(history) or [0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(vals))),
        y=vals,
        mode="lines",
        line=dict(color=color, width=1.8, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=100,
        margin=dict(l=30, r=6, t=4, b=18),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(
            showgrid=True, gridcolor="#1a1d28", zeroline=False,
            tickfont={"color": "#5a6278", "size": 9},
            range=[0, max(max(vals) * 1.25, 10)],
        ),
        showlegend=False,
    )
    return fig


def score_tile(icon, label, score):
    tier = risk_tier(score)
    color = risk_color(score)
    return f"""
    <div class="score-tile score-tile-{tier}">
        <div class="score-label">{icon} {label}</div>
        <div class="score-value" style="color:{color}">{score:.0f}<span style="font-size:14px;color:#5a6278">%</span></div>
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{score}%;background:{color}"></div>
        </div>
    </div>
    """


def stat_pill(label, value, unit=""):
    return f"""
    <div class="stat-pill">
        <div class="stat-pill-label">{label}</div>
        <div class="stat-pill-value">{value}<span style="font-size:11px;color:#5a6278;margin-left:2px">{unit}</span></div>
    </div>
    """


# ============================================================
# Detection Cycle
# ============================================================
def run_detection_cycle(frame, audio_data=None):
    """
    Run one detection cycle on the given frame and optional audio.
    Works on Streamlit Cloud (no hardware needed).

    Args:
        frame: BGR numpy array from camera or uploaded image
        audio_data: optional 1D float32 numpy array of audio samples
    """
    from src.emotion.detector import EmotionDetector
    from src.object_detection.detector import ObjectDetector
    from src.stress.classifier import StressClassifier
    from src.audio.classifier import AudioClassifier
    from src.gesture.detector import GestureDetector

    if "emotion_det" not in st.session_state:
        st.session_state.emotion_det = EmotionDetector()
    if "object_det" not in st.session_state:
        st.session_state.object_det = ObjectDetector()
    if "stress_clf" not in st.session_state:
        st.session_state.stress_clf = StressClassifier()
    if "audio_clf" not in st.session_state:
        st.session_state.audio_clf = AudioClassifier()
    if "gesture_det" not in st.session_state:
        st.session_state.gesture_det = GestureDetector()

    # Emotion
    emo_results = st.session_state.emotion_det.analyze_frame(frame)
    if emo_results:
        st.session_state.emotion_score = max(r["trigger_score"] for r in emo_results)
        st.session_state.dominant_emotion = emo_results[0].get("dominant_emotion", "neutral")
    else:
        st.session_state.emotion_score = 0
        st.session_state.dominant_emotion = "neutral"

    # Objects
    obj_results = st.session_state.object_det.detect_frame(frame)
    st.session_state.object_score = obj_results["trigger_score"]
    st.session_state.detected_objects = [o["name"] for o in obj_results.get("trigger_objects", [])]

    # Audio (from uploaded file or zeroed out)
    if audio_data is not None:
        try:
            st.session_state.audio_volume = st.session_state.audio_clf.get_volume_level(audio_data)
            audio_results = st.session_state.audio_clf.classify_audio(audio_data)
            st.session_state.audio_score = audio_results["trigger_score"]
            st.session_state.top_sounds = audio_results.get("top_sounds", [])[:5]
            st.session_state.detected_sounds = [s["name"] for s in audio_results.get("trigger_sounds", [])]
        except Exception:
            st.session_state.audio_volume = 0
            st.session_state.audio_score = 0
            st.session_state.top_sounds = []
            st.session_state.detected_sounds = []
    else:
        st.session_state.audio_volume = 0
        st.session_state.audio_score = 0
        st.session_state.top_sounds = []
        st.session_state.detected_sounds = []

    # Stress (uses dummy/simulated data on cloud)
    stress_res = st.session_state.stress_clf.predict()
    st.session_state.stress_score = stress_res["trigger_score"]
    st.session_state.heart_rate = stress_res["reading"].get("heart_rate", 70)
    st.session_state.gsr = stress_res["reading"].get("gsr", 3)
    st.session_state.stress_level = stress_res["stress_level"]

    # Gesture — runs on same camera frame
    gesture_results = st.session_state.gesture_det.detect_frame(frame)
    st.session_state.gesture_score = gesture_results["trigger_score"]
    st.session_state.detected_gestures = [
        g["name"] for g in gesture_results.get("gestures", [])
    ]

    # Fused risk (5 modules)
    w = FUSION_WEIGHTS
    total = (
        st.session_state.emotion_score * w["emotion"]
        + st.session_state.object_score * w["object"]
        + st.session_state.audio_score * w["audio"]
        + st.session_state.stress_score * w["stress"]
        + st.session_state.gesture_score * w.get("gesture", 0)
    )
    st.session_state.overall_risk = round(min(max(total, 0), 100), 1)
    st.session_state.risk_level = (
        "LOW" if st.session_state.overall_risk < 30
        else "MEDIUM" if st.session_state.overall_risk < 60
        else "HIGH"
    )

    st.session_state.risk_history.append(st.session_state.overall_risk)
    st.session_state.hr_history.append(st.session_state.heart_rate)

    now = datetime.now().strftime("%H:%M:%S")
    if st.session_state.overall_risk > 50:
        st.session_state.trigger_log.appendleft({
            "type": "risk",
            "time": now,
            "text": f"Risk {st.session_state.overall_risk:.0f}% — {st.session_state.dominant_emotion}, {st.session_state.stress_level}",
        })
    for s in st.session_state.detected_sounds:
        st.session_state.trigger_log.appendleft({
            "type": "audio",
            "time": now,
            "text": f"Sound trigger: {s}",
        })
    for g in st.session_state.detected_gestures:
        st.session_state.trigger_log.appendleft({
            "type": "gesture",
            "time": now,
            "text": f"Gesture: {g}",
        })

    annotated = st.session_state.emotion_det.draw_results(frame, emo_results)
    annotated = st.session_state.object_det.draw_results(annotated, obj_results)
    annotated = st.session_state.gesture_det.draw_results(annotated, gesture_results)
    _, buf = cv2.imencode(".jpg", annotated)
    st.session_state.frame_bytes = buf.tobytes()


# ============================================================
# LAYOUT
# ============================================================

# ── Header ──
col_title, col_controls = st.columns([5, 3])
with col_title:
    is_on = st.session_state.running
    dot_cls = "dot-on" if is_on else "dot-off"
    status_txt = "Active" if is_on else "Idle"
    st.markdown(f"""
    <div style="padding:4px 0 16px">
        <div style="font-size:11px;letter-spacing:0.15em;text-transform:uppercase;
                    color:#5a6278;margin-bottom:4px;">PTSD SENTINEL</div>
        <div style="font-size:26px;font-weight:600;color:#e2e8f0;line-height:1">
            Trigger Detection System
        </div>
        <div style="margin-top:8px;font-size:13px;color:#5a6278">
            <span class="status-dot {dot_cls}"></span>
            {status_txt} &nbsp;·&nbsp; Real-time multi-modal monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_controls:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶  Start", type="primary", width="stretch"):
            st.session_state.running = True
    with c2:
        if st.button("■  Stop", width="stretch"):
            st.session_state.running = False
    with c3:
        if st.button("↺  Reset", width="stretch"):
            st.session_state.running = False
            st.session_state.overall_risk = 0.0
            st.session_state.risk_level = "LOW"
            st.session_state.risk_history.clear()
            st.session_state.hr_history.clear()
            st.session_state.trigger_log.clear()
            st.session_state.frame_bytes = None

# ── Alert Banner ──
rl = st.session_state.risk_level
risk_val = st.session_state.overall_risk
if rl == "HIGH":
    st.markdown(
        f'<div class="alert-high">🚨 &nbsp;<strong>High Risk Detected</strong> &nbsp;—&nbsp; '
        f'Overall score {risk_val:.0f}%. Take slow, deep breaths. Seek support if needed.</div>',
        unsafe_allow_html=True)
elif rl == "MEDIUM":
    st.markdown(
        f'<div class="alert-medium">⚠️ &nbsp;<strong>Elevated Risk</strong> &nbsp;—&nbsp; '
        f'Overall score {risk_val:.0f}%. Monitor situation closely.</div>',
        unsafe_allow_html=True)
elif st.session_state.running:
    st.markdown(
        f'<div class="alert-safe">✓ &nbsp;<strong>Environment Normal</strong> &nbsp;—&nbsp; '
        f'All signals within safe thresholds.</div>',
        unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── ROW 1: Score Tiles ──
sc1, sc2, sc3, sc4 = st.columns(4)
for col, icon, label, score in [
    (sc1, "EMTN", "Emotion",  st.session_state.emotion_score),
    (sc2, "OBJT", "Objects",  st.session_state.object_score),
    (sc3, "AUDI", "Audio",    st.session_state.audio_score),
    (sc4, "STRS", "Stress",   st.session_state.stress_score),
]:
    with col:
        st.markdown(score_tile(icon, label, score), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Quick Stats Row ──
overall_color = risk_color(st.session_state.overall_risk)
st.markdown(f"""
<div style="background:#13161f;border:1px solid #1e2130;border-radius:12px;
            padding:14px 20px;display:flex;gap:0;align-items:center;flex-wrap:wrap;">
    <div style="flex:2;min-width:160px;border-right:1px solid #1e2130;padding-right:20px;margin-right:20px;">
        <div class="score-label">Overall Risk Index</div>
        <div style="font-family:'DM Mono',monospace;font-size:48px;font-weight:500;
                    color:{overall_color};line-height:1;margin:4px 0">
            {st.session_state.overall_risk:.0f}<span style="font-size:20px;color:#5a6278">%</span>
        </div>
        <div style="font-size:12px;color:#5a6278;text-transform:uppercase;
                    letter-spacing:0.1em;">{st.session_state.risk_level}</div>
    </div>
    <div style="flex:1;min-width:90px;text-align:center;padding:0 12px;">
        <div class="score-label">Heart Rate</div>
        <div style="font-family:'DM Mono',monospace;font-size:24px;color:#38b2ff;margin-top:4px">
            {st.session_state.heart_rate:.0f}<span style="font-size:11px;color:#5a6278"> bpm</span>
        </div>
    </div>
    <div style="flex:1;min-width:90px;text-align:center;padding:0 12px;">
        <div class="score-label">GSR</div>
        <div style="font-family:'DM Mono',monospace;font-size:24px;color:#a78bfa;margin-top:4px">
            {st.session_state.gsr:.1f}<span style="font-size:11px;color:#5a6278"> µS</span>
        </div>
    </div>
    <div style="flex:1;min-width:90px;text-align:center;padding:0 12px;">
        <div class="score-label">Emotion</div>
        <div style="font-size:18px;color:#e2e8f0;margin-top:4px;font-weight:500">
            {st.session_state.dominant_emotion.capitalize()}
        </div>
    </div>
    <div style="flex:1;min-width:90px;text-align:center;padding:0 12px;">
        <div class="score-label">Stress Level</div>
        <div style="font-size:18px;color:#e2e8f0;margin-top:4px;font-weight:500">
            {st.session_state.stress_level.replace('_',' ').title()}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── ROW 2: Camera | Gauge+Charts | Audio ──
cam_col, center_col, audio_col = st.columns([3, 2, 2])

with cam_col:
    st.markdown('<div class="sec-header">📷 Camera Input</div>', unsafe_allow_html=True)
    with st.container():
        # Use st.camera_input for cloud-compatible camera access
        camera_photo = st.camera_input("Capture a snapshot for analysis", key="camera_snap")

        if st.session_state.frame_bytes:
            st.markdown('<div class="sec-header" style="margin-top:12px">🔍 Annotated Result</div>', unsafe_allow_html=True)
            st.image(st.session_state.frame_bytes, channels="BGR", use_container_width=True)
            # Detected objects
            if st.session_state.detected_objects:
                tags = "".join(f'<span class="obj-tag">⚠ {o}</span>'
                               for o in st.session_state.detected_objects)
                st.markdown(f'<div style="margin-top:8px">{tags}</div>', unsafe_allow_html=True)

with center_col:
    st.markdown('<div class="sec-header">⚡ Risk Gauge</div>', unsafe_allow_html=True)
    st.plotly_chart(make_gauge(st.session_state.overall_risk),
                    key="gauge", width="stretch")

    st.markdown('<div class="sec-header">📈 Risk Timeline</div>', unsafe_allow_html=True)
    if st.session_state.risk_history:
        st.plotly_chart(make_sparkline(st.session_state.risk_history, "#ef4444"),
                        key="risk_line", width="stretch")
    else:
        st.markdown('<div style="height:100px;background:#13161f;border-radius:8px;'
                    'border:1px solid #1e2130"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-header">💓 Heart Rate</div>', unsafe_allow_html=True)
    if st.session_state.hr_history:
        st.plotly_chart(make_sparkline(st.session_state.hr_history, "#38b2ff"),
                        key="hr_line", width="stretch")
    else:
        st.markdown('<div style="height:100px;background:#13161f;border-radius:8px;'
                    'border:1px solid #1e2130"></div>', unsafe_allow_html=True)

with audio_col:
    st.markdown('<div class="sec-header">🔊 Audio Monitor</div>', unsafe_allow_html=True)

    # Audio file uploader (replaces live microphone for cloud)
    audio_file = st.file_uploader(
        "Upload a .wav file for audio analysis",
        type=["wav"],
        key="audio_upload",
    )

    # Volume bar
    vol = st.session_state.audio_volume
    vc = "#22c55e" if vol < 40 else "#f59e0b" if vol < 70 else "#ef4444"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="font-size:11px;color:#5a6278;letter-spacing:0.08em;text-transform:uppercase">Volume</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;color:{vc}">{vol:.0f}%</span>
    </div>
    <div class="vol-track">
        <div class="vol-fill" style="width:{min(vol,100):.1f}%;background:{vc}"></div>
    </div>
    """, unsafe_allow_html=True)

    # Detected sounds
    if st.session_state.top_sounds:
        for sound in st.session_state.top_sounds:
            name = sound["name"]
            conf = sound["confidence"]
            is_trig = name in st.session_state.detected_sounds
            dot_color = "#ef4444" if is_trig else "#22c55e"
            extra_cls = "sound-row-trigger" if is_trig else ""
            icon = "⚠" if is_trig else ""
            st.markdown(f"""
            <div class="sound-row {extra_cls}">
                <div class="sound-dot" style="background:{dot_color}"></div>
                <div class="sound-name">{icon} {name}</div>
                <div class="sound-conf">{conf:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:24px 0;color:#5a6278;font-size:13px">
            Upload a .wav file to analyze audio
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Trigger alerts inline
    if st.session_state.detected_sounds:
        for s in st.session_state.detected_sounds[:3]:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
                        border-radius:8px;padding:8px 12px;margin:4px 0;font-size:12px;color:#fca5a5">
                🔊 Trigger detected: <strong>{s}</strong>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── ROW 3: Event Log ──
st.markdown('<div class="sec-header">📋 Event Log</div>', unsafe_allow_html=True)

if st.session_state.trigger_log:
    log_items = list(st.session_state.trigger_log)[:10]
    cols = st.columns(2)
    for i, event in enumerate(log_items):
        cls = "log-gesture" if event["type"] == "gesture" else "log-audio" if event["type"] == "audio" else "log-risk"
        icon = "🤜" if event["type"] == "gesture" else "🔊" if event["type"] == "audio" else "⚠️"
        with cols[i % 2]:
            st.markdown(f"""
            <div class="log-item {cls}">
                <span class="log-time">{event['time']}</span>
                <span class="log-text">{icon} {event['text']}</span>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:20px;color:#5a6278;font-size:13px;
                background:#13161f;border:1px dashed #1e2130;border-radius:10px">
        No trigger events yet — start monitoring to detect activity
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Settings expander ──
with st.expander("⚙  System Configuration"):
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        st.markdown("**Module Weights**")
        for mod, w in FUSION_WEIGHTS.items():
            c = risk_color(w * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <div style="width:80px;font-size:12px;color:#94a3b8;text-transform:capitalize">{mod}</div>
                <div style="flex:1;background:#1a1d28;border-radius:99px;height:5px;overflow:hidden">
                    <div style="height:100%;background:{c};width:{w*100:.0f}%;border-radius:99px"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a6278;width:30px;text-align:right">{w:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
    with cfg2:
        st.markdown("**Data Sources**")
        st.markdown(f"""
        <div style="font-size:12px;color:#94a3b8;line-height:2">
            Heart Rate: <code style="color:#38b2ff">{HEART_RATE_SOURCE}</code><br>
            Sample Rate: <code style="color:#38b2ff">{AUDIO_SAMPLE_RATE} Hz</code><br>
            Audio Chunk: <code style="color:#38b2ff">{AUDIO_DURATION}s</code>
        </div>
        """, unsafe_allow_html=True)
    with cfg3:
        st.markdown("**Models**")
        for m in ["DeepFace — Emotion", "YOLOv8 — Objects", "YAMNet — Audio", "Random Forest — Stress"]:
            st.markdown(f"""
            <div style="font-size:12px;color:#94a3b8;padding:3px 0;
                        border-bottom:1px solid #1e2130">{m}</div>
            """, unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:11px;color:#5a6278;margin-top:8px">
            Refresh: {DASHBOARD_REFRESH_RATE}s</div>""", unsafe_allow_html=True)

# ============================================================
# Process Inputs (Cloud-compatible: snapshot + file upload)
# ============================================================
# Decode camera snapshot if available
_frame = None
if camera_photo is not None:
    file_bytes = np.frombuffer(camera_photo.getvalue(), dtype=np.uint8)
    _frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Decode uploaded audio if available
_audio_data = None
if audio_file is not None:
    try:
        from scipy.io import wavfile
        import io
        sr, audio_raw = wavfile.read(io.BytesIO(audio_file.getvalue()))
        # Convert to float32 mono
        if audio_raw.dtype != np.float32:
            audio_raw = audio_raw.astype(np.float32) / np.iinfo(audio_raw.dtype).max
        if len(audio_raw.shape) > 1:
            audio_raw = audio_raw.mean(axis=1)
        # Resample to 16kHz if needed
        if sr != AUDIO_SAMPLE_RATE:
            import librosa
            audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        _audio_data = audio_raw
    except Exception as e:
        st.warning(f"Could not process audio file: {e}")

# Run detection when a camera snapshot is captured
if _frame is not None:
    st.session_state.running = True
    run_detection_cycle(_frame, _audio_data)