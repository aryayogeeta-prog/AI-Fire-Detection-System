
# =========================================================
# INSTALLATION
# =========================================================
#
# pip install streamlit
# pip install torch torchvision torchaudio
# pip install opencv-python
# pip install numpy==1.26.4
# pip install pyttsx3
#
# =========================================================
# DOWNLOAD YOLOv5
# =========================================================
#
# git clone https://github.com/ultralytics/yolov5
#
# Put fire.pt inside:
#
# yolov5/fire.pt
#
# =========================================================
# RUN
# =========================================================
#
# streamlit run main.py
#
# =========================================================

import streamlit as st
import cv2
import numpy as np
import torch
import pyttsx3
import time
import threading
from playsound import playsound
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fire Detection",
    layout="wide"
)
# =========================================================
# PROFESSIONAL UI CSS
# =========================================================

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #f4f7fb;
}

/* Remove default padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Title */
h1 {
    color: #0f172a;
    font-weight: 700;
    text-align: center;
    padding-bottom: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    color: white;
    border-right: 2px solid #334155;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Slider */
.stSlider > div > div {
    color: #ef4444 !important;
}

/* Checkbox */
.stCheckbox {
    padding-top: 10px;
    padding-bottom: 10px;
}

/* Camera Frame */
[data-testid="stImage"] img {
    border-radius: 20px;
    border: 4px solid #1e293b;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
}

/* Status Messages */
.stAlert {
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
}

/* Graph Card */
.element-container:has(canvas) {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Footer */
footer {
    visibility: hidden;
}

/* Custom Footer */
.custom-footer {
    text-align: center;
    padding: 15px;
    color: #64748b;
    font-size: 14px;
}

/* Webcam Started Success */
.stSuccess {
    border-left: 5px solid #22c55e;
}

/* Error Alert */
.stError {
    border-left: 5px solid #ef4444;
}

/* Warning Alert */
.stWarning {
    border-left: 5px solid #f59e0b;
}

/* Info Alert */
.stInfo {
    border-left: 5px solid #3b82f6;
}

/* Smooth UI */
* {
    transition: all 0.2s ease-in-out;
}

/* Professional Card Effect */
.css-1r6slb0 {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)
# =========================================================
# TITLE
# =========================================================

st.title(" AI Based Fire Detection System")



# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Fire Confidence",
    0.1,
    1.0,
    0.3
)

run = st.sidebar.checkbox("Start Camera")

# =========================================================
# DISPLAY PLACEHOLDERS
# =========================================================

FRAME_WINDOW = st.image([])

status_text = st.empty()

severity_text = st.empty()

graph_placeholder = st.empty()

# =========================================================
# LOAD YOLOv5 MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = torch.hub.load(
        './yolov5',
        'custom',
        path='./yolov5/fire.pt',
        source='local',
        force_reload=True
    )

    return model

model = load_model()

# =========================================================
# VOICE ENGINE
# =========================================================

engine = pyttsx3.init()

engine.setProperty('rate', 150)

engine.setProperty('volume', 1.0)

# =========================================================
# NON-BLOCKING VOICE ALERT
# =========================================================

def speak_alert(message):

    threading.Thread(
        target=lambda: (
            engine.say(message),
            engine.runAndWait()
        )
    ).start()
# =========================================================
# ALARM SOUND
# =========================================================

def play_alarm():

    threading.Thread(
        target=lambda: playsound("alert.mp3"),
        daemon=True
    ).start()
# =========================================================
# FIRE DETECTION FUNCTION
# =========================================================

def detect_fire(frame):

    results = model(frame)

    fire_detected = False

    fire_count = 0

    max_confidence = 0

    detections = results.xyxy[0]

    for detection in detections:

        x1, y1, x2, y2, conf, cls = detection

        conf = float(conf)

        if conf >= confidence_threshold:

            fire_detected = True

            fire_count += 1

            if conf > max_confidence:

                max_confidence = conf

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # =====================================
            # DRAW BOX
            # =====================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            # =====================================
            # LABEL
            # =====================================

            label = f"FIRE {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    return fire_detected, fire_count, max_confidence, frame

# =========================================================
# FIRE SEVERITY
# =========================================================

def get_severity(confidence):

    if confidence > 0.85:

        return "HIGH", "Warning. High severity fire detected"

    elif confidence > 0.70:

        return "MEDIUM", "Warning. Medium severity fire detected"

    elif confidence > 0.50:

        return "LOW", "Warning. Low severity fire detected"

    else:

        return "NORMAL", "No fire detected"

# =========================================================
# GRAPH DATA
# =========================================================

severity_history = []

# =========================================================
# ALERT VARIABLES
# =========================================================

last_alert_time = 0

last_fire_detected_time = 0

# =========================================================
# START CAMERA
# =========================================================

if run:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error("❌ Cannot Open Webcam")

    else:

        st.success("✅ Webcam Started")

        while run:

            ret, frame = cap.read()

            if not ret:

                st.error("❌ Failed to Read Webcam")
                break

            # =====================================
            # RESIZE FRAME
            # =====================================

            frame = cv2.resize(
                frame,
                (640, 480)
            )

            # =====================================
            # FIRE DETECTION
            # =====================================

            fire_detected, fire_count, max_confidence, frame = detect_fire(frame)

            # =====================================
            # SEVERITY
            # =====================================

            severity, voice_message = get_severity(max_confidence)

            # =====================================
            # GRAPH VALUE
            # =====================================

            if fire_detected:

                severity_value = max_confidence * 100

            else:

                severity_value = 0

            severity_history.append(severity_value)

            if len(severity_history) > 30:

                severity_history.pop(0)

            # =====================================
            # ALERT MEMORY
            # =====================================

            current_time = time.time()

            if fire_detected:

                last_fire_detected_time = current_time

                # =================================
                # VOICE ALERT EVERY 5 SEC
                # =================================

                if current_time - last_alert_time > 5:

                    speak_alert(voice_message)
                    play_alarm()

                    last_alert_time = current_time

            # =====================================
            # SHOW ALERT FOR 5 SECONDS
            # =====================================

            if current_time - last_fire_detected_time < 5:

                # ALERT BORDER
                cv2.rectangle(
                    frame,
                    (10, 10),
                    (630, 470),
                    (0, 0, 255),
                    5
                )

                # ALERT TEXT
                cv2.putText(
                    frame,
                    f"FIRE DETECTED - {severity}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # STATUS
                status_text.error(
                    f" FIRE DETECTED ({severity})"
                )

                severity_text.warning(
                    f" Severity Level: {severity}"
                )

            else:

                status_text.success(
                    "✅ NORMAL CONDITION"
                )

                severity_text.success(
                    "✅ No Fire"
                )

            # =====================================
            # SHOW FRAME
            # =====================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            FRAME_WINDOW.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            # =====================================
            # FAST REAL-TIME GRAPH
            # =====================================

            graph_placeholder.line_chart(
                severity_history
            )

        cap.release()

else:

    st.info("📷 Click Start Camera")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "##  AI Fire Severity Monitoring System"
)