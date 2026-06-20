import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os

try:
    import winsound
    WINDOWS = True
except:
    WINDOWS = False

st.set_page_config(
    page_title="AI Fire Detection",
    page_icon="🔥",
    layout="wide"
)

# ------------------------------
# DARK THEME CSS
# ------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
    color:white !important;
}

/* Main Background */
.stApp{
    background:linear-gradient(
        135deg,
        #020617,
        #0F172A,
        #1E293B
    );
}

/* Hide Streamlit Branding */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#0F172A;
    border-right:1px solid #334155;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Title */
.main-title{
    text-align:center;
    font-size:50px;
    font-weight:800;
    color:white !important;
    letter-spacing:2px;
    margin-bottom:20px;
}

/* KPI Cards */
.kpi-card{
    border-radius:20px;
    padding:25px;
    text-align:center;
    min-height:150px;
    color:white !important;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    box-shadow:0 10px 30px rgba(0,0,0,.35);
    transition:all .3s ease;
}

.kpi-card:hover{
    transform:translateY(-5px);
}

.status-card{
    background:linear-gradient(135deg,#3B82F6,#1D4ED8);
}

.conf-card{
    background:linear-gradient(135deg,#8B5CF6,#6D28D9);
}

.alert-card{
    background:linear-gradient(135deg,#EF4444,#991B1B);
}

.fps-card{
    background:linear-gradient(135deg,#10B981,#047857);
}

.kpi-title{
    color:white !important;
    font-size:18px;
    font-weight:700;
    letter-spacing:1px;
    text-transform:uppercase;
}

.kpi-value{
    color:white !important;
    font-size:42px;
    font-weight:800;
    margin-top:10px;
}

/* Streamlit Metrics */
[data-testid="metric-container"]{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:15px;
    box-shadow:0 5px 15px rgba(0,0,0,.25);
}

[data-testid="metric-container"] *{
    color:white !important;
}

[data-testid="stMetricLabel"]{
    color:white !important;
    font-weight:700 !important;
}

[data-testid="stMetricValue"]{
    color:white !important;
    font-size:40px !important;
    font-weight:800 !important;
}

/* Alert Boxes */
.fire-alert{
    background:linear-gradient(135deg,#EF4444,#991B1B);
    color:white !important;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:26px;
    font-weight:700;
    animation:pulse 1s infinite;
}

.safe-alert{
    background:linear-gradient(135deg,#22C55E,#166534);
    color:white !important;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:700;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

[data-testid="stDataFrame"] *{
    color:white !important;
}

/* Buttons */
.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:700;
    color:white !important;
    background:linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    );
    transition:all .3s ease;
}

.stButton > button:hover{
    transform:scale(1.03);
}

/* Slider */
.stSlider *{
    color:white !important;
}

/* Checkbox */
.stCheckbox *{
    color:white !important;
}

/* General Text */
p,h1,h2,h3,h4,h5,h6,span,label,div{
    color:white !important;
}

/* Pulse Animation */
@keyframes pulse{
    0%{transform:scale(1);}
    50%{transform:scale(1.03);}
    100%{transform:scale(1);}
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
text-align:center;
font-size:45px;
font-weight:800;
color:white;
padding:10px;
letter-spacing:2px;">
 AI FIRE DETECTION SYSTEM
</div>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL
# ------------------------------
model = YOLO("best.pt")

# ------------------------------
# SESSION STATE
# ------------------------------
if "alerts" not in st.session_state:
    st.session_state.alerts = 0

if "logs" not in st.session_state:
    st.session_state.logs = []

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:

    st.header("⚙ Controls")

    confidence_threshold = st.slider(
        "Confidence",
        0.0,
        1.0,
        0.5
    )

    enable_alarm = st.checkbox(
        "Enable Alarm",
        True
    )

    save_snapshot = st.checkbox(
        "Save Snapshot",
        True
    )

    start = st.button(
        "▶ Start Detection"
    )

# ------------------------------
# KPI CARDS
# ------------------------------
c1,c2,c3,c4 = st.columns(4)

status_metric = c1.empty()
conf_metric = c2.empty()
alert_metric = c3.empty()
fps_metric = c4.empty()

status_metric.metric("Status","Waiting")
conf_metric.metric("Confidence","0%")
alert_metric.metric("Alerts",0)
fps_metric.metric("FPS",0)

# ------------------------------
# PLACEHOLDERS
# ------------------------------
video_placeholder = st.empty()
alert_placeholder = st.empty()
table_placeholder = st.empty()

os.makedirs("snapshots", exist_ok=True)

# ------------------------------
# START CAMERA
# ------------------------------
if start:

    cap = cv2.VideoCapture(1)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        fire_detected = False
        max_conf = 0

        for r in results:

            if len(r.boxes) > 0:

                confs = r.boxes.conf.cpu().numpy()

                if len(confs) > 0:

                    max_conf = max(confs)

                    if max_conf >= confidence_threshold:
                        fire_detected = True

        annotated = results[0].plot()

        video_placeholder.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        if fire_detected:

            st.session_state.alerts += 1

            status_metric.metric(
                "Status",
                "FIRE"
            )

            conf_metric.metric(
                "Confidence",
                f"{max_conf*100:.2f}%"
            )

            alert_metric.metric(
                "Alerts",
                st.session_state.alerts
            )

            alert_placeholder.error(
                f"🔥 FIRE DETECTED ({max_conf*100:.2f}%)"
            )

            if enable_alarm and WINDOWS:
                winsound.Beep(2500, 400)

            if save_snapshot:

                filename = (
                    f"snapshots/fire_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

            st.session_state.logs.append([
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Fire",
                round(max_conf*100,2)
            ])

        else:

            status_metric.metric(
                "Status",
                "SAFE"
            )

            conf_metric.metric(
                "Confidence",
                "0%"
            )

            alert_placeholder.success(
                "✅ No Fire Detected"
            )

        # ------------------------------
        # LOG TABLE
        # ------------------------------
        if len(st.session_state.logs) > 0:

            df = pd.DataFrame(
                st.session_state.logs,
                columns=[
                    "Timestamp",
                    "Status",
                    "Confidence"
                ]
            )

            table_placeholder.dataframe(
                df.tail(20),
                use_container_width=True
            )

            df.to_csv(
                "fire_log.csv",
                index=False
            )

    cap.release()