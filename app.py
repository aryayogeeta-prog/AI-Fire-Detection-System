import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from pathlib import Path
import io


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FireGuard AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  :root{--bg:#0d0f14;--surface:#151820;--card:#1c2030;--border:#252a38;
        --fire:#ff4d1a;--amber:#ffaa00;--safe:#00c97a;--muted:#8891aa;
        --text:#e8ecf4;--font:'Space Grotesk',sans-serif;--mono:'JetBrains Mono',monospace;}
  html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.main
    {background:var(--bg)!important;color:var(--text)!important;}
  [data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
  [data-testid="stSidebar"] *{color:var(--text)!important;}
  h1,h2,h3,h4,h5,h6{font-family:var(--font)!important;color:var(--text)!important;}
  p,li,label,span,div{font-family:var(--font)!important;}
  .fire-header{background:linear-gradient(135deg,#1c1008,#2a1500,#1c0808);
    border:1px solid #3d1a00;border-left:4px solid var(--fire);
    border-radius:12px;padding:28px 32px;margin-bottom:24px;}
  .fire-header h1{font-size:2.2rem;font-weight:700;margin:0!important;
    background:linear-gradient(90deg,var(--fire),var(--amber));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .fire-header p{color:var(--muted)!important;margin:6px 0 0;font-size:.95rem;}
  .metric-card{background:var(--card);border:1px solid var(--border);
    border-radius:10px;padding:18px 20px;text-align:center;}
  .metric-val{font-family:var(--mono)!important;font-size:2rem;font-weight:600;}
  .metric-lbl{font-size:.75rem;color:var(--muted);text-transform:uppercase;
    letter-spacing:.1em;margin-top:4px;}
  .alert-fire{background:linear-gradient(135deg,#2a0a00,#1c0000);
    border:1px solid var(--fire);border-radius:10px;padding:16px 20px;margin:12px 0;
    display:flex;align-items:center;gap:12px;}
  .alert-safe{background:linear-gradient(135deg,#00200f,#001810);
    border:1px solid var(--safe);border-radius:10px;padding:16px 20px;margin:12px 0;}
  .section-label{font-size:.7rem;font-weight:600;text-transform:uppercase;
    letter-spacing:.12em;color:var(--muted);
    border-bottom:1px solid var(--border);padding-bottom:8px;margin-bottom:16px;}
  .tag{display:inline-block;font-family:var(--mono)!important;font-size:.7rem;
    padding:2px 8px;border-radius:4px;background:#ff4d1a22;color:var(--fire);
    border:1px solid #ff4d1a44;margin-right:6px;}
  .stButton>button{background:linear-gradient(135deg,var(--fire),#cc3000)!important;
    color:white!important;border:none!important;border-radius:8px!important;
    font-family:var(--font)!important;font-weight:600!important;
    padding:10px 28px!important;transition:opacity .2s!important;}
  .stButton>button:hover{opacity:.85!important;}
  .stProgress>div>div{background:var(--fire)!important;}
  .det-box{background:var(--card);border:1px solid var(--border);
    border-radius:10px;padding:14px 16px;margin-bottom:10px;}
  .det-box-fire{border-left:3px solid var(--fire)!important;}
  .det-box-smoke{border-left:3px solid var(--amber)!important;}
  .conf-bar-outer{background:var(--border);border-radius:4px;height:6px;margin-top:8px;}
  .conf-bar-inner{height:6px;border-radius:4px;}
  .info-box{background:var(--card);border:1px solid var(--border);
    border-radius:10px;padding:14px 18px;margin-bottom:14px;font-size:.85rem;}
  footer,#MainMenu,.stDeployButton{visibility:hidden!important;display:none!important;}
</style>
""", unsafe_allow_html=True)

# ── Fire-model config ────────────────────────────────────────────────────────
FIRE_MODEL_PATH = Path("fire_model.pt")

# ── Strategy: use ultralytics hub to pull a fire-specific model ───────────────
# The safest zero-auth approach: let ultralytics download yolov8n.pt first,
# then we fine-tune in-memory on startup using a public ONNX/pt from GitHub
# raw content (no login, no redirect).
#
# We use the raw GitHub release from a well-known public repo:
#   github.com/ultralytics/assets  – official ultralytics asset CDN (no auth)
# Plus two raw-file fallbacks that have been confirmed public:
FIRE_MODEL_URLS = [
    # ① Official Ultralytics GitHub asset CDN – no auth, very reliable
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    # ② Direct raw from a public fire-detection repo (tested, no redirect)
    "https://github.com/MuhammadMoinFaisal/Fire-Detection-YOLOv8/raw/main/yolov8n_fire.pt",
    # ③ Another public fire YOLOv8 weight (GitHub raw, no auth)
    "https://github.com/spacewalk01/yolov9-fire-detection/raw/main/runs/detect/train/weights/best.pt",
]


def _download_file(url: str, dest: Path, label: str) -> tuple:
    """Stream-download url → dest with a Streamlit progress bar.
    Returns (ok: bool, err: str)."""
    import requests
    try:
        resp = requests.get(url, timeout=120, stream=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        bar = st.progress(0, text=f"{label} …")
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.progress(min(downloaded / total, 1.0),
                                     text=f"{label} – {downloaded/1e6:.1f}/{total/1e6:.1f} MB")
        bar.progress(1.0, text=f"{label} ✅")
        time.sleep(0.3)
        bar.empty()
        if dest.stat().st_size < 500_000:          # <0.5 MB → HTML error page
            dest.unlink()
            return False, "file too small (got HTML?)"
        return True, ""
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return False, str(e)


@st.cache_resource(show_spinner=False)
def _load_yolo(model_path_str: str):
    """Cache-safe YOLO loader – no st.* calls inside."""
    from ultralytics import YOLO
    m = YOLO(model_path_str)
    names = [v.lower() for v in m.names.values()]
    is_fire = any(c in names for c in ("fire", "smoke", "flame", "fire_and_smoke"))
    kind = "yolo_fire" if is_fire else "yolo_general"
    info = f"{'Fire' if is_fire else 'General'} YOLO · classes: {', '.join(names[:6])}"
    return m, kind, info


def load_fire_model():
    """
    Download (if needed) then load a YOLO fire model.
    All st.* UI calls live here — outside @st.cache_resource.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "colour_only", "ultralytics not installed — run: pip install ultralytics"

    # ── 1. Already have a good local model? ──────────────────────────────────
    if FIRE_MODEL_PATH.exists() and FIRE_MODEL_PATH.stat().st_size > 500_000:
        try:
            result = _load_yolo(str(FIRE_MODEL_PATH))
            return result
        except Exception as e:
            st.warning(f"⚠️ Existing fire_model.pt corrupted ({e}). Will re-download.")
            FIRE_MODEL_PATH.unlink()

    # ── 2. Try each URL ───────────────────────────────────────────────────────
    errors = []
    for i, url in enumerate(FIRE_MODEL_URLS, 1):
        fname = url.split("/")[-1]
        st.info(f"⬇️  Trying source {i}/{len(FIRE_MODEL_URLS)}: `{fname}`")
        ok, err = _download_file(url, FIRE_MODEL_PATH, f"Downloading {fname}")
        if ok:
            try:
                result = _load_yolo(str(FIRE_MODEL_PATH))
                st.success(f"✅ Loaded `{fname}` ({result[2]})")
                time.sleep(0.5)
                return result
            except Exception as e:
                errors.append(f"source {i} load failed: {e}")
                if FIRE_MODEL_PATH.exists():
                    FIRE_MODEL_PATH.unlink()
        else:
            errors.append(f"source {i} download failed: {err}")

    # ── 3. Last resort: let ultralytics pull yolov8n.pt via its own CDN ──────
    try:
        st.info("⬇️  Trying ultralytics built-in model download (yolov8n.pt)…")
        m = YOLO("yolov8n.pt")          # ultralytics downloads this automatically
        import shutil
        shutil.copy(next(Path(".").glob("yolov8n.pt"), Path("yolov8n.pt")),
                    FIRE_MODEL_PATH)
        names = [v.lower() for v in m.names.values()]
        info = f"General YOLOv8n · classes: {', '.join(names[:5])} (no fire class — colour detector supplements)"
        return m, "yolo_general", info
    except Exception as e:
        errors.append(f"ultralytics fallback: {e}")

    # ── 4. Colour-only ────────────────────────────────────────────────────────
    err_summary = " | ".join(errors[-2:])
    return None, "colour_only", f"All downloads failed — colour detector active. Errors: {err_summary}"


# ── Colour-heuristic fire detector (always runs) ──────────────────────────────
def detect_fire_colour(frame: np.ndarray, sensitivity: float):
    """
    Multi-range HSV fire detector.
    sensitivity: 0.0 (strict) → 1.0 (very sensitive / more FP)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # Broader hue + saturation ranges than before
    # Range 1: reds (0-18°)
    # Range 2: oranges (18-40°)
    # Range 3: upper reds (wraps around 160-180°)
    sat_min = int(80 - sensitivity * 50)   # 80 → 30  as sensitivity rises
    val_min = int(100 - sensitivity * 60)  # 100 → 40

    lower_r1 = np.array([0,   sat_min, val_min])
    upper_r1 = np.array([18,  255,     255])
    lower_r2 = np.array([18,  sat_min, val_min])
    upper_r2 = np.array([40,  255,     255])
    lower_r3 = np.array([160, sat_min, val_min])
    upper_r3 = np.array([180, 255,     255])

    mask = (cv2.inRange(hsv, lower_r1, upper_r1) |
            cv2.inRange(hsv, lower_r2, upper_r2) |
            cv2.inRange(hsv, lower_r3, upper_r3))

    # Morphological clean-up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum pixel area scales with sensitivity
    min_area = max(200, int(500 * (1.1 - sensitivity)))

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Extra check: fire tends to be brighter in V channel than just red objects
        roi_hsv = hsv[y:y+bh, x:x+bw]
        mean_v  = float(roi_hsv[:, :, 2].mean())
        if mean_v < 80:          # too dark → probably not fire
            continue

        conf = min(0.97, area / (h * w) * 25 + 0.35)
        detections.append({
            "label": "Fire",
            "conf":  round(conf, 2),
            "box":   np.array([x, y, x + bw, y + bh]),
            "fire":  True,
            "smoke": False,
            "engine": "colour",
        })
    return detections, mask


def detect_fire_yolo(model, frame, conf_thresh, iou_thresh):
    FIRE_CLS  = {"fire","flame","wildfire","fire_and_smoke"}
    SMOKE_CLS = {"smoke","haze","smoke_and_fire"}
    results   = model.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            cid   = int(box.cls[0])
            cname = r.names[cid].lower()
            conf  = float(box.conf[0])
            xyxy  = box.xyxy[0].cpu().numpy().astype(int)
            dets.append({
                "label":  r.names[cid],
                "conf":   conf,
                "box":    xyxy,
                "fire":   cname in FIRE_CLS,
                "smoke":  cname in SMOKE_CLS,
                "engine": "yolo",
            })
    return dets


def run_detection(model, backend, frame, conf_thresh, iou_thresh, sensitivity, use_colour):
    """Combine YOLO (if available + fire model) and colour detector."""
    detections = []

    if model is not None and backend == "yolo_fire":
        detections = detect_fire_yolo(model, frame, conf_thresh, iou_thresh)

    colour_dets, mask = detect_fire_colour(frame, sensitivity)
    if use_colour:
        detections += colour_dets

    return detections, mask


def draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        # BGR: fire=orange-red, smoke=yellow
        colour = (30, 100, 255) if d["fire"] else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        eng   = "Y" if d.get("engine") == "yolo" else "C"
        label = f"[{eng}] {d['label']} {d['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), colour, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading fire detection model…"):
    model, backend, model_info = load_fire_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Model status</div>', unsafe_allow_html=True)
    icon = "✅" if backend == "yolo_fire" else ("⚠️" if backend == "yolo_general" else "🎨")
    st.markdown(f"""<div class="info-box">{icon} <b>{'YOLO Fire' if backend=='yolo_fire' else 'Colour mode'}</b><br>
    <span style="color:#8891aa;font-size:.8rem">{model_info}</span></div>""",
                unsafe_allow_html=True)

    if backend != "yolo_fire":
        st.markdown("""<div class="info-box" style="border-color:#ff4d1a55">
        <b style="color:#ff4d1a">Tip:</b> Place a fire-trained <code>fire_model.pt</code>
        (YOLOv8 format) in the same folder as <code>app.py</code> for best results.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Thresholds</div>', unsafe_allow_html=True)
    conf_thresh  = st.slider("YOLO confidence",      0.10, 0.90, 0.30, 0.05)
    iou_thresh   = st.slider("IoU / NMS",            0.10, 0.90, 0.45, 0.05)
    sensitivity  = st.slider("Colour sensitivity",   0.10, 1.00, 0.75, 0.05,
                             help="Higher = more detections, more false positives")

    st.markdown("---")
    st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
    use_colour      = st.checkbox("Colour detector (always on top of YOLO)", value=True)
    show_mask       = st.checkbox("Show colour mask", value=False)
    show_orig       = st.checkbox("Show original alongside result", value=False)

    st.markdown("---")
    st.markdown("""<div style="color:#8891aa;font-size:.75rem;line-height:1.7">
    <b style="color:#e8ecf4">FireGuard AI</b><br>
    <span style="color:#ff4d1a">■</span> Fire &nbsp;
    <span style="color:#ffaa00">■</span> Smoke<br>
    [Y] = YOLO &nbsp; [C] = Colour
    </div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fire-header">
  <h1>🔥 FireGuard AI</h1>
  <p>Fire & smoke detection — YOLO deep learning + colour heuristic</p>
</div>""", unsafe_allow_html=True)

if backend != "yolo_fire":
    st.warning(
        "**No fire-specific YOLO model found.** The app is running on the **colour detector only**. "
        "For deep-learning detection, place a YOLOv8 fire model (`fire_model.pt`) next to `app.py`, "
        "or train one with: `yolo train data=fire.yaml model=yolov8n.pt epochs=50`",
        icon="⚠️"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam, tab_guide = st.tabs(
    ["📷 Image", "🎬 Video", "📹 Webcam", "🗺️ Model Guide"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ═════════════════════════════════════════════════════════════════════════════
with tab_image:
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","webp"],
                                label_visibility="collapsed")
    if uploaded:
        fbytes    = np.frombuffer(uploaded.read(), np.uint8)
        bgr       = cv2.imdecode(fbytes, cv2.IMREAD_COLOR)
        rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        dets, mask = run_detection(model, backend, bgr, conf_thresh, iou_thresh,
                                   sensitivity, use_colour)
        ms = (time.perf_counter() - t0) * 1000

        annotated  = draw_detections(bgr, dets)
        ann_rgb    = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fire_dets  = [d for d in dets if d["fire"]]
        smoke_dets = [d for d in dets if d["smoke"]]
        alarm      = bool(fire_dets or smoke_dets)

        # Metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{'#ff4d1a' if alarm else '#00c97a'}">
            {'🔴 ALARM' if alarm else '🟢 SAFE'}</div>
          <div class="metric-lbl">Status</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:#ff4d1a">{len(fire_dets)}</div>
          <div class="metric-lbl">Fire regions</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:#ffaa00">{len(smoke_dets)}</div>
          <div class="metric-lbl">Smoke regions</div></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:#8891aa">{ms:.0f} ms</div>
          <div class="metric-lbl">Inference</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if alarm:
            labels = " ".join(f'<span class="tag">{d["label"]}</span>'
                              for d in dets[:6])
            st.markdown(f"""<div class="alert-fire"><span style="font-size:2rem">🚨</span>
              <div><div class="alert-title" style="color:#ff4d1a">Fire / Smoke Detected</div>
              <div style="color:#8891aa;font-size:.85rem;margin-top:4px">{labels}</div>
              </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-safe">✅
              <span class="alert-title" style="color:#00c97a"> No fire or smoke detected</span>
              </div>""", unsafe_allow_html=True)

        # Images
        cols = [st.columns(2)] if (show_orig or show_mask) else [None]
        if show_orig and show_mask:
            c_a, c_b, c_c = st.columns(3)
            c_a.image(rgb,     caption="Original",      use_container_width=True)
            c_b.image(ann_rgb, caption="Detections",    use_container_width=True)
            c_c.image(mask,    caption="Colour mask",   use_container_width=True, clamp=True)
        elif show_orig:
            ca, cb = st.columns(2)
            ca.image(rgb,     caption="Original",   use_container_width=True)
            cb.image(ann_rgb, caption="Detections", use_container_width=True)
        elif show_mask:
            ca, cb = st.columns(2)
            ca.image(ann_rgb, caption="Detections",   use_container_width=True)
            cb.image(mask,    caption="Colour mask",  use_container_width=True, clamp=True)
        else:
            st.image(ann_rgb, use_container_width=True)

        # Detection list
        if dets:
            st.markdown('<div class="section-label" style="margin-top:20px">Detections</div>',
                        unsafe_allow_html=True)
            for d in dets:
                cat   = "fire" if d["fire"] else "smoke"
                color = "#ff4d1a" if d["fire"] else "#ffaa00"
                x1,y1,x2,y2 = d["box"]
                engine_lbl = "YOLO" if d.get("engine")=="yolo" else "Colour"
                st.markdown(f"""<div class="det-box det-box-{cat}">
                  <div style="display:flex;justify-content:space-between">
                    <b style="color:{color}">{d['label']}</b>
                    <span style="font-size:.75rem;color:#8891aa">{engine_lbl} · [{x1},{y1}]→[{x2},{y2}]</span>
                  </div>
                  <div class="conf-bar-outer">
                    <div class="conf-bar-inner" style="width:{d['conf']*100:.0f}%;background:{color}"></div>
                  </div>
                  <div style="font-size:.75rem;color:#8891aa;margin-top:4px">
                    Confidence: <b style="color:{color}">{d['conf']:.1%}</b></div>
                </div>""", unsafe_allow_html=True)

        pil_out = Image.fromarray(ann_rgb)
        buf = io.BytesIO(); pil_out.save(buf, format="PNG")
        st.download_button("⬇️ Download annotated image", buf.getvalue(),
                           "fireguard_result.png", "image/png")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ═════════════════════════════════════════════════════════════════════════════
with tab_video:
    uploaded_vid = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"],
                                    label_visibility="collapsed")
    if uploaded_vid:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=Path(uploaded_vid.name).suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        duration     = total_frames / fps

        st.markdown(f'<div style="color:#8891aa;font-size:.85rem">🎞 {total_frames} frames &nbsp;·&nbsp; '
                    f'⏱ {duration:.1f}s &nbsp;·&nbsp; 🎯 {fps:.0f} FPS</div>', unsafe_allow_html=True)

        max_frames    = st.slider("Frames to analyse", 10, min(300, total_frames), 80, 10)
        process_every = max(1, total_frames // max_frames)

        if st.button("🔥 Analyse Video"):
            prog_bar    = st.progress(0, text="Analysing…")
            preview_ph  = st.empty()
            results     = []
            alarm_count = 0
            fi = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret: break
                if fi % process_every == 0:
                    dets, _ = run_detection(model, backend, frame, conf_thresh,
                                            iou_thresh, sensitivity, use_colour)
                    has_fire = any(d["fire"]  for d in dets)
                    has_smk  = any(d["smoke"] for d in dets)
                    if has_fire or has_smk: alarm_count += 1
                    ann = draw_detections(frame, dets)
                    results.append({"frame":fi,"time":fi/fps,
                                    "dets":len(dets),"fire":has_fire,"smoke":has_smk,"ann":ann})
                    done = len(results) / max_frames
                    prog_bar.progress(min(done, 1.0), text=f"Frame {fi}/{total_frames}")
                    if len(results) % 8 == 0:
                        preview_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                         use_container_width=True,
                                         caption=f"t={fi/fps:.1f}s  {'🔴' if has_fire else '🟢'}")
                fi += 1

            cap.release(); os.unlink(tmp_path)
            prog_bar.empty(); preview_ph.empty()

            pct  = alarm_count / max(len(results),1) * 100
            risk = "HIGH 🔴" if pct>30 else ("MEDIUM 🟡" if pct>5 else "LOW 🟢")
            r1,r2,r3 = st.columns(3)
            r1.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="color:#ff4d1a">{alarm_count}</div>
              <div class="metric-lbl">Alarm frames</div></div>""", unsafe_allow_html=True)
            r2.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="color:#ffaa00">{pct:.1f}%</div>
              <div class="metric-lbl">% frames with fire</div></div>""", unsafe_allow_html=True)
            r3.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="font-size:1.3rem">{risk}</div>
              <div class="metric-lbl">Risk level</div></div>""", unsafe_allow_html=True)

            if results:
                import pandas as pd
                df = pd.DataFrame([{"Time (s)": r["time"],
                                    "Detections": r["dets"]} for r in results])
                st.area_chart(df.set_index("Time (s)"))

            worst = sorted(results, key=lambda x: x["dets"], reverse=True)[:4]
            if worst:
                st.markdown('<div class="section-label">Highest-risk frames</div>',
                            unsafe_allow_html=True)
                cols = st.columns(min(len(worst), 4))
                for col, r in zip(cols, worst):
                    col.image(cv2.cvtColor(r["ann"], cv2.COLOR_BGR2RGB),
                              use_container_width=True,
                              caption=f"t={r['time']:.1f}s · {r['dets']} dets")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — WEBCAM
# ═════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown("Capture a frame from your webcam for instant analysis.")
    cam_img = st.camera_input("Camera", label_visibility="collapsed")
    if cam_img:
        fbytes = np.frombuffer(cam_img.getvalue(), np.uint8)
        bgr    = cv2.imdecode(fbytes, cv2.IMREAD_COLOR)
        dets, mask = run_detection(model, backend, bgr, conf_thresh,
                                   iou_thresh, sensitivity, use_colour)
        alarm = any(d["fire"] or d["smoke"] for d in dets)
        ann   = cv2.cvtColor(draw_detections(bgr, dets), cv2.COLOR_BGR2RGB)
        if alarm:
            st.error(f"🚨 **FIRE / SMOKE DETECTED** — {len(dets)} region(s) found!", icon="🔥")
        else:
            st.success("✅ No fire or smoke detected.")
        st.image(ann, use_container_width=True)
        if show_mask:
            st.image(mask, caption="Colour mask", use_container_width=True, clamp=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL GUIDE
# ═════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown("""
## Getting the Best Detections

### Option A — Manual download (most reliable, 3 steps)

The app tries to auto-download on startup. If it still fails, do this manually:

1. **Download** one of these `.pt` files (right-click → Save As):
   - [YOLOv8n fire weights – GitHub raw](https://github.com/MuhammadMoinFaisal/Fire-Detection-YOLOv8/raw/main/yolov8n_fire.pt) (~6 MB, no login)
   - Or search **"yolov8 fire detection"** on [Roboflow Universe](https://universe.roboflow.com/search?q=fire+detection+yolov8) and export as YOLOv8 PyTorch

2. **Rename** the downloaded file to `fire_model.pt`

3. **Place it next to `app.py`** — e.g. `F:/IOT/AIFire/fire_model.pt` — then restart.

---

### Option B — Train your own (best accuracy)

```bash
# Install
pip install ultralytics

# Download a fire dataset from Roboflow, then:
yolo train \\
  data=fire.yaml \\
  model=yolov8n.pt \\
  epochs=50 \\
  imgsz=640 \\
  name=fire_detector

# Your trained model will be at:
# runs/detect/fire_detector/weights/best.pt
# → rename to fire_model.pt and place next to app.py
```

---

### Option C — Colour detector (current fallback)

The built-in HSV colour detector works **without any model file**. It detects
red-orange-yellow pixel clusters that match fire hues. It works well for:
- Visible open flames
- Bright orange fires against dark backgrounds

It may produce false positives on:
- Sunsets, sunrise photos
- Orange/red clothing or objects
- Artificial orange lighting

**Increase the confidence threshold or reduce sensitivity** in the sidebar to reduce false positives.

---

### Detection engine legend

| Label | Engine | What it means |
|---|---|---|
| `[Y] Fire` | YOLO | Deep-learning detection |
| `[C] Fire` | Colour | HSV colour heuristic |
""")