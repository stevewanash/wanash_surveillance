import streamlit as st
import cv2
import numpy as np
import os
import face_recognition
from ultralytics import YOLO
import datetime
import csv
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Wanash Guardian Node",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Dark Mode" Professional Look
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Trassir_logo.svg/1200px-Trassir_logo.svg.png", width=200) # Optional: Replace with local logo
st.sidebar.title("⚙️ Control Panel")

# Toggle Source
app_mode = st.sidebar.selectbox("Select Input Source", ["Webcam", "Video File"])
video_file_path = "laydownvideo.mp4" # Ensure this file exists!

# Interactive Sliders
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
crowd_threshold = st.sidebar.number_input("Crowd Limit Alert", min_value=1, value=5)
enable_face_rec = st.sidebar.checkbox("Enable Face Recognition", value=True)

# Optimization Settings (Hidden in an expander to keep UI clean)
with st.sidebar.expander("🚀 Performance Tuning"):
    frame_skip = st.slider("Frame Skip (Motion)", 1, 10, 4)
    face_skip = st.slider("Frame Skip (Face Rec)", 1, 60, 30)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE**")

# --- GLOBAL CONSTANTS ---
STERILE_ZONE = np.array([[100, 100], [600, 100], [600, 400], [100, 400]], np.int32)
STERILE_ZONE_COLOR = (255, 0, 0)
LOG_FILE = 'security_log.csv'

# --- 1. CACHED MODEL LOADING (Prevents reloading on interaction) ---
@st.cache_resource
def load_models():
    print("Loading Models...")
    pose = YOLO('yolov8n-pose.pt')
    obj = YOLO('yolov8n.pt')
    return pose, obj

pose_model, object_model = load_models()

# --- FACE DATA LOADING ---
known_faces, known_names, face_roles = [], [], []

def load_faces(folder, role):
    if not os.path.exists(folder): return
    for f in os.listdir(folder):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            try:
                img = face_recognition.load_image_file(os.path.join(folder, f))
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_faces.append(enc[0])
                    known_names.append(os.path.splitext(f)[0])
                    face_roles.append(role)
            except Exception as e:
                pass

# Load faces only once
if not known_faces:
    load_faces("authorized", "authorized")
    load_faces("blacklist", "blacklist")

# --- HELPER FUNCTIONS ---
def log_alert(alert_list):
    if not alert_list: return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    is_new = not os.path.exists(LOG_FILE)
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new: writer.writerow(['Time', 'Alert', 'Status'])
            for alert in alert_list:
                writer.writerow([timestamp, alert, 'ACTIVE'])
    except: pass

def check_zone_intrusion(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# --- MAIN DASHBOARD LAYOUT ---
st.title("🛡️ Wanash National Guardian Node")

col_video, col_stats = st.columns([3, 1.5])

with col_stats:
    st.subheader("📊 Live Telemetry")
    # Create empty placeholders for metrics so we can update them fast
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        metric_people = st.empty()
    with kpi2:
        metric_status = st.empty()
    
    st.divider()
    st.subheader("🚨 Activity Log")
    log_placeholder = st.empty()

with col_video:
    st_frame = st.empty() # The video window

# --- VIDEO PROCESSING LOOP ---
start_button = st.button("▶️ START SURVEILLANCE")

if start_button:
    # Initialize Video
    if app_mode == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file_path)
    
    cap.set(3, 1280)
    cap.set(4, 720)

    frame_count = 0
    last_person_count = 0
    last_alerts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if app_mode == "Video File": cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else: break
            continue

        frame_count += 1
        
        # --- 1. AI PROCESSING (Skipped Logic) ---
        if frame_count % frame_skip == 0:
            alerts = []
            
            # Resize for processing speed
            process_frame = cv2.resize(frame, (640, 360))
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 360

            # A. POSE ANALYTICS
            pose_results = pose_model.track(process_frame, persist=True, conf=conf_threshold, verbose=False)
            if pose_results[0].boxes.id is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                
                for box, kpts in zip(boxes, keypoints):
                    # Scale up for drawing
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    
                    # Hands Up Logic
                    rw_y, lw_y = kpts[[10, 9], 1]
                    re_y, le_y = kpts[[4, 3], 1]
                    if all(kpts[[9, 10], 1] > 0) and (rw_y < re_y and lw_y < le_y):
                        if "HANDS UP (ROBERRY?)" not in alerts: alerts.append("HANDS UP (ROBBERY?)")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                        #cv2.putText(frame, "HANDS UP", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                    # Fall Logic
                    bbox_w = x2 - x1; bbox_h = y2 - y1
                    head_y = kpts[0][1] * scale_y
                    hip_y = ((kpts[11][1] + kpts[12][1]) / 2) * scale_y
                    
                    if (bbox_w > bbox_h * 1.5) or (hip_y < head_y and head_y != 0):
                        if "MAN DOWN (FALL)" not in alerts: alerts.append("MAN DOWN (FALL)")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        #cv2.putText(frame, "FALL DETECTED", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # B. OBJECT DETECTION
            obj_results = object_model.predict(process_frame, conf=conf_threshold, verbose=False)
            person_count = 0
            
            for r in obj_results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1 = int(coords[0] * scale_x)
                    y1 = int(coords[1] * scale_y)
                    x2 = int(coords[2] * scale_x)
                    y2 = int(coords[3] * scale_y)
                    
                    cls = int(box.cls[0])
                    label = object_model.names[cls]
                    
                    if label == 'person':
                        person_count += 1
                        center = (int((x1+x2)/2), int(y2))
                        if check_zone_intrusion(center, STERILE_ZONE):
                            if "ZONE INTRUSION" not in alerts: alerts.append("ZONE INTRUSION")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                            #cv2.putText(frame, "RESTRICTED", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    elif label in ['backpack', 'suitcase', 'car']:
                        if "STRAY OBJECT" not in alerts: alerts.append("STRAY OBJECT")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            last_person_count = person_count
            last_alerts = alerts
            log_alert(alerts)

        else:
            alerts = last_alerts
            person_count = last_person_count

        # C. FACE RECOGNITION (Independent Skip)
        if enable_face_rec and frame_count % face_skip == 0:
            # Face rec logic here (Drawing directly on frame)
            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            
            # Note: We aren't caching face boxes between frames to keep code simple
            # But we will draw them this one time
            for (top, right, bottom, left), enc in zip(locs, encs):
                top*=4; right*=4; bottom*=4; left*=4
                matches = face_recognition.compare_faces(known_faces, enc, tolerance=0.55)
                if True in matches:
                    idx = np.argmin(face_recognition.face_distance(known_faces, enc))
                    name = known_names[idx]
                    role = face_roles[idx]
                    color = (0, 0, 255) if role == "blacklist" else (0, 255, 0)
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, f"{name} ({role})", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if role == "blacklist" and f"BLACKLIST: {name}" not in alerts:
                        alerts.append(f"BLACKLIST: {name}")
        # 1. Create a semi-transparent black overlay for background readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, 720), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # 2. Draw PEOPLE COUNT metric
        y_ui = 50
        # White text for count
        cv2.putText(frame, f"PEOPLE COUNT: {person_count}", (20, y_ui), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 3. Draw SYSTEM STATUS and ALERT LIST
        y_ui += 50
        if alerts:
            # Red text for system status
            cv2.putText(frame, "SYSTEM STATUS: ALERT!", (20, y_ui), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) 
            
            # List all active alerts
            for alert in alerts:
                y_ui += 40
                cv2.putText(frame, f"• {alert}", (20, y_ui), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Green text for normal status
            cv2.putText(frame, "SYSTEM STATUS: NORMAL", (20, y_ui), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- 2. UPDATE DASHBOARD ---
        
        # Draw Zone
        cv2.polylines(frame, [STERILE_ZONE], True, STERILE_ZONE_COLOR, 2)

        # Display Video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame, channels="RGB", use_column_width=True)

        # Update Metrics
        metric_people.metric("People Count", person_count, delta_color="inverse")
        
        status_state = "CRITICAL" if alerts else "NORMAL"
        metric_status.metric("System Status", status_state)

        # Update Log Table (Read from CSV)
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            # Show last 5 alerts, sorted newest first
            log_placeholder.dataframe(df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)

    cap.release()