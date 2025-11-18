import cv2
import numpy as np
import os
import face_recognition
from ultralytics import YOLO
import datetime
import csv

# --- CONFIGURATION & GLOBAL STATE ---

# Primary Settings
USE_WEBCAM = False        
VIDEO_PATH = "laydownvideo.mp4" 
CONFIDENCE_THRESHOLD = 0.6
CROWD_THRESHOLD = 5      

# OPTIMIZATION PARAMETERS (ADJUST THESE TO BALANCE SPEED/ACCURACY)
FRAME_SKIP = 4             # Process YOLO models only every 3rd frame (50-66% faster)
FACE_REC_SKIP = 20        # Run Face Recognition only every 15th frame (Huge speed boost)
YOLO_ANALYTICS_SIZE = 640  # Resize frame to 640x360 before feeding into YOLO models (Faster processing)

# Zone Definition (ADJUST THESE COORDS!)
# NOTE: These coordinates are for the ORIGINAL frame size (e.g., 1280x720)
STERILE_ZONE = np.array([[100, 100], [600, 100], [600, 400], [100, 400]], np.int32)
STERILE_ZONE_COLOR = (255, 0, 0) 

# --- LOGGING FUNCTION ---
LOG_FILE = 'security_log.csv'

def log_alerts_to_csv(alert_list):
    if not alert_list:
        return
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_new_file = not os.path.exists(LOG_FILE)
    
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new_file:
                # Write header row only if the file is new
                writer.writerow(['Timestamp', 'Alert Type', 'Severity'])
                
            for alert in alert_list:
                 # Print to terminal AND write to CSV
                 print(f"🚨 ALERT [{timestamp}]: {alert}")
                 writer.writerow([timestamp, alert, 'HIGH'])
                 
    except Exception as e:
        print(f"Error writing to log file: {e}")

# --- 1. LOAD MODELS & DATA ---

print("⏳ Loading AI Models and Face Data...")

# Model 1: Pose Estimation (Specialized for Human Behavior)
pose_model = YOLO('yolov8n-pose.pt') 

# Model 2: General Object Detection (For Cars, Bags, etc.)
object_model = YOLO('yolov8n.pt')

# Load Face Databases (Unchanged logic)
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
                print(f"Could not load face file {f}: {e}")

load_faces("authorized", "authorized")
load_faces("blacklist", "blacklist")
print(f"Loaded {len(known_faces)} known faces.")

# --- 2. CORE ANALYTICS FUNCTIONS ---

def check_zone_intrusion(point, polygon):
    # Checks if a person's center point is inside the sterile zone 
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def process_pose_analytics(frame, results, alerts, scale_factor):
    # Runs the behavioral analytics for falling and hands-up
    
    if results[0].boxes.id is None:
        return frame, alerts 

    boxes = results[0].boxes.xyxy.cpu().numpy()
    keypoints = results[0].keypoints.xy.cpu().numpy()
    
    for box, kpts in zip(boxes, keypoints):
        # Scale coordinates back up to original frame size for accurate drawing
        x1, y1, x2, y2 = (box / scale_factor).astype(int)
        color = (0, 255, 0) 
        
        # Calculate BBox properties
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Scale keypoints back up for logic checks and drawing
        kpts = kpts / scale_factor

        # --- HANDS UP (Robbery/Surrender) ---
        rw_y, lw_y = kpts[[10, 9], 1]
        re_y, le_y = kpts[[4, 3], 1]
        
        if all(kpts[[9, 10], 1] > 0): 
            if (rw_y < re_y and lw_y < le_y):
                if "HANDS UP (ROBBERY?)" not in alerts: alerts.append("HANDS UP (ROBBERY?)")
                color = (0, 165, 255) 

        # --- FALL DETECTION (Medical Emergency) ---
        head_y = kpts[0][1]
        hip_y = (kpts[11][1] + kpts[12][1]) / 2
        
        if (bbox_w > bbox_h * 1.5) or (hip_y < head_y and head_y != 0): 
             if "MAN DOWN (FALL)" not in alerts: alerts.append("MAN DOWN (FALL)")
             color = (0, 0, 255) 
        
        # Draw BBox and Keypoints
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for kp in kpts: 
            if kp[0] > 0: cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)

    return frame, alerts


def process_face_recognition(frame, alerts):
    # Runs face recognition on a reduced size frame (x1/4)
    
    # Use small_frame (x1/4 of original) for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)
    
    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
        # Scale coordinates back up (x4)
        top*=4; right*=4; bottom*=4; left*=4
        
        name = "Unknown"
        role_color = (128, 128, 128) 

        matches = face_recognition.compare_faces(known_faces, enc, tolerance=0.55)

        if True in matches:
            match_idx = np.argmin(face_recognition.face_distance(known_faces, enc))
            name = known_names[match_idx]
            role = face_roles[match_idx]
            
            if role == "blacklist":
                role_color = (0, 0, 255)
                if f"BLACKLIST: {name}" not in alerts: alerts.append(f"BLACKLIST: {name}")
            elif role == "authorized":
                role_color = (0, 255, 0) 

        # Draw Face Box & Label
        cv2.rectangle(frame, (left, top), (right, bottom), role_color, 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, role_color, 2)

    return frame, alerts

# --- 3. MAIN LOOP EXECUTION ---

# Initialize video capture 
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
last_person_count = 0
last_alerts = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        if not USE_WEBCAM: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1
    
    # 1. --- A. FRAME SKIP CHECK (Controls heavy AI processing) ---
    if frame_count % FRAME_SKIP == 0:
        alerts = [] 

        # Resize frame for YOLO processing 
        YOLO_PROCESS_SIZE = YOLO_ANALYTICS_SIZE 
        process_frame = cv2.resize(frame, (YOLO_PROCESS_SIZE, int(frame.shape[0] * YOLO_PROCESS_SIZE / frame.shape[1])))
        scale_factor = frame.shape[0] / process_frame.shape[0]

        # 1. RUN BEHAVIORAL ANALYTICS (POSE MODEL)
        pose_results = pose_model.track(process_frame, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        frame, alerts = process_pose_analytics(frame, pose_results, alerts, scale_factor)

        # 2. RUN GENERAL OBJECT DETECTION (OBJECT MODEL)
        object_results = object_model.predict(process_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        person_count = 0
        
        for r in object_results:
            for box in r.boxes:
                # FIX APPLIED: Convert Tensor to Numpy before scaling
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (coords * scale_factor).astype(int)
                
                cls = int(box.cls[0])
                label = object_model.names[cls]
                
                if label == 'person':
                    person_count += 1
                    center = (int((x1+x2)/2), int(y2))
                    
                    if check_zone_intrusion(center, STERILE_ZONE):
                        if "RESTRICTED ZONE" not in alerts: alerts.append("RESTRICTED ZONE")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4) 
                        cv2.putText(frame, "ZONE BREACH", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif label in ['backpack', 'suitcase', 'car', 'bicycle']: 
                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) 
                     cv2.putText(frame, label.upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 3. LOGGING: Store current alert state and print to terminal/CSV
        log_alerts_to_csv(alerts)
        last_person_count = person_count
        last_alerts = alerts
    
    # 1. --- B. USE LAST PROCESSED RESULTS FOR SKIPPED FRAMES ---
    else:
        alerts = last_alerts
        person_count = last_person_count


    # 2. RUN FACE RECOGNITION (Independent Library - Skip frames here too!)
    if frame_count % FACE_REC_SKIP == 0:
        frame, alerts = process_face_recognition(frame, alerts)
    
    # 3. FINAL POST-PROCESSING
    if person_count > CROWD_THRESHOLD:
        alerts.append(f"CROWD LIMIT EXCEEDED ({person_count})")
    
    # --- DASHBOARD OVERLAY (Unchanged) ---
    
    cv2.polylines(frame, [STERILE_ZONE], True, STERILE_ZONE_COLOR, 2)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, 720), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    y_ui = 50
    cv2.putText(frame, f"PEOPLE COUNT: {person_count}", (20, y_ui), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    y_ui += 50
    if alerts:
        cv2.putText(frame, "SYSTEM STATUS: ALERT!", (20, y_ui), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for alert in alerts:
            y_ui += 40
            cv2.putText(frame, f"• {alert}", (20, y_ui), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        y_ui += 40
        cv2.putText(frame, "SYSTEM STATUS: NORMAL", (20, y_ui), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("TRASSIR INTEGRATED PROTOTYPE", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()