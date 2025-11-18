# Wanash_surveillance
An AI-assisted surveillance system combining Pose (Fall/Hands Up), Object, and Face Recognition to deliver real-time security, perimeter control, and identity verification on a single dashboard, mirroring advanced VMS capabilities.

# 🛡️ Wanash Guardian Node: Integrated AI Security MVP

This repository serves as a proof-of-concept (PoC) demonstrating how commercial **Video Management System (VMS)** capabilities can be deployed using lightweight, open-source technology.

The project combines specialized deep learning models and custom Python logic to deliver real-time security and behavioral anomaly detection, mirroring advanced enterprise solutions.

---

## 🚀 Core Value Proposition

The **Wanash Guardian Node** provides a unified surveillance and security platform, addressing critical national needs:

* **Behavioral Safety:** Instant alerts for medical emergencies (**Man Down**) and potential threats (**Hands Up** posture detection).
* **Perimeter Security:** Monitors custom defined polygonal boundaries for **Zone Intrusion** alerts.
* **Identity & Access Control:** Uses facial recognition to identify **Blacklisted Individuals** and verify **Authorized Personnel** in real-time.
* **Audit & Reporting:** Automatically generates auditable **CSV records** of all security events for compliance and post-incident analysis.

---

## 🧠 Technical Architecture & Components

The prototype uses a **Two-Model Strategy** to achieve high speed and detection fidelity on standard CPU hardware.

| Component | Technology | Role in the MVP |
| :--- | :--- | :--- |
| **Video Engine** | Python, OpenCV | Handles video stream capture and basic frame processing. |
| **Behavioral AI** | YOLOv8-Pose (`-pose.pt`) | Analyzes human skeletons to trigger alerts for **Falls** and **Hands Up** gestures. |
| **Object AI** | YOLOv8 (Standard `n.pt`) | Detects general objects (cars, backpacks, suitcases) and is used for **Person Counting** |
| **Identity** | `face_recognition` (dlib) | Encodes and compares facial features against Blacklist/Authorized databases. |
| **Interface** | Streamlit | Provides the browser-based dashboard, live metrics, and interactive controls (sliders, checkboxes). |
| **Data Audit** | `csv`, `datetime` | Logs all alerts to `security_log.csv` for persistent storage and reporting. |

---

## 🖥️ Demonstration & Screenshots

The system is designed for clear visual communication.

### Status: Normal Operations

A clean dashboard showing system health and crowd metrics.



### Alert: Behavioral Anomaly

The system detects and flags a suspicious posture (e.g., Man Down or Hands Up).



### Alert: Security Breach

The system identifies a blacklisted individual or a zone intrusion event.



---

## 🛠️ Installation & Setup

### Prerequisites

* **Python 3.9+**
* **Install Libraries:**
    ```bash
    pip install streamlit opencv-python numpy ultralytics face-recognition pandas
    ```

### Execution

1.  **Prepare Data Folders:** Create two folders in the project root:
    * `./authorized` (for authorized face images)
    * `./blacklist` (for suspect face images)

2.  **Run the Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

3.  Access the URL provided by Streamlit in your browser. Click **"START SURVEILLANCE"** to begin analysis.
