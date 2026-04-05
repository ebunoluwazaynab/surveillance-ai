import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="AI Surveillance Portal", layout="wide")
st.title("🛡️ Real-Time AI Monitoring System")

# 2. Sidebar Settings
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
stop_button = st.sidebar.button("Stop Monitoring")

# 3. Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 4. Video Stream Logic
frame_placeholder = st.empty() # This replaces the cv2 window
cap = cv2.VideoCapture(0)

# Set resolution (as we discussed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera.")
        break

    # Run YOLOv8
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    # CRITICAL: Convert BGR (OpenCV) to RGB (Web Browsers)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # 5. Display the frame in the browser
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

cap.release()
st.info("Monitoring Stopped.")