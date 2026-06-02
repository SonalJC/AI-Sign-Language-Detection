import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import threading

# --- PAGE CONFIG ---
st.set_page_config(page_title="ASL AI Translator", layout="wide")

# Custom CSS for better look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTitle { color: #2e4053; font-family: 'Helvetica'; }
    .status-box { 
        padding: 20px; 
        border-radius: 10px; 
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TITLE SECTION ---
st.title("Sign Language Recognition System")
st.markdown("---")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    return joblib.load('sign_language_model.p')

model = load_model()

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Audio Engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- MAIN LAYOUT ---
# Camera left mein aur Prediction Info right mein
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Live Camera Feed")
    FRAME_WINDOW = st.image([], use_container_width=True)

with col2:
    st.subheader("📊 Recognition Status")
    prediction_placeholder = st.empty()
    st.info("Tip: Keep your hand steady inside the frame for better accuracy.")
    
    # Sidebar control
    st.sidebar.header("Control Panel")
    run = st.sidebar.checkbox('Power On Camera', value=False)
    if st.sidebar.button('Reset System'):
        st.rerun()

# --- PROCESSING ---
cap = cv2.VideoCapture(0)
last_spoken = ""

while run:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_label = "Scanning..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Match Dataset Format (Green on White)
            blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            mp_draw.draw_landmarks(blank_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3),
                                 mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3))
            
            # Predict
            gray_skeleton = cv2.cvtColor(blank_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_skeleton, (64, 64)).flatten()
            prediction = model.predict([resized])
            current_label = str(prediction[0])

            # Audio
            if current_label != last_spoken:
                msg = "Help" if current_label == 'H' else ("Water" if current_label == 'W' else current_label)
                threading.Thread(target=speak, args=(msg,), daemon=True).start()
                last_spoken = current_label

            # Drawing landmarks on camera frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Update placeholders
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prediction_placeholder.markdown(f"""
        <div class="status-box">
            <h1 style="color: #28a745; font-size: 80px;">{current_label}</h1>
            <p style="font-size: 20px;">Detected Sign</p>
        </div>
    """, unsafe_allow_html=True)

if not run:
    cap.release()
    prediction_placeholder.warning("System is Standby. Toggle 'Power On' to start.")