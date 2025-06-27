import os
import cv2 # type: ignore
import json
import base64
import time
import streamlit as st # type: ignore
import mediapipe as mp # type: ignore
import google.generativeai as genai # type: ignore
import difflib

# Load API Key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBV--3FY0KJXMhl9M-dN2JWN5QWCg9MMgs")  # Replace with actual key
genai.configure(api_key=API_KEY)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture file
GESTURE_FILE = "gestures.json"

# Load saved gestures
if os.path.exists(GESTURE_FILE):
    with open(GESTURE_FILE, "r") as file:
        gesture_mappings = json.load(file)
else:
    gesture_mappings = {}

# Function to extract gesture label from AI response
def extract_gesture_label(response_text):
    label = response_text.split("\n")[0].strip()
    label = label.replace("", "").replace("*", "").replace("'", "").replace(".", "").lower()
    return label

# Function to generate chatbot response
def get_chatbot_reply(gesture):
    # st.session_state.chat_history.clear()  # Clear previous responses
    # ✅ Fix: Check before accessing
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        st.session_state.chat_history.clear()

    try:
        # ✅ Ensure chat history is initialized
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Initialize Gemini models
        model_pro = genai.GenerativeModel("gemini-1.5-flash")
        model_flash = genai.GenerativeModel("gemini-1.5-flash")

        # Conversation history for "gemini-1.5-pro"
        messages = [{"role": "user", "parts": [{"text": f"User made a '{gesture}' gesture."}]}]
        messages += [{"role": "assistant", "parts": [{"text": reply}]} for reply in st.session_state.chat_history]

        # Generate responses from both models
        response_pro = model_pro.generate_content(messages)
        response_flash = model_flash.generate_content(f"If a user makes a '{gesture}' gesture, how should a chatbot respond? Provide a friendly response.")

        # Extract and combine responses
        response_pro_text = response_pro.text.strip() if response_pro and response_pro.text else "I couldn't understand that gesture."
        response_flash_text = response_flash.text.strip() if response_flash and response_flash.text else "I couldn't understand that gesture."

        # ✅ Save chatbot responses in session history
        st.session_state.chat_history.append(response_pro_text)
        st.session_state.chat_history.append(response_flash_text)

        # Return combined response
        return f"**Gemini Pro:** {response_pro_text}\n\n**Gemini Flash:** {response_flash_text}"

    except Exception as e:
        return f"Error generating response: {e}"
time.sleep(3)  # Wait before processing the next response

# Function to capture gesture using Gemini API
def capture_gesture(frame):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode()

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Identify the hand gesture and return a single short label.",
            {"mime_type": "image/jpeg", "data": img_base64}
        ])

        if response.text:
            return extract_gesture_label(response.text.strip())  # Get short name
    except Exception as e:
        return None

# Streamlit UI
st.title("🖐 Gesture-Based AI Assistant")

# Start/Stop Camera Buttons
start_camera = st.sidebar.button("Start Camera", key="start_camera_btn")
stop_camera = st.sidebar.button("Stop Camera", key="stop_camera_btn")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_window = st.empty()

if start_camera:
    st.sidebar.write("🎥 Camera Started. Show a gesture!")

gesture_processing = False  # Flag to prevent multiple detections at once

while start_camera and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Couldn't capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and not gesture_processing:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Recognize gesture
        gesture_name = capture_gesture(frame)
        if gesture_name:
            gesture_processing = True  # Block further detections until response is given
            st.sidebar.write(f"🖐 Detected Gesture: {gesture_name}")

            chatbot_reply = get_chatbot_reply(gesture_name)
            st.write(f"🤖 Chatbot: {chatbot_reply}")

            time.sleep(3)  # Wait before allowing the next gesture detection
            gesture_processing = False  # Reset flag

    # Display video feed in Streamlit
    frame_window.image(frame, channels="RGB")

    # Stop camera condition
    if stop_camera:
        cap.release()
        cv2.destroyAllWindows()
        st.sidebar.write("⏹ Camera Stopped")
        break

cap.release()
cv2.destroyAllWindows()