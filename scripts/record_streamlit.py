# import streamlit as st
# import cv2
# import numpy as np
# import os
# from datetime import datetime
# from PIL import Image

# GESTURES = [
#     "HELLO", "YES", "NO", "PLEASE", "THANKYOU",
#     "SORRY", "HELP", "STOP", "ILOVEYOU", "GOODBYE"
# ]

# SAVE_PATH = "dataset/real"

# os.makedirs(SAVE_PATH, exist_ok=True)

# st.title("Sign Language Data Recorder ✋🤟")
# st.write("Select a gesture and record samples using your webcam.")

# gesture = st.selectbox("Choose a Gesture to Record:", GESTURES)

# img = st.camera_input("Show the gesture and capture it")

# if img is not None:
#     # Convert to numpy
#     pil_img = Image.open(img)
#     frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder = os.path.join(SAVE_PATH, gesture)
#     os.makedirs(folder, exist_ok=True)

#     file_path = os.path.join(folder, f"{gesture}_{timestamp}.jpg")
#     cv2.imwrite(file_path, frame)

#     st.success(f"Saved: {file_path}")


import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import mediapipe as mp

# --------- Setup ---------
GESTURES = [
    "HELLO", "YES", "NO", "PLEASE", "THANKYOU",
    "SORRY", "HELP", "STOP", "ILOVEYOU", "GOODBYE"
]

SAVE_PATH = "dataset/real"
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --------- Streamlit UI ---------
st.title("Sign Language Data Recorder ✋🤟")
st.write("Select a gesture and record samples using your webcam.")

gesture = st.selectbox("Choose a Gesture to Record:", GESTURES)

img_file = st.camera_input("Show the gesture and capture it")

if img_file is not None:
    # Convert to OpenCV format
    pil_img = Image.open(img_file)
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    landmarks_array = None  # default if no hand detected

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convert landmarks to numpy array: 21 points x 3 (x, y, z)
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Prepare saving paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(SAVE_PATH, gesture)
    os.makedirs(folder, exist_ok=True)

    # Save image
    image_path = os.path.join(folder, f"{gesture}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)

    # Save landmarks if detected
    if landmarks_array is not None:
        landmarks_path = os.path.join(folder, f"{gesture}_{timestamp}.npy")
        np.save(landmarks_path, landmarks_array)
        st.success(f"Saved image and landmarks for {gesture}!")
    else:
        st.warning(f"Saved image, but no hand landmarks detected for {gesture}.")

    # Display the frame with landmarks
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
