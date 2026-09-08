# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import torch
# from model.inference import GestureRecognizer

# # Load trained model
# recognizer = GestureRecognizer(model_path="model/trained_model.pth", device="cpu")

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# def run_streamlit_recognizer():
#     st.title("Sign → Text (Webcam)")

#     run = st.button("Start Webcam")
#     stop = st.button("Stop Webcam")

#     if run:
#         cap = cv2.VideoCapture(0)
#         with mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.6,
#             min_tracking_confidence=0.6
#         ) as hands:
#             stframe = st.empty()
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 frame = cv2.flip(frame, 1)
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = hands.process(rgb)

#                 if results.multi_hand_landmarks:
#                     # Draw landmarks
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                     # Extract landmarks
#                     lm = []
#                     for lm_point in results.multi_hand_landmarks[0].landmark:
#                         lm.extend([lm_point.x, lm_point.y])
#                     lm = np.array(lm)  # shape: (42,)

#                     # Predict gesture
#                     gesture = recognizer.predict(lm)  # GestureRecognizer handles reshaping internally
#                     cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#                 if stop:
#                     break

#         cap.release()
#         cv2.destroyAllWindows()


import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

st.title("Sign → Text (Webcam Preview)")

img_file = st.camera_input("Show your gesture")

if img_file is not None:
    pil_img = Image.open(img_file)
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Optional: show raw landmarks
            lm = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            st.write("Landmarks shape:", lm.shape)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
