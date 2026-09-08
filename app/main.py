# """
# Streamlit app launcher
# """

# import streamlit as st
# from app import sign_to_text, text_to_sign

# st.set_page_config(page_title="Sign Language Translator", layout="wide")
# st.sidebar.title("Sign Translator")
# mode = st.sidebar.selectbox("Mode", ["Sign → Text (Webcam)", "Text → Sign (Playback)"])

# if mode == "Sign → Text (Webcam)":
#     sign_to_text.run_streamlit_recognizer()
# else:
#     text_to_sign.show_text_to_sign()


# import streamlit as st
# import cv2
# import numpy as np
# import os
# from datetime import datetime
# from PIL import Image
# import mediapipe as mp
# import glob

# # --------- Setup ---------
# GESTURES = [
#     "HELLO", "YES", "NO", "PLEASE", "THANKYOU",
#     "SORRY", "HELP", "STOP", "ILOVEYOU", "GOODBYE"
# ]

# SAVE_PATH = "dataset/real"
# os.makedirs(SAVE_PATH, exist_ok=True)

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# # --------- Streamlit UI Tabs ---------
# tab = st.sidebar.radio("Select Mode", ["Record Gesture", "Text → Sign", "Sign → Text"])

# # ------------------ RECORD GESTURE ------------------
# if tab == "Record Gesture":
#     st.title("Sign Language Data Recorder ✋🤟")
#     st.write("Select a gesture and record samples using your webcam.")

#     gesture = st.selectbox("Choose a Gesture to Record:", GESTURES)
#     img_file = st.camera_input("Show the gesture and capture it")

#     if img_file is not None:
#         # Convert to OpenCV format
#         pil_img = Image.open(img_file)
#         frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

#         # Process with MediaPipe
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb_frame)

#         landmarks_array = None
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

#         # Save image & landmarks
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         folder = os.path.join(SAVE_PATH, gesture)
#         os.makedirs(folder, exist_ok=True)

#         image_path = os.path.join(folder, f"{gesture}_{timestamp}.jpg")
#         cv2.imwrite(image_path, frame)

#         if landmarks_array is not None:
#             landmarks_path = os.path.join(folder, f"{gesture}_{timestamp}.npy")
#             np.save(landmarks_path, landmarks_array)
#             st.success(f"Saved image and landmarks for {gesture}!")
#         else:
#             st.warning(f"Saved image, but no hand landmarks detected for {gesture}.")

#         st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# # ------------------ TEXT → SIGN ------------------
# elif tab == "Text → Sign":
#     st.title("Text → Sign (Playback)")

#     text = st.text_input("Type a gesture")
#     if st.button("Show Gesture"):
#         gesture = text.upper()
#         if gesture in GESTURES:
#             folder = f"{SAVE_PATH}/{gesture}"
#             npy_files = glob.glob(os.path.join(folder, "*.npy"))
#             if len(npy_files) > 0:
#                 # Load the first recorded landmark for preview
#                 landmarks = np.load(npy_files[0])

#                 # Simple 2D plot for visualization
#                 import matplotlib.pyplot as plt
#                 x = landmarks[:, 0]
#                 y = 1 - landmarks[:, 1]  # flip y-axis
#                 plt.figure(figsize=(3,3))
#                 plt.scatter(x, y, c='red')
#                 for i in range(len(x)):
#                     plt.text(x[i], y[i], str(i))
#                 plt.xlim(0,1)
#                 plt.ylim(0,1)
#                 plt.title(f"{gesture} landmarks")
#                 st.pyplot(plt)
#             else:
#                 st.warning(f"No landmarks found for {gesture}. Record the gesture first!")
#         else:
#             st.warning("Gesture not found. Available gestures: " + ", ".join(GESTURES))

# # ------------------ SIGN → TEXT ------------------
# elif tab == "Sign → Text":
#     st.title("Sign → Text (Webcam Preview)")

#     img_file = st.camera_input("Show your gesture")
#     if img_file is not None:
#         pil_img = Image.open(img_file)
#         frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

#         with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(rgb_frame)

#             if result.multi_hand_landmarks:
#                 for hand_landmarks in result.multi_hand_landmarks:
#                     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 # Optional: show raw landmarks
#                 lm = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark])
#                 st.write("Landmarks shape:", lm.shape)

#         st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")


import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import mediapipe as mp
import glob
import torch
from model.mlp import MLP
from model.gesture_dataset import GestureDataset
import matplotlib.pyplot as plt

# --------- Setup ---------
GESTURES = [
    "HELLO", "YES", "NO", "PLEASE", "THANKYOU",
    "SORRY", "HELP", "STOP", "ILOVEYOU", "GOODBYE"
]

SAVE_PATH = "dataset/real"
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ------------------- Load Model for Prediction -------------------
if "model_loaded" not in st.session_state:
    # Dataset for class map
    dataset = GestureDataset()
    class_map = {v: k for k, v in dataset.class_map.items()}  # idx → label

    # Load trained model
    input_size = dataset.num_features
    num_classes = len(class_map)
    model = MLP(input_size=input_size, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_path = os.path.join("model", "gesture_mlp.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    st.session_state.model = model
    st.session_state.class_map = class_map
    st.session_state.device = device
    st.session_state.model_loaded = True

# ----------------- Streamlit Tabs -----------------
tab = st.sidebar.radio("Select Mode", ["Record Gesture", "Text → Sign", "Sign → Text"])

# ------------------ RECORD GESTURE ------------------
if tab == "Record Gesture":
    st.title("Sign Language Data Recorder ✋🤟")
    st.write("Select a gesture and record samples using your webcam.")

    gesture = st.selectbox("Choose a Gesture to Record:", GESTURES)
    img_file = st.camera_input("Show the gesture and capture it")

    if img_file is not None:
        pil_img = Image.open(img_file)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        landmarks_array = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # Save image & landmarks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(SAVE_PATH, gesture)
        os.makedirs(folder, exist_ok=True)

        image_path = os.path.join(folder, f"{gesture}_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

        if landmarks_array is not None:
            landmarks_path = os.path.join(folder, f"{gesture}_{timestamp}.npy")
            np.save(landmarks_path, landmarks_array)
            st.success(f"Saved image and landmarks for {gesture}!")
        else:
            st.warning(f"Saved image, but no hand landmarks detected for {gesture}.")

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# ------------------ TEXT → SIGN ------------------
# elif tab == "Text → Sign":
#     st.title("Text → Sign (Playback)")

#     text = st.text_input("Type a gesture")
#     if st.button("Show Gesture"):
#         gesture = text.upper()
#         if gesture in GESTURES:
#             folder = f"{SAVE_PATH}/{gesture}"
#             npy_files = glob.glob(os.path.join(folder, "*.npy"))
#             if len(npy_files) > 0:
#                 landmarks = np.load(npy_files[0])

#                 import matplotlib.pyplot as plt
#                 x = landmarks[:, 0]
#                 y = 1 - landmarks[:, 1]  # flip y-axis
#                 plt.figure(figsize=(3,3))
#                 plt.scatter(x, y, c='red')
#                 for i in range(len(x)):
#                     plt.text(x[i], y[i], str(i))
#                 plt.xlim(0,1)
#                 plt.ylim(0,1)
#                 plt.title(f"{gesture} landmarks")
#                 st.pyplot(plt)
#             else:
#                 st.warning(f"No landmarks found for {gesture}. Record the gesture first!")
#         else:
#             st.warning("Gesture not found. Available gestures: " + ", ".join(GESTURES))


# ------------------ TEXT → SIGN ------------------
elif tab == "Text → Sign":
    st.title("Text → Sign (Playback)")

    text = st.text_input("Type a gesture")

    if st.button("Show Gesture"):
        gesture = text.upper()

        if gesture in GESTURES:
            
            folder = f"{SAVE_PATH}/{gesture}"
            import glob, os
            
            img_files = glob.glob(os.path.join(folder, "*.jpg"))
            npy_files = glob.glob(os.path.join(folder, "*.npy"))

            # ----- 1️⃣ Show REAL IMAGE if available -----
            if len(img_files) > 0:
                st.image(img_files[0], caption=f"{gesture} example", use_column_width=True)

            # ----- 2️⃣ Show LANDMARKS if available -----
            if len(npy_files) > 0:
                landmarks = np.load(npy_files[0])

                x = landmarks[:, 0]
                y = 1 - landmarks[:, 1]  # flip y-axis

                # Create a figure for Streamlit (IMPORTANT)
                fig, ax = plt.subplots(figsize=(3, 3))

                ax.scatter(x, y)
                for i, (xx, yy) in enumerate(zip(x, y)):
                    ax.text(xx, yy, str(i), fontsize=8)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"{gesture} landmarks")

                st.pyplot(fig)

            # ----- 3️⃣ No data -----
            if len(img_files) == 0 and len(npy_files) == 0:
                st.warning(f"No data found for {gesture}. Please record this gesture first!")
        
        else:
            st.warning("Gesture not found. Available gestures: " + ", ".join(GESTURES))

# ------------------ SIGN → TEXT ------------------
elif tab == "Sign → Text":
    st.title("Sign → Text (Webcam Preview)")

    img_file = st.camera_input("Show your gesture")
    if img_file is not None:
        pil_img = Image.open(img_file)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands_mp:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_mp.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Flatten landmarks and predict
                lm = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark], dtype=np.float32)
                x = torch.tensor(lm).view(1, -1).to(st.session_state.device, dtype=torch.float32)

                with torch.no_grad():
                    output = st.session_state.model(x)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_label = st.session_state.class_map[pred_idx]

                st.success(f"Predicted Gesture: {pred_label}")

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
