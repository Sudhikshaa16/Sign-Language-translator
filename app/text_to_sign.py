# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from scripts.visualizer import animate_landmarks

# GESTURES = ["HELLO","YES","NO","PLEASE","THANKYOU","SORRY","HELP","STOP","ILOVEYOU","GOODBYE"]

# def show_text_to_sign():
#     st.title("Text → Sign (Playback)")
#     text = st.text_input("Type a word (gesture)")
    
#     if st.button("Show Gesture"):
#         gesture = text.upper()
#         if gesture in GESTURES:
#             file_path = f"dataset/synthetic/{gesture}/{gesture}_000.npy"
#             landmarks = np.load(file_path)
#             st.pyplot(animate_landmarks(landmarks))
#         else:
#             st.warning("Gesture not found. Available gestures: " + ", ".join(GESTURES))


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scripts.visualizer import animate_landmarks  # your existing visualizer

GESTURES = ["HELLO","YES","NO","PLEASE","THANKYOU"]

st.title("Text → Sign (Playback)")

text = st.text_input("Type a gesture")

if st.button("Show Gesture"):
    gesture = text.upper()
    if gesture in GESTURES:
        # Pick the first saved landmark from real dataset
        import os, glob
        folder = f"dataset/real/{gesture}"
        npy_files = glob.glob(os.path.join(folder, "*.npy"))
        
        if len(npy_files) > 0:
            landmarks = np.load(npy_files[0])
            st.pyplot(animate_landmarks(landmarks))
        else:
            st.warning(f"No landmarks found for {gesture}. Record the gesture first!")
    else:
        st.warning("Gesture not found. Available gestures: " + ", ".join(GESTURES))
