import cv2
import mediapipe as mp
import numpy as np
import os
import time

GESTURES = ["HELLO", "YES", "NO", "PLEASE", "THANKYOU", "SORRY", "HELP", "STOP", "ILOVEYOU", "GOODBYE"]

SAVE_DIR = "dataset/raw/real"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def record_gesture(gesture_name, samples=200):
    gesture_path = os.path.join(SAVE_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"Recording: {gesture_name} ...")

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
        count = 0
        while count < samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract 21 landmarks → 42 features (x,y)
                    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

                    fname = f"{gesture_name}_{count:03d}.npy"
                    np.save(os.path.join(gesture_path, fname), landmarks)

                    # Draw for visual feedback
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    count += 1

            cv2.putText(frame, f"{gesture_name}: {count}/{samples}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Recording...", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished recording {gesture_name}!")


# RECORD ALL GESTURES
for gesture in GESTURES:
    input(f"\nPress ENTER to start recording: {gesture}")
    record_gesture(gesture, samples=150)

print("All gestures recorded successfully!")
