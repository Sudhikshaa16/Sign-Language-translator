# helper utilities for app
import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y])
    return np.array(coords, dtype=np.float32)
