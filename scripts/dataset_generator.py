# # """
# # Record small number of sequences via webcam.
# # Saves numpy files: dataset/raw/<label>_<timestamp>_<idx>.npz
# # Each file stores a sequence of landmarks: shape (T, 42) -- 21 (x,y) per hand.
# # """

# # import argparse
# # import os
# # import time
# # from pathlib import Path
# # import numpy as np
# # import cv2
# # import mediapipe as mp

# # OUT_DIR = Path("dataset/raw")
# # OUT_DIR.mkdir(parents=True, exist_ok=True)

# # mp_hands = mp.solutions.hands
# # mp_draw = mp.solutions.drawing_utils

# # def extract_landmarks(results):
# #     # If no hand detected, return None
# #     if not results.multi_hand_landmarks:
# #         return None
# #     # pick first detected hand
# #     hand = results.multi_hand_landmarks[0]
# #     coords = []
# #     for lm in hand.landmark:
# #         coords.extend([lm.x, lm.y])
# #     return coords  # 42 floats

# # def record(label, seq_len=40, wait=2, save_every=1):
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         raise RuntimeError("Cannot open webcam")
# #     with mp_hands.Hands(static_image_mode=False,
# #                         max_num_hands=1,
# #                         min_detection_confidence=0.6,
# #                         min_tracking_confidence=0.6) as hands:
# #         print(f"Recording for label='{label}' in {wait}s. Prepare...")
# #         time.sleep(wait)
# #         idx = 0
# #         collected = []
# #         print("Starting... Press 'q' to quit early.")
# #         while True:
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break
# #             frame = cv2.flip(frame, 1)
# #             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #             res = hands.process(rgb)
# #             mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS) if res.multi_hand_landmarks else None
# #             lm = extract_landmarks(res)
# #             if lm:
# #                 collected.append(lm)
# #                 cv2.putText(frame, f"Collected: {len(collected)}/{seq_len}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
# #             else:
# #                 cv2.putText(frame, f"No hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
# #             cv2.imshow("Record", frame)
# #             key = cv2.waitKey(1) & 0xFF
# #             if key == ord('q'):
# #                 break
# #             if len(collected) >= seq_len:
# #                 # save
# #                 fname = OUT_DIR / f"{label}_{int(time.time())}_{idx}.npz"
# #                 np.savez_compressed(fname, seq=np.array(collected))
# #                 print("Saved", fname)
# #                 idx += 1
# #                 collected = []
# #                 print("Continue recording or press 'q' to stop.")
# #     cap.release()
# #     cv2.destroyAllWindows()


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--label", required=True, help="label name (must be in dataset/labels.txt)")
# #     parser.add_argument("--seq_len", type=int, default=40, help="timesteps per sample")
# #     parser.add_argument("--wait", type=int, default=2)
# #     args = parser.parse_args()
# #     record(args.label, seq_len=args.seq_len, wait=args.wait)


# # scripts/dataset_generator.py
# import numpy as np
# import os
# import random

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# GESTURES = ["HELLO", "YES", "NO", "PLEASE", "THANKYOU", "SORRY", "HELP", "STOP", "ILOVEYOU", "GOODBYE"]
# RAW_DIR = "dataset/raw"
# SYNTHETIC_DIR = "dataset/synthetic"
# SAMPLES_PER_GESTURE = 100     # raw samples
# AUGMENT_PER_SAMPLE = 5        # synthetic variations
# NUM_KEYPOINTS = 21            # MediaPipe hand landmarks

# os.makedirs(RAW_DIR, exist_ok=True)
# os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# # ----------------------------
# # HELPER FUNCTIONS
# # ----------------------------
# def generate_base_landmarks():
#     """Generate a random hand pose in normalized coordinates (x, y only)"""
#     return np.random.rand(NUM_KEYPOINTS, 2)  # 21 landmarks × 2 = 42 features

# def augment_landmarks(landmarks):
#     """Small variations: rotation, scaling, noise"""
#     new_landmarks = landmarks.copy()
    
#     # Add Gaussian noise
#     new_landmarks += np.random.normal(0, 0.02, new_landmarks.shape)
    
#     # Clip to [0,1]
#     new_landmarks = np.clip(new_landmarks, 0, 1)
    
#     # Optional: simple rotation around z-axis (just rotate x and y)
#     angle = random.uniform(-0.2, 0.2)  # radians
#     cos_a, sin_a = np.cos(angle), np.sin(angle)
#     x, y = new_landmarks[:, 0], new_landmarks[:, 1]
#     new_landmarks[:, 0] = x * cos_a - y * sin_a
#     new_landmarks[:, 1] = x * sin_a + y * cos_a
#     new_landmarks[:, 0] = np.clip(new_landmarks[:, 0], 0, 1)
#     new_landmarks[:, 1] = np.clip(new_landmarks[:, 1], 0, 1)
    
#     return new_landmarks


# # ----------------------------
# # GENERATE RAW DATA
# # ----------------------------
# print("Generating raw dataset...")
# for gesture in GESTURES:
#     gesture_dir = os.path.join(RAW_DIR, gesture)
#     os.makedirs(gesture_dir, exist_ok=True)
#     for i in range(SAMPLES_PER_GESTURE):
#         landmarks = generate_base_landmarks()
#         file_path = os.path.join(gesture_dir, f"{gesture}_{i:03d}.npy")
#         np.save(file_path, landmarks)

# # ----------------------------
# # GENERATE SYNTHETIC DATA
# # ----------------------------
# print("Generating synthetic dataset...")
# for gesture in GESTURES:
#     raw_gesture_dir = os.path.join(RAW_DIR, gesture)
#     synthetic_gesture_dir = os.path.join(SYNTHETIC_DIR, gesture)
#     os.makedirs(synthetic_gesture_dir, exist_ok=True)
    
#     raw_files = os.listdir(raw_gesture_dir)
#     for file in raw_files:
#         landmarks = np.load(os.path.join(raw_gesture_dir, file))
#         for j in range(AUGMENT_PER_SAMPLE):
#             augmented = augment_landmarks(landmarks)
#             aug_file_path = os.path.join(synthetic_gesture_dir, f"{file.split('.')[0]}_aug{j}.npy")
#             np.save(aug_file_path, augmented)

# print("Dataset generation complete!")
# print(f"Raw samples per gesture: {SAMPLES_PER_GESTURE}")
# print(f"Synthetic samples per raw sample: {AUGMENT_PER_SAMPLE}")
