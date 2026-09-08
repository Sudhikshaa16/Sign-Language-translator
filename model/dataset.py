# import numpy as np
# import os
# import torch 
# from torch.utils.data import Dataset

# class GestureDataset(Dataset):
#     def __init__(self, root_dir):
#         self.samples = []
#         self.labels = []
#         self.gestures = sorted(os.listdir(root_dir))
#         for idx, gesture in enumerate(self.gestures):
#             folder = os.path.join(root_dir, gesture)
#             for file in os.listdir(folder):
#                 if file.endswith(".npy"):
#                     self.samples.append(os.path.join(folder, file))
#                     self.labels.append(idx)
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         data = np.load(self.samples[idx])  # shape: (21, 3)
        
#         # Take only x and y (ignore z)
#         data = data[:, :2]  # shape: (21, 2)

#         # Flatten each frame to a single vector
#         data = data.flatten()  # shape: 42

#         # Convert to float32 tensor
#         data = torch.tensor(data, dtype=torch.float32)

#         # If your LSTM expects (seq_len, input_size) for a single timestep, unsqueeze:
#         # For a single frame sequence, we can do:
#         data = data.unsqueeze(0)  # shape: (1, 42)

#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return data, label
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

GESTURES = ["HELLO","YES","NO","PLEASE","THANKYOU"]

class GestureDataset(Dataset):
    def __init__(self, root_dir="dataset/real"):
        self.data = []
        self.labels = []

        for idx, gesture in enumerate(GESTURES):
            folder = os.path.join(root_dir, gesture)
            npy_files = glob.glob(os.path.join(folder, "*.npy"))

            for f in npy_files:
                lm = np.load(f)  # shape: (21,3)
                lm = lm[:, :2].flatten()  # use x,y → shape 42
                self.data.append(lm)
                self.labels.append(idx)

        if len(self.data) == 0:
            raise ValueError("❌ No .npy files found! Check dataset folder.")

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
