# # model/gesture_dataset.py
# import os
# import torch
# from torch.utils.data import Dataset
# import numpy as np

# class GestureDataset(Dataset):
#     def __init__(self, root_dir=None):
#         """
#         Expects folder structure:
#         sign_language_translator/
#             dataset/
#                 real/
#                     CLASS_1/
#                     CLASS_2/
#         """
#         if root_dir is None:
#             # Default path: ../dataset/real from model/
#             root_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "real")
#             root_dir = os.path.abspath(root_dir)

#         if not os.path.exists(root_dir):
#             raise FileNotFoundError(f"Dataset folder not found at {root_dir}")

#         self.root_dir = root_dir
#         self.images = []
#         self.labels = []

#         # Map class names to integers
#         self.class_map = {name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))}

#         # Load .npy files
#         for label_name in os.listdir(root_dir):
#             class_dir = os.path.join(root_dir, label_name)
#             if os.path.isdir(class_dir):
#                 for img_name in os.listdir(class_dir):
#                     if img_name.endswith(".npy"):
#                         self.images.append(os.path.join(class_dir, img_name))
#                         self.labels.append(self.class_map[label_name])

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         label = self.labels[idx]

#         # Load numpy array
#         image = np.load(img_path).astype(np.float32)

#         # Normalize to 0-1
#         if image.max() > 1.0:
#             image = image / 255.0

#         # Convert HxWxC -> CxHxW if needed
#         image = torch.tensor(image)
#         if image.ndim == 3:
#             image = image.permute(2, 0, 1)

#         return image, label

# model/gesture_dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class GestureDataset(Dataset):
    def __init__(self, root_dir=None):
        """
        Dataset structure:
        sign_language_translator/
            dataset/
                real/
                    CLASS_1/
                    CLASS_2/
        """
        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "real")
            root_dir = os.path.abspath(root_dir)

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset folder not found at {root_dir}")

        self.root_dir = root_dir
        self.images = []
        self.labels = []

        # Map class names to integers
        self.class_map = {name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))}

        # Load .npy files
        for label_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(".npy"):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_map[label_name])

        # Detect image shape automatically
        sample = np.load(self.images[0])
        self.image_shape = sample.shape  # e.g., (H, W, C) or (C, H, W)
        self.num_features = np.prod(self.image_shape)  # total features for MLP

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = np.load(img_path).astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        image = torch.tensor(image)

        # Flatten to 1D vector for MLP
        image = image.view(-1)

        return image, label
