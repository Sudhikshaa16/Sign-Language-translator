# """
# Augment raw landmark sequences to create synthetic variations.
# Saves into dataset/synthetic/
# This performs:
#  - time-warping (stretch/compress)
#  - jitter (small noise)
#  - horizontal flip
#  - slight scaling / translation
# """

# import numpy as np
# from pathlib import Path
# import random
# import os
# from tqdm import tqdm

# RAW_DIR = Path("dataset/raw")
# SYN_DIR = Path("dataset/synthetic")
# SYN_DIR.mkdir(parents=True, exist_ok=True)

# def time_warp(seq, target_len):
#     # simple interpolation or decimation
#     orig = np.linspace(0,1,len(seq))
#     target = np.linspace(0,1,target_len)
#     out = np.stack([np.interp(target, orig, seq[:,i]) for i in range(seq.shape[1])], axis=1)
#     return out

# def jitter(seq, sigma=0.01):
#     noise = np.random.normal(0, sigma, seq.shape)
#     return seq + noise

# def flip_horizontal(seq):
#     # flip x coordinate: x -> 1 - x (landmarks are normalized 0..1)
#     s = seq.copy()
#     s[:,0::2] = 1.0 - s[:,0::2]
#     return s

# def scale_translate(seq, scale_range=(0.9,1.1), t_range=(-0.05, 0.05)):
#     s = seq.copy()
#     sx = random.uniform(*scale_range)
#     sy = random.uniform(*scale_range)
#     tx = random.uniform(*t_range)
#     ty = random.uniform(*t_range)
#     s[:,0::2] = (s[:,0::2] - 0.5) * sx + 0.5 + tx
#     s[:,1::2] = (s[:,1::2] - 0.5) * sy + 0.5 + ty
#     return np.clip(s, 0.0, 1.0)

# def augment_file(path, n_aug=10, target_len=40):
#     data = np.load(path)["seq"]  # (T, 42)
#     outs = []
#     for i in range(n_aug):
#         # random target length +/- small
#         tl = int(target_len * random.uniform(0.8, 1.2))
#         s = time_warp(data, tl)
#         s = jitter(s, sigma=random.uniform(0.005, 0.02))
#         if random.random() < 0.5:
#             s = flip_horizontal(s)
#         s = scale_translate(s)
#         # finally ensure length target_len by warping again
#         s = time_warp(s, target_len)
#         outs.append(s.astype(np.float32))
#     return outs

# def run(n_aug=10, target_len=40):
#     src = list(RAW_DIR.glob("*.npz"))
#     if not src:
#         print("No raw files in dataset/raw. Run dataset_generator.py first.")
#         return
#     for p in tqdm(src):
#         label = p.stem.split("_")[0]
#         outs = augment_file(p, n_aug=n_aug, target_len=target_len)
#         for i, arr in enumerate(outs):
#             out_name = SYN_DIR / f"{label}_{p.stem}_aug{i}.npz"
#             np.savez_compressed(out_name, seq=arr, label=label)
#     print("Augmentation done. Synthetic files in", SYN_DIR)

# if __name__ == "__main__":
#     run(n_aug=10, target_len=40)
