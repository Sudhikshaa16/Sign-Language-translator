"""
Simple visualizer to plot a sequence of landmarks over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_seq(npz_path):
    d = np.load(npz_path)
    seq = d['seq']  # (T, 42)
    T = seq.shape[0]
    fig, axes = plt.subplots(3,1, figsize=(6,8))
    xs = seq[:,0::2]
    ys = seq[:,1::2]
    axes[0].plot(xs)
    axes[0].set_title("x coordinates per keypoint")
    axes[1].plot(ys)
    axes[1].set_title("y coordinates per keypoint")
    axes[2].plot(xs.mean(axis=1), label='mean_x'); axes[2].plot(ys.mean(axis=1), label='mean_y')
    axes[2].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        print("Usage: python scripts/visualize.py dataset/synthetic/...npz")
    else:
        plot_seq(sys.argv[1])
