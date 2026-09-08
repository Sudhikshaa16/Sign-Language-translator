import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_landmarks(landmarks):
    fig, ax = plt.subplots()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ln, = plt.plot([], [], 'ro-')

    def init():
        ln.set_data([], [])
        return ln,

    def update(frame):
        x = landmarks[:,0]
        y = 1 - landmarks[:,1]  # invert y for display
        ln.set_data(x, y)
        return ln,

    ani = FuncAnimation(fig, update, frames=1, init_func=init, blit=True)
    return fig
