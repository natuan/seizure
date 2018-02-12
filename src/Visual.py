import os
import matplotlib.pyplot as plt

import preprocess as pp

def plot_channel(file_path):
    channel = -1 * pp.read_channel_file(file_path)
    plt.plot(channel)
    plt.show()
