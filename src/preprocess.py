import os
import numpy as np

SZ = 4097

def read_channel_file(file_path):
    channel = np.zeros(SZ)
    idx = 0
    with open(file_path) as f:
        for line in f:
            channel[idx] = float(line)
            idx += 1
    return channel

def read_channels_in_set(set_name):
    """
    Read all 100 files of a set.
    
    Param:
    - set_name: "A", "B", "C", "D", "E"

    Return:
    - matrix of signals with the shape (100, 4097)
    """
    folder = {"A": "Z",
              "B": "O",
              "C": "N",
              "D": "F",
              "E": "S"}
    files = os.listdir(folder[set_name])
    assert(len(files) == 100), "Wrong number of signal files for set {}".format(set_name)
    channels = np.zeros((len(files), SZ))
    idx = 0
    for f in files:
        file_path = os.path.join("..", "input", f)
        channels[idx] = read_signal_file(file_path)
        idx += 1
    return channels

def read_channels_all_sets():
    channels_dict = {}
    sets = ["A", "B", "C", "D", "E"]
    for s in sets:
        channels_dict[s] = read_channels_in_set(s)
    return channels_dict

def create_data_set(segments_per_channel = 23):
    channels_dict = read_channels_all_sets()
    vals_per_seg = SZ / segments_per_channel
        
