import os
import numpy as np

# Total number of values in each channel (file)
CHANNEL_LENGTH = 4097

# Total number of channels per set
NUM_CHANNELS = 100

# Input folder containing the data set
INPUT_DIR = os.path.join("..", "input")

def read_channel_file(file_path):
    """
    Read a single channel given a full path file
    Param:
    - file_path: full path file name
    Return: 
    - channel: a numpy array of CHANNEL_LENGTH elements
    """
    channel = np.zeros(CHANNEL_LENGTH)
    idx = 0
    with open(file_path) as f:
        for line in f:
            channel[idx] = float(line)
            idx += 1
    return channel

def read_channels_in_set(set_name):
    """
    Read all NUM_CHANNELS files of a set.
    
    Param:
    - set_name: "A", "B", "C", "D", "E"

    Return:
    - matrix of signals with the shape (NUM_CHANNELS, CHANNEL_LENGTH)
    """
    folder = {"A": "Z",
              "B": "O",
              "C": "N",
              "D": "F",
              "E": "S"}
    files = os.listdir(os.path.join(INPUT_DIR, folder[set_name]))
    assert(len(files) == NUM_CHANNELS), "Wrong number of signal files for set {}".format(set_name)
    channels = np.zeros((len(files), CHANNEL_LENGTH))
    idx = 0
    for f in files:
        file_path = os.path.join(INPUT_DIR, folder[set_name], f)
        channels[idx] = read_channel_file(file_path)
        idx += 1
    return channels

def read_channels_all_sets():
    """
    Read all NUM_CHANNELS files for all sets

    Param: None
    
    Return:
    - channel_dict: dictionary with set names as keys, matrices of shape (NUM_CHANNELS, CHANNEL_LENGTH) as values
    """
    channel_dict = {}
    sets = ["A", "B", "C", "D", "E"]
    for s in sets:
        channel_dict[s] = read_channels_in_set(s)
    return channel_dict

def create_data_set(channel_dict, segments_per_channel, target_map):
    """
    Split channels into segments, creating data frame where each segment
    has its set name as target feature

    Param:
    - channel_dict: dictionary of channels with set names as keys, matrices of shape (NUM_CHANNELS, CHANNEL_LENGTH) as values
    - segments_per_channel: the number of segments to be created from each channel; each segment will be created
    with the same length CHANNEL_LENGTH / segments_per_channel. The remainings of channels are discarded (for now).
    - target_map: dictionary containing information about the target class in the returned ndarray, where set names
    "A", ..., "E" are keys and the values are the mapped integers

    Return:
    - data_set: a numpy ndarray of shape (num_segments, num_features); num_segments is the number of segments, num_features is 
    (CHANNEL_LENGTH / segments_per_channel) + 1 (the last feature is the target set name)
    """
    channel_dict = read_channels_all_sets()
    num_segments = 5 * NUM_CHANNELS * segments_per_channel
    num_features = int(CHANNEL_LENGTH / segments_per_channel) + 1  # including the target feature in the last position
    data_set = np.zeros((num_segments, num_features))
    row_idx = 0
    sets = ["A", "B", "C", "D", "E"]
    for s in sets:
        channels = channel_dict[s]
        assert(channels.shape == (NUM_CHANNELS, CHANNEL_LENGTH)), "Wrong channel matrix of set ".format(s)
        for channel_idx in np.arange(NUM_CHANNELS):
            for seg_idx in np.arange(segments_per_channel):
                start = seg_idx * (num_features - 1)
                end = start + num_features - 1
                data_set[row_idx] = np.append(channels[channel_idx][start : end], target_map[s])
                row_idx += 1
    return data_set

def create_default_data_set():
    channel_dict = read_channels_all_sets()
    segments_per_channel = 23
    target_map = {"A": 0,
                  "B": 0,
                  "C": 1,
                  "D": 1,
                  "E": 2}
    data_set = create_data_set(channel_dict, segments_per_channel, target_map)
    return data_set

