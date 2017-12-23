import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

class DataSet:
    
    def __init__(self, input_dir):
        """
        Create an instance given the data folder

        Param:
        - input_dir: folder containing the seizure data set

        Return: None
        """
        self.input_dir = input_dir
        self.channel_length = 4097
        self.num_channels_per_set = 100
        self.data_set = self.read_channels_all_sets()
        self.num_segments_per_channel = 23
        self.target_map = {"A": 0,
                           "B": 0,
                           "C": 1,
                           "D": 1,
                           "E": 2}
        self.segment_data = self.create_segment_data() # use the number of segments and target map above
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.min_max_scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
    def read_channel_file(self, folder_name, file_name):
        """
        Read a single channel given a full path file
        Param:
        - folder_name: "Z", "O", "N", "F", "E"
        - file_name: file name in the folder
        Return: 
        - channel: a numpy array of CHANNEL_LENGTH elements
        """
        channel = np.zeros(self.channel_length)
        idx = 0
        file_path = os.path.join(self.input_dir, folder_name, file_name)
        with open(file_path) as f:
            for line in f:
                channel[idx] = float(line)
                idx += 1
        return channel

    def read_channels_in_set(self, set_name):
        """
        Read all NUM_CHANNELS files of a set.

        Param:
        - set_name: "A", "B", "C", "D", "E"

        Return:
        - channels_dict: dictionary with file name as key and numpy arrays of length CHANNEL_LENGTH as values
        """
        folder = {"A": "Z",
                  "B": "O",
                  "C": "N",
                  "D": "F",
                  "E": "S"}
        files = os.listdir(os.path.join(self.input_dir, folder[set_name]))
        assert(len(files) == self.num_channels_per_set), "Wrong number of signal files for set {}".format(set_name)
        channels = {}
        idx = 0
        for f in files:
            channels[f] = self.read_channel_file(folder[set_name], f)
            idx += 1
        return channels

    def read_channels_all_sets(self):
        """
        Read all NUM_CHANNELS files for all sets

        Param: None

        Return:
        - channels_dict: dictionary with set names as keys, and each value is another dictionary 
        with file name as keys and numpy ndarray of length CHANNEL_LENGTH as values
        """
        channels_dict = {}
        sets = ["A", "B", "C", "D", "E"]
        for s in sets:
            channels_dict[s] = self.read_channels_in_set(s)
        return channels_dict
    
    def create_segment_data(self, num_segments_per_channel = None, target_map = None):
        """
        Split channels into segments, creating data frame where each segment
        has its set name as target feature

        Param:
        - channel_dict: dictionary of channels with set names as keys, matrices of shape (NUM_CHANNELS, CHANNEL_LENGTH) as values
        - num_segments_per_channel: the number of segments to be created from each channel; each segment will be created
        with the same length CHANNEL_LENGTH / segments_per_channel. The remainings of channels are discarded (for now).
        - target_map: dictionary containing information about the target class in the returned ndarray, where set names
        "A", ..., "E" are keys and the values are the mapped integers

        Return:
        - segment_data: a dictionary with keys indicating the following info: the set name ("A",...,"E"), file name, start and end indices 
        within the channel from which the segment was extracted; values are numpy array of length num_features is 
        (channel_length / num_segments_per_channel) + 1 (the last feature is the target set name)
        """
        self.num_segments_per_channel = num_segments_per_channel if num_segments_per_channel is not None else self.num_segments_per_channel
        self.target_map = target_map if target_map is not None else self.target_map
        num_segments = 5 * self.num_channels_per_set * self.num_segments_per_channel
        num_features = int(self.channel_length / self.num_segments_per_channel) + 1  # including the target feature in the last position
        segment_data = {}
        sets = ["A", "B", "C", "D", "E"]
        for s in sets:
            channels = self.data_set[s]
            for file_name, channel in channels.items():
                for seg_idx in np.arange(self.num_segments_per_channel):
                    start = seg_idx * (num_features - 1)
                    end = start + num_features - 1
                    key_name = s + "_" + file_name + "_" + str(start)
                    segment_data[key_name] = np.append(channel[start : end], self.target_map[s])
        return segment_data

    def split_segment_train_test(self, test_ratio = 0.2, random_state = 0):
        assert(self.segment_data is not None), "Invalid segment_data"
        num_cols = int(self.channel_length / self.num_segments_per_channel) # excluding the target feature
        num_rows = len(self.segment_data.keys())
        X = np.zeros((num_rows, num_cols))
        y = np.zeros(num_rows)
        row = 0
        for _, values in self.segment_data.items():
            assert(num_cols == len(values) - 1), "Wrong segment length: {} vs {}".format(num_cols, len(values) - 1)
            X[row] = values[:num_cols]
            y[row] = values[num_cols]
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        for train_index, test_index in sss.split(X, y):
            self.X_train, self.X_test = X[train_index], X[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]

    def min_max_scaling_train_set(self):        
        assert(self.X_train is not None), "Invalid X_train"
        self.min_max_scaler = MinMaxScaler()
        self.X_train_scaled = self.min_max_scaler.fit_transform(self.X_train)

    def min_max_scaling_test_set(self):
        assert(self.min_max_scaler is not None), "Invalid min-max scaler"
        self.X_test_scaled = self.min_max_scaler.transform(self.X_test)
