import os
import numpy as np
import pandas as pd

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
        self.num_channels = 100
        self.data_set = self.read_channels_all_sets()
        
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
        assert(len(files) == self.num_channels), "Wrong number of signal files for set {}".format(set_name)
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
