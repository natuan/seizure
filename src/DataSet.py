import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

class DataSet:
    
    def __init__(self, input_dir, cache_dir):
        """
        Create an instance given the data folder

        Param:
        - input_dir: folder containing the seizure data set

        Return: None
        """
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.channel_length = 4097
        self.num_channels_per_set = 100
        self.data_set = self.read_channels_all_sets()

        # Segment the original data into segments with the default settings
        self.num_segments_per_channel = 23
        self.target_map = {"A": 0,
                           "B": 0,
                           "C": 1,
                           "D": 1,
                           "E": 2}
        self.segment_data_df = self.create_segment_data() # use the number of segments and target map above
        
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

    def _create_target_class_string(self):
        target_class_str = "@"
        for class_id in range(5):
            found = False
            for class_str in ["A", "B", "C", "D", "E"]:
                if self.target_map[class_str] == class_id:
                    target_class_str += class_str
                    found = True
            if found:
                target_class_str += "@"
        return target_class_str
            
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
        - segment_data_df: a data frame with keys indicating the following info: the set name ("A",...,"E"), file name, start and end indices 
        within the channel from which the segment was extracted; values are numpy array of length num_features, which is 
        (channel_length / num_segments_per_channel) + 1 (the last feature is the target set name)
        """
        self.num_segments_per_channel = num_segments_per_channel if num_segments_per_channel is not None else self.num_segments_per_channel
        self.target_map = target_map if target_map is not None else self.target_map
        num_segments = 5 * self.num_channels_per_set * self.num_segments_per_channel
        num_features = int(self.channel_length / self.num_segments_per_channel) + 1  # including the target feature in the last position
        segment_data_dict = {}
        sets = ["A", "B", "C", "D", "E"]
        for s in sets:
            channels = self.data_set[s]
            for file_name, channel in channels.items():
                for seg_idx in np.arange(self.num_segments_per_channel):
                    start = seg_idx * (num_features - 1)
                    end = start + num_features - 1
                    key_name = s + "_" + file_name + "_" + str(start)
                    segment_data_dict[key_name] = np.append(channel[start : end], self.target_map[s])
        segment_data_df = pd.DataFrame.from_dict(segment_data_dict, orient = 'index')
        segment_data_df.index.name = "segment_id"
        num_features = int(self.channel_length / self.num_segments_per_channel) + 1 # the last feature is target
        col_names = ["f_{}".format(i) for i in range(num_features - 1)]
        col_names.append("target_class")
        segment_data_df.columns = col_names
        segment_data_df.sort_index(inplace=True)          

        target_class_str = self._create_target_class_string()
        file_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}.csv".format(self.num_segments_per_channel, target_class_str))
        if (not os.path.exists(file_path)):
            print("Saving segment data into {}...".format(file_path))
            segment_data_df.to_csv(file_path)
            print(">> Done")
            
        return segment_data_df

    def split(self, test_ratio = 0.2, random_state = 0):
        assert(self.segment_data_df is not None), "Invalid segment_data"
        num_cols = self.segment_data_df.shape[1] - 1 # excluding the target feature
        num_rows = len(self.segment_data_df.index)

        X = np.array(self.segment_data_df.index.values)
        y = [self.segment_data_df.loc[segment_id, "target_class"] for segment_id in X]
        assert(len(X) == len(y))

        split = StratifiedShuffleSplit(n_splits=1, test_size=2*test_ratio, random_state=random_state)
        for train_indices, val_test_indices in split.split(X, y):            
            train_segment_ids = X[train_indices]
            train_df = self.segment_data_df.loc[train_segment_ids]

            # Split the validation and test indices into two equal parts
            X_val_test = X[val_test_indices]
            y_val_test = [self.segment_data_df.loc[segment_id, "target_class"] for segment_id in X_val_test]
            
            val_test_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 2 * random_state)
            for val_indices, test_indices in val_test_splitter.split(X_val_test, y_val_test):
                val_segment_ids = X_val_test[val_indices]
                test_segment_ids = X_val_test[test_indices]
                val_df = self.segment_data_df.loc[val_segment_ids]
                test_df = self.segment_data_df.loc[test_segment_ids]
            train_df.sort_index(inplace=True)
            val_df.sort_index(inplace=True)
            test_df.sort_index(inplace=True)
            
        target_class_str = self._create_target_class_string()
        train_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}_ratio{}_rand{}_TRAIN.csv".format(self.num_segments_per_channel, target_class_str, test_ratio, random_state))
        if (not os.path.exists(train_path)):
            print("Saving {}...".format(train_path))
            train_df.to_csv(train_path)
            print(">> Done")

        val_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}_ratio{}_rand{}_VALID.csv".format(self.num_segments_per_channel, target_class_str, test_ratio, random_state))
        if (not os.path.exists(val_path)):
            print("Saving {}...".format(val_path))
            val_df.to_csv(val_path)
            print(">> Done")
            
        test_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}_ratio{}_rand{}_TEST.csv".format(self.num_segments_per_channel, target_class_str, test_ratio, random_state))
        if (not os.path.exists(test_path)):
            print("Saving {}...".format(test_path))
            test_df.to_csv(test_path)
            print(">> Done")

    def load_features_and_target(self, file_path):
        """
        Load segment data frame and separate it into features and target

        Params:
        - file_path: segment data frame file in full path

        Return: (X, y) where X is features of shape (n_samples, n_features ) and y is target of shape (n_samples, 1)
        """
        assert(os.path.exists(file_path)), "File {} not exist".format(file_path)
        print("Loading features and target from segment data frame {}...".format(file_path))
        df = pd.DataFrame.from_csv(file_path, index_col = 0)
        X = pd.DataFrame.as_matrix(df.drop(["target_class"], axis = 1))
        y = pd.DataFrame.as_matrix(df["target_class"])
        print(">> Done\n")
        return X, y
        
    
        
