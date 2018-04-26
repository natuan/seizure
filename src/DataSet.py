import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        self.segment_data_df = None
    
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

    def _statistics(self, signal_array):
        stats = [np.amin(signal_array), np.amax(signal_array), np.mean(signal_array), np.std(signal_array), np.percentile(signal_array, 25), np.percentile(signal_array, 50), np.percentile(signal_array, 75)]
        return stats
    
    def statistics(self):
        sets = ["A", "B", "C", "D", "E"]
        stats = {}
        all_signals = np.array([])
        for s in sets:
            signals = np.array([])
            for _, signal_array in self.data_set[s].items():
                signals = np.append(signals, signal_array)
            all_signals = np.append(all_signals, signals)
            stats[s] = self._statistics(signals)
        stats["all"] = self._statistics(all_signals)
        columns = ["min", "max", "mean", "std", "25percentile", "50percentile", "75percentile"]
        df = pd.DataFrame.from_dict(stats, orient="index")
        df.index.name = "class"
        df.columns = columns
        df.sort_index(inplace=True)
        df.to_csv(os.path.join(self.cache_dir, "statistics.csv"))
        return stats

    def plot(self, plot_dir_path, segments_per_class = 10):
        sets = ["A", "B", "C", "D", "E"]
        for s in sets:
            folder = os.path.join(plot_dir_path, s)
            if not os.path.exists(folder):
                os.makedirs(folder)
            key_list = list(self.data_set[s].keys())
            file_names = np.random.choice(key_list, segments_per_class) if segments_per_class is not None else key_list
            for f in file_names:
                signals = self.data_set[s][f]
                fig = plt.figure()
                plt.plot(signals)
                fig_name = "{}.eps".format(str(f.upper()).replace(".TXT",""))
                plot_file_path = os.path.join(folder, fig_name)
                plt.savefig(plot_file_path)
                plt.close()       
        
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
        target_map = target_map if target_map is not None else self.target_map
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
                    segment_data_dict[key_name] = np.append(channel[start : end], target_map[s])
        segment_data_df = pd.DataFrame.from_dict(segment_data_dict, orient = 'index')
        segment_data_df.index.name = "segment_id"
        num_features = int(self.channel_length / self.num_segments_per_channel) + 1 # the last feature is target
        col_names = ["f_{}".format(i) for i in range(num_features - 1)]
        col_names.append("target_class")
        segment_data_df.columns = col_names
        segment_data_df.sort_index(inplace=True)

        target_class_str = self._create_target_class_string()
        file_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}.csv".format(self.num_segments_per_channel, target_class_str))
        print("Saving segment data into {}...".format(file_path))
        segment_data_df.to_csv(file_path)
        print(">> Done") 
        self.segment_data_df = segment_data_df

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
        print("Saving {}...".format(train_path))
        train_df.to_csv(train_path)
        print(">> Done")

        val_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}_ratio{}_rand{}_VALID.csv".format(self.num_segments_per_channel, target_class_str, test_ratio, random_state))
        print("Saving {}...".format(val_path))
        val_df.to_csv(val_path)
        print(">> Done")
            
        test_path = os.path.join(self.cache_dir, "segment_numseg{}_target{}_ratio{}_rand{}_TEST.csv".format(self.num_segments_per_channel, target_class_str, test_ratio, random_state))
        print("Saving {}...".format(test_path))
        test_df.to_csv(test_path)
        print(">> Done")

    def load_features_and_target(self, file_path, signal_range = None):
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

    def load_and_preprocess(self, train_file_path, valid_file_path, test_file_path, whiten=False):
        X_train, y_train = self.load_features_and_target(train_file_path)
        X_valid, y_valid = self.load_features_and_target(valid_file_path)
        X_test, y_test = self.load_features_and_target(test_file_path)
        # Making sure the data is zero-mean and unit-variance
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
        if whiten:
            # Making sure uncorrelated between features
            pca = PCA(whiten=True).fit(X_train_scaled)
            X_train_preprocessed = pca.transform(X_train_scaled)
            X_valid_preprocessed = pca.transform(X_valid_scaled)
            X_test_preprocessed = pca.transform(X_test_scaled)
        else:
            X_train_preprocessed = X_train_scaled
            X_valid_preprocessed = X_valid_scaled
            X_test_preprocessed = X_test_scaled
        return X_train_preprocessed, X_valid_preprocessed, X_test_preprocessed, y_train, y_valid, y_test
  
def min_max_scale(X, signal_range, scaling_range = (0,1)):
    signal_min, signal_max = signal_range
    signal_delta = signal_max - signal_min
    scaling_min, scaling_max = scaling_range
    scaling_delta = scaling_max - scaling_min
    new_X = np.zeros(X.shape)
    for i in range(X.shape[0]):
        cutoff_Xi = [min(signal_max, max(signal_min, x)) for x in X[i]]
        new_X[i] = [((x - signal_min) / signal_delta) * scaling_delta + scaling_min for x in cutoff_Xi]
    test1 = scaling_min <= new_X
    test2 = new_X <= scaling_max
    assert(test1.all() and test2.all()), "Invalid scaled values"
    return new_X

def create_binary_data_set_class_E():
    root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
    data_set = DataSet(input_dir=os.path.join(root_dir, "input"),
                       cache_dir=os.path.join(root_dir, "cache"))
    data_set.target_map = {"A": 0,
                           "B": 0,
                           "C": 0,
                           "D": 0,
                           "E": 1}
    data_set.create_segment_data()
    data_set.split(test_ratio=0.2, random_state=50)

def create_data_set_class_AB_CD_E():
    root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
    data_set = DataSet(input_dir=os.path.join(root_dir, "input"),
                       cache_dir=os.path.join(root_dir, "cache"))
    data_set.target_map = {"A": 0,
                           "B": 0,
                           "C": 1,
                           "D": 1,
                           "E": 2}
    data_set.create_segment_data()
    data_set.split(test_ratio=0.2, random_state=25)

def plot_signals(segments_per_class = None):
    root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
    data_set = DataSet(input_dir=os.path.join(root_dir, "input"),
                       cache_dir=os.path.join(root_dir, "cache"))
    data_set.plot(os.path.join(data_set.cache_dir, "images"), segments_per_class=segments_per_class)
