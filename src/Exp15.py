import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from StackedAutoencoders import *
from StackBuilder import *
from Classifier import Classifier
from Utils import *

# Load the original data set, and creating segment data
# using default number of segments and target map
# (See DataSet constructor)
root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
data_set = DataSet(input_dir=os.path.join(root_dir, "input"),
                   cache_dir=os.path.join(root_dir, "cache"))

# Load train and test sets
X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TEST.csv"))

scaler = MinMaxScaler(feature_range=(-1,1))
noise_stddev = 0.1
n_folds = 5
n_inputs = X_train.shape[1]
n_neurons_range = [64] #[5, 10, 50, 75, 100, 125, 178, 200, 250, 300]
hidden_activation = tf.nn.tanh
n_epochs = 10000
batch_size = 256
checkpoint_steps = 1024
seed = 0
hidden_weights_size_to_plot = 1.0
reconstructed_examples_per_class_to_plot = 20

name = config_str(prefix="denoise", n_inputs=n_inputs, hidden_activation=hidden_activation, noise_stddev=noise_stddev)
name += "_folds{}".format(n_folds)
cache_dir = os.path.join(root_dir, name)
tf_log_dir = os.path.join(cache_dir, "tf_logs")
generate_unit_autoencoders(X_train,
                           y_train,
                           scaler,
                           n_neurons_range,
                           n_folds=n_folds,
                           hidden_activation=hidden_activation,
                           noise_stddev=noise_stddev,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           checkpoint_steps=checkpoint_steps,
                           seed=seed,
                           hidden_weights_size_to_plot=hidden_weights_size_to_plot,
                           reconstructed_examples_per_class_to_plot=reconstructed_examples_per_class_to_plot,
                           cache_dir=cache_dir,
                           tf_log_dir=tf_log_dir)
