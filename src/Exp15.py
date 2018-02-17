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
ratio = 0.15
X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand0_TRAIN.csv".format(ratio)))
X_valid, y_valid = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand0_VALID.csv".format(ratio)))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand0_TEST.csv".format(ratio)))

min_scale, max_scale = -1, 1
scaler = MinMaxScaler(feature_range=(-1,1))
noise_stddev = 0.1
n_folds = 1
n_inputs = X_train.shape[1]
n_neurons_range = [1000] #[50, 100, 150, 200, 300, 500, 1000]
hidden_activation = tf.nn.elu
n_epochs = 5000
batch_size = 32
checkpoint_steps = 100
seed = 0
n_observable_hidden_neurons = 0
n_reconstructed_examples_per_class_to_plot = 20

adam_lr = 0.000005
optimizer = tf.train.AdamOptimizer(adam_lr)

name = config_str(prefix="denoise_1000NEURONS_", ratio=ratio, min_scale=min_scale, max_scale=max_scale, adam_lr=adam_lr, n_epochs=n_epochs, n_folds=n_folds,
                  n_inputs=n_inputs, hidden_activation=hidden_activation, noise_stddev=noise_stddev)
cache_dir = os.path.join(root_dir, name)
tf_log_dir = os.path.join(cache_dir, "tf_logs")
generate_unit_autoencoders(X_train,
                           X_valid,
                           y_train,
                           y_valid,
                           scaler,
                           n_neurons_range,
                           n_folds=n_folds,
                           hidden_activation=hidden_activation,
                           noise_stddev=noise_stddev,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           checkpoint_steps=checkpoint_steps,
                           seed=seed,
                           optimizer=optimizer,
                           n_observable_hidden_neurons=n_observable_hidden_neurons,
                           n_reconstructed_examples_per_class_to_plot=n_reconstructed_examples_per_class_to_plot,
                           cache_dir=cache_dir,
                           tf_log_dir=tf_log_dir)
