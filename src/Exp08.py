import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from Denoise import *

from Utils import dicts_equal

from Classifier import Classifier

from exp_utils import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the original data set, and creating segment data
# using default number of segments and target map
# (See DataSet constructor)
root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
data_set = DataSet(input_dir=os.path.join(root_dir, "input"),
                   cache_dir=os.path.join(root_dir, "cache"))

# Load train and test sets
X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TEST.csv"))

# First unit autoencoder
n_inputs_0 = X_train.shape[1]
n_neurons_0 = 75
hidden_activation_0 = tf.nn.relu
reg_val = 0.0005
regularizer_0 = tf.contrib.layers.l2_regularizer(reg_val) if reg_val is not None else None
noise_stddev_0 = None
dropout_rate_0 = None
n_epochs_0 = 1000
unit_name_0 = "unit0"
config_str_0 = unit_config_str(n_inputs_0, n_neurons_0, hidden_activation_0, reg_val, noise_stddev_0, dropout_rate_0)
unit_long_name_0 = "{}_{}".format(unit_name_0, config_str_0)
model_path_0 = os.path.join(root_dir,"Exp08","{}.ckpt".format(unit_long_name_0))
tf_log_dir = os.path.join(root_dir, "Exp08", "tf_logs")
unit0 = UnitAutoencoder(unit_long_name_0, n_inputs_0, n_neurons_0,
                        hidden_activation=hidden_activation_0,
                        regularizer=regularizer_0,
                        noise_stddev = noise_stddev_0, tf_log_dir = tf_log_dir)

# For training from scratch
scaler = MinMaxScaler(feature_range=(-1,1))

# For constructing and training the network from scratch
#unit0_ext = FeatureExtractor(X_train, y_train, scaler=scaler, extractor=unit0)
#unit0_ext.fit(n_epochs = n_epochs_0, model_path=model_path_0, tfdebug=False)

# For restoring trained model
unit0_ext = FeatureExtractor(X_train, y_train, scaler=scaler, extractor=unit0, model_path = model_path_0)

# The stacked autoencoders
n_epochs = 1000
stack_path = os.path.join(root_dir, "Exp08", "stack")
cache_dir = os.path.join(stack_path, "cache")
tf_log_dir = os.path.join(stack_path, "tf_log")
stack_model_path = os.path.join(stack_path, "stack_model")
stack = stacked_autoencoders("exp08_stack", [unit0], cache_dir=cache_dir, tf_log_dir=tf_log_dir)
stack_ext = FeatureExtractor(X_train, y_train, scaler=scaler, extractor=stack)
stack_ext.fit(model_path=stack_model_path, n_epochs=n_epochs)

"""
sample_indices_to_plot = [0,1, 9100, 9101]
reconstructed_dir_path = os.path.join(root_dir, "Exp08", "plots", "reconstructed_{}".format(unit_long_name_0))
inverse_transform = True if scaler is not None else False
X_outputs = unit0_ext.outputs(X_train,
                              inverse_transform = inverse_transform,
                              sample_indices_to_plot = sample_indices_to_plot,
                              plot_dir_path = reconstructed_dir_path,
                              tfdebug = False)

weights_dir_path = os.path.join(root_dir, "Exp08", "plots", "weights_{}".format(unit_long_name_0))
unit0_ext.extractor.plot_hidden_neurons(weights_dir_path)
"""
