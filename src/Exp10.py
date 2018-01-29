import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from StackedAutoencoders import *

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

# For training from scratch
scaler = MinMaxScaler(feature_range=(-1,1))

#####################################################
# First unit autoencoder
#####################################################
n_inputs_0 = X_train.shape[1]
n_neurons_0 = 200
hidden_activation_0 = tf.nn.relu
reg_val = None
regularizer_0 = tf.contrib.layers.l2_regularizer(reg_val) if reg_val is not None else None
noise_stddev_0 = None
dropout_rate_0 = 0.1
n_epochs_0 = 1000
unit_name_0 = "unit_0"
config_str_0 = unit_config_str(n_inputs_0, n_neurons_0, hidden_activation_0, reg_val, noise_stddev_0, dropout_rate_0)
unit_long_name_0 = "{}_{}".format(unit_name_0, config_str_0)
model_dir_0 = os.path.join(root_dir, "Exp10", "{}".format(unit_name_0))
model_path_0 = os.path.join(model_dir_0, "{}.ckpt".format(unit_long_name_0))
tf_log_dir = os.path.join(model_dir_0, "tf_logs")
unit_0 = UnitAutoencoder(unit_long_name_0, n_inputs_0, n_neurons_0,
                        hidden_activation=hidden_activation_0,
                        regularizer=regularizer_0,
                         noise_stddev = noise_stddev_0,
                         dropout_rate=dropout_rate_0, tf_log_dir = tf_log_dir)

# For constructing and training the network from scratch
unit_0_ext = FeatureExtractor(X_train, scaler=scaler, extractor=unit_0)
print("Training {}...".format(unit_name_0))
unit_0_ext.fit(n_epochs = n_epochs_0, model_path=model_path_0, tfdebug=False)
print(">> Done")

# For restoring trained model
"""
print("Restoring {}...".format(unit_name_0))
unit_0_ext = FeatureExtractor(X_train, scaler=scaler, extractor=unit_0, model_path = model_path_0)
print(">> Done\n")
"""


weights_dir_path = os.path.join(model_dir_0, "plots", "weights")
unit_0_ext.extractor.plot_hidden_neurons(weights_dir_path)

import pdb
pdb.set_trace()
#######################################################
# Second unit autoencoder
#######################################################
n_inputs_1 = n_neurons_0
n_neurons_1 = 200
hidden_activation_1 = tf.nn.relu
reg_val = None
regularizer_1 = tf.contrib.layers.l2_regularizer(reg_val) if reg_val is not None else None
noise_stddev_1 = None
dropout_rate_1 = 0.1
n_epochs_1 = 1000
unit_name_1 = "unit_1"
config_str_1 = unit_config_str(n_inputs_1, n_neurons_1, hidden_activation_1, reg_val, noise_stddev_1, dropout_rate_1)
unit_long_name_1 = "{}_{}".format(unit_name_1, config_str_1)
model_dir_1 = os.path.join(root_dir, "Exp10", "{}".format(unit_name_1))
model_path_1 = os.path.join(model_dir_1, "{}.ckpt".format(unit_long_name_1))
tf_log_dir = os.path.join(model_dir_1, "tf_logs")
unit_1 = UnitAutoencoder(unit_long_name_1, n_inputs_1, n_neurons_1,
                        hidden_activation=hidden_activation_1,
                        regularizer=regularizer_1,
                         noise_stddev = noise_stddev_1,
                         dropout_rate=dropout_rate_1, tf_log_dir = tf_log_dir)

X_coded_0 = unit_0_ext.codings(X_train)

# For constructing and training the network from scratch

unit_1_ext = FeatureExtractor(X_coded_0, scaler=None, extractor=unit_1)
print("Training {}...".format(unit_name_1))
unit_1_ext.fit(n_epochs = n_epochs_1, model_path=model_path_1, tfdebug=False)
print(">> Done\n")

# For restoring trained model
"""
print("Restoring {}...".format(unit_name_1))
unit_1_ext = FeatureExtractor(X_coded_0, scaler=None, extractor=unit_1, model_path = model_path_1)
print(">> Done\n")
"""

#######################################################
# Third unit autoencoder
#######################################################
n_inputs_2 = n_neurons_0
n_neurons_2 = 200
hidden_activation_2 = tf.nn.relu
reg_val = None
regularizer_2 = tf.contrib.layers.l2_regularizer(reg_val) if reg_val is not None else None
noise_stddev_2 = None
dropout_rate_2 = 0.1
n_epochs_2 = 1000
unit_name_2 = "unit_2"
config_str_2 = unit_config_str(n_inputs_2, n_neurons_2, hidden_activation_2, reg_val, noise_stddev_2, dropout_rate_2)
unit_long_name_2 = "{}_{}".format(unit_name_2, config_str_2)
model_dir_2 = os.path.join(root_dir, "Exp10", "{}".format(unit_name_2))
model_path_2 = os.path.join(model_dir_2, "{}.ckpt".format(unit_long_name_2))
tf_log_dir = os.path.join(model_dir_2, "tf_logs")
unit_2 = UnitAutoencoder(unit_long_name_2, n_inputs_2, n_neurons_2,
                         hidden_activation=hidden_activation_2,
                         regularizer=regularizer_2,
                         noise_stddev = noise_stddev_2,
                         dropout_rate=dropout_rate_2, tf_log_dir = tf_log_dir)

X_coded_1 = unit_1_ext.codings(X_coded_0)

# For constructing and training the network from scratch
unit_2_ext = FeatureExtractor(X_coded_1, scaler=None, extractor=unit_2)
print("Training {}...".format(unit_name_2))
unit_2_ext.fit(n_epochs = n_epochs_2, model_path=model_path_2, tfdebug=False)
print(">> Done\n")

# For restoring trained model
"""
print("Restoring {}...".format(unit_name_2))
unit_2_ext = FeatureExtractor(X_coded_1, scaler=None, extractor=unit_2, model_path = model_path_2)
print(">> Done\n")
"""

##########################################################
# The stacked autoencoders
##########################################################
n_epochs = 1000
stack_path = os.path.join(root_dir, "Exp10", "stack")
cache_dir = os.path.join(stack_path, "cache")
tf_log_dir = os.path.join(stack_path, "tf_logs")
stack_model_path = os.path.join(stack_path, "unit_0_unit_1_unit_2")
stack = stacked_autoencoders("Exp10_stack", [unit_0, unit_1, unit_2], cache_dir=cache_dir, tf_log_dir=tf_log_dir)

stack_ext = FeatureExtractor(X_train, scaler=scaler, extractor=stack)
print("Training {}...".format(stack_model_path))
stack_ext.fit(model_path=stack_model_path, n_epochs=n_epochs)
print(">> Done\n")

#stack_ext = FeatureExtractor(X_train, y_train, scaler=scaler, extractor=stack, model_path = stack_model_path)

sample_indices_to_plot = [0,1, 9100, 9101]
reconstructed_dir_path = os.path.join(stack_path, "plots", "reconstructed_outputs")
inverse_transform = True if scaler is not None else False
X_outputs = stack_ext.outputs(X_train,
                              inverse_transform = inverse_transform,
                              sample_indices_to_plot = sample_indices_to_plot,
                              plot_dir_path = reconstructed_dir_path,
                              tfdebug = False)
