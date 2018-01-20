import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from Denoise import UnitAutoencoder

from Utils import dicts_equal

from Classifier import Classifier

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

# Train an ordinary autoencoder (i.e., clean inputs, fewer hidden neurons than input)
n_inputs = X_train.shape[1]
n_neurons = 350
n_epochs = 10000
noise_stddev = 0.2
hidden_activation = tf.nn.relu
scaler = MinMaxScaler(feature_range=(-1,1))
regularizer = None
model_path = os.path.join(root_dir,"Exp06","neuron{}_epoch{}_relu.ckpt".format(n_neurons, n_epochs))
tf_log_dir = os.path.join(root_dir, "Exp06", "tf_logs")
autoencoder = UnitAutoencoder("MyAutoencoder", n_inputs, n_neurons,
                              hidden_activation=hidden_activation,
                              regularizer=regularizer,
                              noise_stddev = noise_stddev, tf_log_dir = tf_log_dir)

# For training from scratch
feature_extractor = FeatureExtractor(X_train, y_train, scaler=scaler, extractor=autoencoder)
feature_extractor.fit(n_epochs = n_epochs, model_path=model_path, tfdebug=False)

# For restoring trained model
#feature_extractor = FeatureExtractor(X_train, y_train, scaler=scaler, extractor=autoencoder, model_path = model_path) # (0,1) min-max scaler

sample_indices_to_plot = [0,1, 9100, 9101]
plot_dir_path = os.path.join(root_dir, "Exp06", "plots")
inverse_transform = True if scaler is not None else False
#import pdb
#pdb.set_trace()
X_outputs = feature_extractor.outputs(X_train,
                                      inverse_transform = inverse_transform,
                                      sample_indices_to_plot = sample_indices_to_plot,
                                      plot_dir_path = plot_dir_path,
                                      tfdebug = False)

weights_dir_path = os.path.join(root_dir, "Exp06", "plots_hidden_weights")
feature_extractor.extractor.plot_hidden_neurons(weights_dir_path)
