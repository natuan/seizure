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
X_scaled = scaler.fit_transform(X_train) if (scaler) else X_train

min_reg = 0.001
max_reg = 0.005
builder_name = "ordinary_builder_01"
builder_path = os.path.join(root_dir, builder_name)
cache_dir = os.path.join(builder_path, "cache")
tf_log_dir = os.path.join(builder_path, "tf_logs")
builder = StackBuilder(X_scaled,
                       name="builder",
                       unit_type="ordinary",
                       hidden_activation=tf.nn.relu,
                       n_units=2,
                       accepted_reconstruction_loss=0.01,
                       l2_reg_range=(min_reg, max_reg),
                       cache_dir=cache_dir,
                       tf_log_dir=tf_log_dir)
builder.max_search_trials = 1
model_path = os.path.join(cache_dir, "stacked_autoencoder_final_model")
autoencoder = builder.doit(model_path)


