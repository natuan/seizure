import numpy as np
import os
import sys
import pandas as pd
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from Denoise import UnitAutoencoder

from Utils import dicts_equal

from Classifier import Classifier

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
n_neurons = 200
n_epochs = 1000
noise_stddev = 0.1
model_path = os.path.join(root_dir,"Exp03","neuron{}_epoch{}.ckpt".format(n_neurons, n_epochs))
tf_log_dir = os.path.join(root_dir, "Exp03", "tf_logs")
autoencoder = UnitAutoencoder("MyAutoencoder", n_inputs, n_neurons, noise_stddev = noise_stddev, tf_log_dir = tf_log_dir)

# For training from scratch
#feature_extractor = FeatureExtractor(X_train, y_train, extractor=autoencoder) # (0,1) min-max scaler
#feature_extractor.fit(n_epochs = n_epochs, model_path=model_path)

# For restoring trained model
feature_extractor = FeatureExtractor(X_train, y_train, extractor=autoencoder, model_path = model_path) # (0,1) min-max scaler

# Using the learned feature extractor to extract the coding of the training set
# This provides input to the classifier
X_train_coded = feature_extractor.codings(X_train)
X_test_coded = feature_extractor.codings(X_test)

svc = SVC(random_state = 42)
knn = KNeighborsClassifier()
gradient_boost = GradientBoostingClassifier(random_state = 42)

for clf in [svc, knn, gradient_boost]:
    print("=== {} ===".format(clf.__class__.__name__))
    classifier = Classifier(classifier = clf)
    train_results = classifier.fit(X_train_coded, y_train)

    print("Training results: ")
    print(train_results)
    test_results = classifier.predict(X_test_coded, y_test)
    
    print("Testing results:")
    print(test_results)

