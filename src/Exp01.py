import numpy as np
import os
import sys
import pandas as pd
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from Classifier import Classifier

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the original data set, and creating segment data
# using default number of segments and target map
# (See DataSet constructor)
data_set = DataSet(input_dir="/home/natuan/MyHDD/ml_nano_capstone/input",
                   cache_dir="/home/natuan/MyHDD/ml_nano_capstone/cache")

# Load train and test sets
X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TEST.csv"))

# Train a feature extractor
feature_extractor = FeatureExtractor() # (0,1) min-max scaler, raw input
feature_extractor.fit(X_train, n_epochs = 1000)

# Using the learned feature extractor to extract the coding of the training set
# This provides input to the classifier
X_train_coded = feature_extractor.codings(X_train)

svc = SVC(random_state = 42)
knn = KNeighborsClassifier()
gradient_boost = GradientBoostingClassifier(random_state = 42)

for clf in [svc, knn, gradient_boost]:
    print("=== {} ===".format(clf.__class__.__name__))
    classifier = Classifier(classifier = clf)
    train_results = classifier.fit(X_train_coded, y_train)

    print("Training results: ")
    print(train_results)

    X_test_coded = feature_extractor.codings(X_test)
    test_results = classifier.predict(X_test_coded, y_test)
    
    print("Testing results:")
    print(test_results)
