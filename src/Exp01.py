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
data_set = DataSet("../input")

# Load train and test sets
data_set.load_train(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))
data_set.load_test(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TEST.csv"))

# Train a feature extractor
feature_extractor = FeatureExtractor(data_set.train_df) # (0,1) min-max scaler, raw input
feature_extractor.fit()

# Using the learned feature extractor to extract the coding of the training set
# This provides input to the classifier
X_train, y_train = feature_extractor.eval()

svc = SVC(random_state = 42)
knn = KNeighborsClassifier()
gradient_boost = GradientBoostingClassifier(random_state = 42)

for clf in [svc, knn, gradient_boost]:
    print("=== {} ===".format(clf.__class__.__name__))
    classifier = Classifier(classifier = clf)
    train_results = classifier.fit(X_train, y_train)

    print("Training results: ")
    print(train_results)

    X_test, y_test = feature_extractor.eval(data_set.test_df)
    test_results = classifier.predict(X_test, y_test)
    
    print("Testing results:")
    print(test_results)
