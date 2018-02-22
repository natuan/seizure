import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from StackedAutoencoders import *
from Classifier import Classifier
from Utils import *

# Load the original data set, and creating segment data
# using default number of segments and target map
# (See DataSet constructor)
root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
data_set = DataSet(input_dir=os.path.join(root_dir, "input"),
                   cache_dir=os.path.join(root_dir, "cache"))

# Load train, validation and test sets
ratio = 0.15
X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand0_TRAIN.csv".format(ratio)))
X_valid, y_valid = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand0_VALID.csv".format(ratio)))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand0_TEST.csv".format(ratio)))

# Initialize coded train, validation and test sets
X_train_codings = None
X_valid_codings = None
X_test_codings = None

# The scaler to apply to the training set
min_scale, max_scale = 0, 1
scaler = MinMaxScaler(feature_range=(min_scale, max_scale))

# The optimizer
adam_lr = 5 * 1e-6
optimizer = tf.train.AdamOptimizer(adam_lr)

############################################################################
def build_and_train_units():
    """
    Build and train individual unit autoencoders
    """
    noise_stddev = 0.3
    n_folds = 1
    n_inputs = X_train.shape[1]
    n_neurons_range = [3] #[300]
    hidden_activation = tf.nn.softmax
    n_epochs = 3 #100000
    batch_size = 64
    checkpoint_steps = 5000
    seed = 0
    n_observable_hidden_neurons = 0
    n_hidden_neurons_to_plot = 50
    n_reconstructed_examples_per_class_to_plot = 50

    name = config_str(prefix="denoise_", ratio=ratio, min_scale=min_scale, max_scale=max_scale, adam_lr=adam_lr, n_epochs=n_epochs, n_folds=n_folds,
                      n_inputs=n_inputs, hidden_activation=hidden_activation, noise_stddev=noise_stddev)
    name += "_neurons"
    for n_neurons in n_neurons_range:
        name += "_{}".format(n_neurons)
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
                               n_hidden_neurons_to_plot=n_hidden_neurons_to_plot,
                               n_reconstructed_examples_per_class_to_plot=n_reconstructed_examples_per_class_to_plot,
                               cache_dir=cache_dir,
                               tf_log_dir=tf_log_dir)

############################################################################
def build_and_train_stack(n_hidden_layers, n_neurons_per_layer, unit_model_paths = []):
    # Stack configuration
    noise_stddev = 0.3
    name = config_str(prefix="stack_", ratio=ratio, noise_stddev=noise_stddev, n_hidden_layers=n_hidden_layers, n_neurons_per_layer=n_neurons_per_layer)
    cache_dir = os.path.join(root_dir, name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    tf_log_dir = os.path.join(cache_dir, "tf_logs")
    if not os.path.exists(tf_log_dir):
        os.makedirs(tf_log_dir)
    stack = StackBuilder(name,
                         noise_stddev=noise_stddev,
                         n_hidden_layers=n_hidden_layers,
                         n_neurons_per_layer=n_neurons_per_layer,
                         cache_dir=cache_dir,
                         tf_log_dir=tf_log_dir)

    # Training configuration
    n_epochs = 100000
    batch_size = 64
    checkpoint_steps = 5000
    seed = 0
    n_observable_hidden_neurons_per_layer = 10
    n_hidden_neurons_to_plot = 1.0
    n_reconstructed_examples_per_class_to_plot = 50

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    stack.build_pretrained_stack(X_train_scaled,
                                 X_valid_scaled,
                                 y_train,
                                 unit_model_paths=unit_model_paths,
                                 n_observable_hidden_neurons_per_layer=n_observable_hidden_neurons_per_layer,
                                 n_hidden_neurons_to_plot=n_hidden_neurons_to_plot,
                                 n_reconstructed_examples_per_class_to_plot=n_reconstructed_examples_per_class_to_plot,
                                 n_epochs=n_epochs,
                                 batch_size=batch_size,
                                 checkpoint_steps=checkpoint_steps,
                                 seed=seed)
    train_file_path = os.path.join(cache_dir, "train_codings.csv")
    valid_file_path = os.path.join(cache_dir, "valid_codings.csv")
    test_file_path = os.path.join(cache_dir, "test_codings.csv")
    X_train_codings = stack.encode(X_train_scaled, train_file_path)
    X_valid_codings = stack.encode(X_valid_scaled, valid_file_path)
    X_test_codings = stack.encode(X_test_scaled, test_file_path)
    assert(X_train_codings.shape[0] == X_train.shape[0]), "Invalid rows"
    assert(X_valid_codings.shape[0] == X_valid.shape[0]), "Invalid rows"
    assert(X_test_codings.shape[0] == X_test.shape[0]), "Invalid rows"
    return stack, X_train_codings, X_valid_codings, X_test_codings

"""
def classify(X_train_codings, X_valid_codings, X_test_codings, y_train, y_valid, y_test, n_folds = 10):
    svc = SVC(random_state = 42)
    knn = KNeighborsClassifier()
    gradient_boost = GradientBoostingClassifier(random_state = 42)

    for clf in [svc, knn, gradient_boost]:
        print("=== {} ===".format(clf.__class__.__name__))
        classifier = Classifier(classifier = clf)
        train_results = classifier.fit(X_train_codings, X_valid_codings, y_train, y_valid, n_folds = n_folds)

        print("Training results: ")
        print(train_results)

        X_test_coded = feature_extractor.codings(X_test)
        test_results = classifier.predict(X_test_coded, y_test)

        print("Testing results:")
        print(test_results)
"""

if __name__ == "__main__":
    # build_and_train_units()
    print("========== BUILDING STACK 1 ============\n")
    stack_1, X_train_codings, X_valid_codings, X_test_codings = build_and_train_stack(1, 200)
    print("========== BUILDING STACK 2 ============\n")
    stack_2, X_train_codings, X_valid_codings, X_test_codings = build_and_train_stack(2, 200, unit_model_paths = stack_1.unit_model_paths)
    print("========== BUILDING STACK 3 ============\n")
    stack_3, X_train_codings, X_valid_codings, X_test_codings = build_and_train_stack(3, 200, unit_model_paths = stack_2.unit_model_paths)
