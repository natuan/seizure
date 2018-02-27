import numpy as np
import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from time import time

from DataSet import *
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

"""
X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@ABCD@E@_ratio{}_rand42_TRAIN.csv".format(ratio)))
X_valid, y_valid = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@ABCD@E@_ratio{}_rand42_VALID.csv".format(ratio)))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@ABCD@E@_ratio{}_rand42_TEST.csv".format(ratio)))

X_train_class_E_indices = [idx for idx, val in enumerate(y_train) if val == 1]
X_train = X_train[X_train_class_E_indices]
y_train = y_train[X_train_class_E_indices]

X_valid_class_E_indices = [idx for idx, val in enumerate(y_valid) if val == 1]
X_valid = X_valid[X_valid_class_E_indices]
y_valid = y_valid[X_valid_class_E_indices]

X_test_class_E_indices = [idx for idx, val in enumerate(y_test) if val == 1]
X_test = X_test[X_test_class_E_indices]
y_test = y_test[X_test_class_E_indices]
"""

# Scaling the train, valid and test sets
signal_range = (np.amin(X_train), np.amax(X_train))
scaling_range = (0, 1)
X_train_scaled = min_max_scale(X_train, signal_range, scaling_range)
X_valid_scaled = min_max_scale(X_valid, signal_range, scaling_range)
X_test_scaled = min_max_scale(X_test, signal_range, scaling_range)

# Initialize coded train, validation and test sets
X_train_codings = None
X_valid_codings = None
X_test_codings = None

# The optimizer
adam_lr = 5 * 1e-6
optimizer = tf.train.AdamOptimizer(adam_lr)

############################################################################
def build_and_train_units():
    """
    Build and train individual unit autoencoders
    """
    noise_stddev = 0.05
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
def build_pretrained_stack(n_hidden_layers, n_neurons_per_layer, noise_stddev = None, dropout_rate = None, unit_model_paths = []):
    # Stack configuration
    name = config_str(prefix="stack_", ratio=ratio, noise_stddev=noise_stddev, dropout_rate=dropout_rate, n_hidden_layers=n_hidden_layers, n_neurons_per_layer=n_neurons_per_layer)
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
    n_epochs = 500 #100000
    batch_size = 64
    n_batches = len(X_train_scaled) // batch_size
    checkpoint_steps = n_batches 
    seed = 0
    n_observable_hidden_neurons_per_layer = 10
    n_hidden_neurons_to_plot = 1.0
    n_reconstructed_examples_per_class_to_plot = 50

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
    X_train_codings = stack.encode(X_train_scaled, file_path=train_file_path)
    X_valid_codings = stack.encode(X_valid_scaled, file_path=valid_file_path)
    X_test_codings = stack.encode(X_test_scaled, file_path=test_file_path)
    assert(X_train_codings.shape[0] == X_train.shape[0]), "Invalid rows"
    assert(X_valid_codings.shape[0] == X_valid.shape[0]), "Invalid rows"
    assert(X_test_codings.shape[0] == X_test.shape[0]), "Invalid rows"
    return stack

def fine_tune_pretrained_stack(stack, X_train, X_valid, y_train, y_valid):
    assert(stack), "Invalid stack"
    n_epochs = 500
    batch_size = 64
    n_batches = len(X_train_scaled) // batch_size
    checkpoint_steps = n_batches
    seed = 0
    stack.stack.fit(X_train, X_valid, y_train, y_valid, model_path=stack.stack_model_path,
              n_epochs=n_epochs, batch_size=batch_size, checkpoint_steps=checkpoint_steps, seed=seed)
   
def performance_metric(y_true, y_predict):
    score = accuracy_score(y_true, y_predict)
    return score

def predict(classifier, X_test, y_test):
    print("Predicting...")
    y_predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    fscore = fbeta_score(y_test, y_predict, beta = .5, average = 'weighted')
    print(">> Accuracy : {}\n".format(accuracy))
    print(">> F-score  : {}\n".format(fscore))
    print(">> Done\n")

def svm_fit_and_classify(X_train, X_test, y_train, y_test):
    print("Searching for best params of SVM...\n")
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    classifier = SVC(random_state = 42)
    params = {'C': [0.5]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(classifier, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(">> Best C value: {}\n".format(best.get_params()['C']))
    print(">> Done\n")
    predict(best, X_test, y_test)

def gradient_boosting_fit_and_classify(X_train, X_test, y_train, y_test):
    print("Searching for best params of GradientBoostingClassifier...\n")
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    classifier = GradientBoostingClassifier()
    params = {'max_depth': [3]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(classifier, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(">> Best max_depth value: {}\n".format(best.get_params()['max_depth']))
    print(">> Done\n")
    predict(best, X_test, y_test)
    
if __name__ == "__main__":
    # build_and_train_units()
    """
    plot_reconstructed_outputs(X_train, y_train, X_train_scaled, size_per_class=20, plot_dir_path="/home/natuan/MyHDD/ml_nano_capstone/tmp/train", seed = 0)
    plot_reconstructed_outputs(X_valid, y_valid, X_valid_scaled, size_per_class=20, plot_dir_path="/home/natuan/MyHDD/ml_nano_capstone/tmp/valid", seed = 0)
    plot_reconstructed_outputs(X_test, y_test, X_test_scaled, size_per_class=20, plot_dir_path="/home/natuan/MyHDD/ml_nano_capstone/tmp/test", seed = 0)
    """
    print("========== BUILDING STACK 1 ============\n")
    stack_1 = build_pretrained_stack(1, 200, noise_stddev=0.05)
    fine_tune_pretrained_stack(stack_1, X_train, X_valid, y_train, y_valid)
    stack_1.stack.restore_and_eval(model_path=stack_1.stack_model_path, X_test, y_test, varlist = ["accuracy"])
    """
    print("========== BUILDING STACK 2 ============\n")
    stack_2, X_train_codings, X_valid_codings, X_test_codings = build_and_train_stack(2, 200, dropout_rate=0.33, unit_model_paths = stack_1.unit_model_paths)
    
    print("========== BUILDING STACK 3 ============\n")
    stack_3, X_train_codings, X_valid_codings, X_test_codings = build_and_train_stack(3, 200, unit_model_paths = stack_2.unit_model_paths)
    """
