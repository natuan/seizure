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
ratio = 0.2

X_train, y_train = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand25_TRAIN.csv".format(ratio)))
X_valid, y_valid = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand25_VALID.csv".format(ratio)))
X_test, y_test = data_set.load_features_and_target(os.path.join(data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio{}_rand25_TEST.csv".format(ratio)))

# Scaling the train, valid and test sets
signal_range = (np.amin(X_train), np.amax(X_train))
scaling_range = (-1, 1)
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

############################################################################################
##
## Experiment with stacking autoencoders
##
############################################################################################
def build_pretrained_stack(name, force_rename=False, ordinary_stack = False,
                           preceding_units=[], preceding_unit_model_paths=[], n_neurons_per_layer=[],
                           unit_regularizer=[],
                           unit_noise_stddev=[],
                           unit_hidden_activations = tf.nn.elu,
                           unit_input_dropout_rate=[],
                           unit_hidden_dropout_rate=[],
                           stack_regularizer=[],
                           stack_input_dropout_rate=0,
                           stack_hidden_dropout_rate=[]):
    # Stack configuration
    cache_dir = os.path.join(root_dir, name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    elif force_rename:
        raise ValueError("Folder {} already exists. Rename the stack!".format(name))
    tf_log_dir = os.path.join(cache_dir, "tf_logs")
    if not os.path.exists(tf_log_dir):
        os.makedirs(tf_log_dir)
    stack_builder = StackBuilder(name,
                                 preceding_units=preceding_units,
                                 preceding_unit_model_paths= preceding_unit_model_paths,
                                 n_neurons_per_layer=n_neurons_per_layer,
                                 unit_regularizer=unit_regularizer,
                                 unit_noise_stddev=unit_noise_stddev,
                                 unit_hidden_activations=unit_hidden_activations,       # ELU
                                 unit_input_dropout_rate=unit_input_dropout_rate,
                                 unit_hidden_dropout_rate=unit_hidden_dropout_rate,
                                 stack_regularizer=stack_regularizer,
                                 stack_input_dropout_rate=stack_input_dropout_rate,
                                 stack_hidden_dropout_rate=stack_hidden_dropout_rate,
                                 cache_dir=cache_dir,
                                 tf_log_dir=tf_log_dir)

    # Training configuration
    n_epochs = 100000
    batch_size = 64
    n_batches = len(X_train_scaled) // batch_size
    checkpoint_steps = 5*n_batches 
    seed = 0
    n_observable_hidden_neurons_per_layer = 50
    n_hidden_neurons_to_plot = 0.5
    n_reconstructed_examples_per_class_to_plot = 50

    stack_builder.build_pretrained_stack(X_train_scaled,
                                         X_valid_scaled,
                                         y_train,
                                         ordinary_stack=ordinary_stack,
                                         n_observable_hidden_neurons_per_layer=n_observable_hidden_neurons_per_layer,
                                         n_hidden_neurons_to_plot=n_hidden_neurons_to_plot,
                                         n_reconstructed_examples_per_class_to_plot=n_reconstructed_examples_per_class_to_plot,
                                         n_epochs=n_epochs,
                                         batch_size=batch_size,
                                         checkpoint_steps=checkpoint_steps,
                                         seed=seed)
    for hidden_layer in range(stack_builder.n_hidden_layers):
        train_file_path = os.path.join(cache_dir, "train_codings_hiddenlayer{}of{}.csv".format(hidden_layer+1, stack_builder.n_hidden_layers))
        valid_file_path = os.path.join(cache_dir, "valid_codings_hiddenlayer{}of{}.csv".format(hidden_layer+1, stack_builder.n_hidden_layers))
        test_file_path = os.path.join(cache_dir, "test_codings_hiddenlayer{}of{}.csv".format(hidden_layer+1, stack_builder.n_hidden_layers))
        stack = stack_builder.get_stack()
        stack.get_codings(stack_builder.stack_model_path, X_train_scaled, file_path=train_file_path)
        stack.get_codings(stack_builder.stack_model_path, X_valid_scaled, file_path=valid_file_path)
        stack.get_codings(stack_builder.stack_model_path, X_test_scaled, file_path=test_file_path)
    return stack_builder

def fine_tune_pretrained_stack(stack_builder, X_train, X_valid, y_train, y_valid):
    n_epochs = 300000
    batch_size = 64
    n_batches = len(X_train_scaled) // batch_size
    checkpoint_steps = 5 * n_batches
    seed = 0
    print("Fine tuning...\n")
    model_step, all_steps = stack_builder.stack.fit(X_train, X_valid, y_train, y_valid,model_path=stack_builder.stack_model_path,
                                                    n_epochs=n_epochs, batch_size=batch_size, checkpoint_steps=checkpoint_steps, seed=seed)
    print(">> Done\n")
    print(">> Best model saved at step {} out of {} total steps\n".format(model_step, all_steps))

    
############################################################################################
##
## Utils
##
############################################################################################
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

    print("========== BUILDING STACK 1 ============\n")
    name = "stack_50_50_dropout_elu_clean_input"
    preceding_units=[]
    preceding_unit_model_paths = []
    n_neurons_per_layer = [50, 50]
    unit_regularizer = [None] * len(n_neurons_per_layer)
    unit_noise_stddev = [None] * len(n_neurons_per_layer)
    unit_hidden_activations = tf.nn.elu
    unit_input_dropout_rate = [None] * len(n_neurons_per_layer)
    unit_hidden_dropout_rate = [0.5] * len(n_neurons_per_layer)
    stack_regularizer = None
    stack_input_dropout_rate = None # for stack
    stack_hidden_dropout_rate = [0.5] * len(n_neurons_per_layer) # for stack
    print("Start: Build pretrained stack...\n")
    stack_builder_1 = build_pretrained_stack(name,
                                             preceding_units=preceding_units,
                                             preceding_unit_model_paths=preceding_unit_model_paths,
                                             n_neurons_per_layer=n_neurons_per_layer,
                                             unit_regularizer=unit_regularizer,
                                             unit_noise_stddev=unit_noise_stddev,
                                             unit_hidden_activations = tf.nn.elu,
                                             unit_input_dropout_rate=unit_input_dropout_rate,
                                             unit_hidden_dropout_rate=unit_hidden_dropout_rate,
                                             stack_regularizer=stack_regularizer,
                                             stack_input_dropout_rate=stack_input_dropout_rate,
                                             stack_hidden_dropout_rate=stack_hidden_dropout_rate)
    print("End: Build pretrained stack 1\n")
    
    print("Start: Fine tuning the pretrained stack...\n")
    fine_tune_pretrained_stack(stack_builder_1, X_train, X_valid, y_train, y_valid)
    print("End: Fine tuning the pretrained stack 1\n")   
    stack_1 = stack_builder_1.get_stack()
    
    print("Start: Checking stack weights before learning...")
    stack_initial_params = stack_1.initial_params
    for k in stack_initial_params.keys():
        for idx, unit in enumerate(stack_builder_1.units):
            if np.array_equal(stack_initial_params[k], unit.hidden_weights()):
                print(">> Values at key {} == hidden weights of unit {}\n".format(k, idx))
            elif np.array_equal(stack_initial_params[k], unit.hidden_biases()):
                print(">> Values at key {} == biases of unit {}\n".format(k, idx))
    print("End: Checking stack weights before learning\n")
    
    train_accuracy = stack_1.restore_and_eval(model_path=stack_builder_1.stack_model_path, X=X_train, y=y_train, varlist = ["accuracy"])
    print("Train accuracy by stack 1: {}".format(train_accuracy))
    valid_accuracy = stack_1.restore_and_eval(model_path=stack_builder_1.stack_model_path, X=X_valid, y=y_valid, varlist = ["accuracy"])
    print("Valid accuracy by stack 1: {}".format(valid_accuracy))
    test_accuracy = stack_1.restore_and_eval(model_path=stack_builder_1.stack_model_path, X=X_test, y=y_test, varlist = ["accuracy"])
    print("Test accuracy by stack 1: {}".format(test_accuracy))

    """
    print("========== BUILDING STACK 2 ============\n")
    name = "stack_50_50"
    preceding_units, preceding_unit_model_paths = stack_builder_1.get_units_and_model_paths()
    n_neurons_per_layer = [50]
    noise_stddev = [0.05] * len(n_neurons_per_layer)
    dropout_rate = [None] * len(n_neurons_per_layer)
    stack_builder_2 = build_pretrained_stack(name,
                                             preceding_units=preceding_units,
                                             preceding_unit_model_paths=preceding_unit_model_paths,
                                             n_neurons_per_layer=n_neurons_per_layer,
                                             noise_stddev=noise_stddev,
                                             dropout_rate=dropout_rate)
    print("End: Build pretrained stack 2\n")
    print("Start: Fine tuning the pretrained stack...\n")
    fine_tune_pretrained_stack(stack_builder_2, X_train, X_valid, y_train, y_valid)
    print("End: Fine tuning the pretrained stack 2\n")
    stack_2 = stack_builder_2.get_stack()
    accuracy = stack_2.restore_and_eval(model_path=stack_builder_2.stack_model_path, X=X_test, y=y_test, varlist = ["accuracy"])
    print("Accuracy by stack 2: {}".format(accuracy))
    """

    """
    print("========== BUILDING STACK 3 ============\n")
    stack_3, X_train_codings, X_valid_codings, X_test_codings = build_and_train_stack(3, 200, unit_model_paths = stack_2.unit_model_paths)
    """
