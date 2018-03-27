import os
import tensorflow as tf
from datetime import *

def unit_config_str(prefix,
                    n_inputs,
                    n_neurons,
                    hidden_activation = None,
                    regularizer_value = None,
                    noise_stddev = None,
                    dropout_rate = None):
    if prefix is not None:
        s = "{}_inputs{}_neurons{}".format(prefix, n_inputs, n_neurons)
    else:
        s = "inputs{}_neurons{}".format(n_inputs, n_neurons)
    if hidden_activation is not None:
        if hidden_activation == tf.nn.softmax:
            s += "_softmax"
        elif hidden_activation == tf.nn.elu:
            s += "_elu"
        elif hidden_activation == tf.nn.relu:
            s += "_relu"
        elif hidden_activation == tf.nn.tanh:
            s += "_tanh"
        else:
            raise ValueError("Invalid hidden activation function")
    if regularizer_value is not None: 
        s += "_reg{}".format(regularizer_value)
    if noise_stddev is not None:
        s += "_noise{}".format(noise_stddev)
    if dropout_rate is not None:
        s += "_drop{}".format(dropout_rate)
    return s

def config_str(prefix = None,
               ratio = None,
               min_scale = None,
               max_scale = None,
               adam_lr = None,
               n_epochs = None,
               n_folds = None,
               n_inputs = None,
               n_neurons = None,
               n_hidden_layers = None,
               n_neurons_per_layer = None,
               hidden_activation = None,
               regularizer_value = None,
               noise_stddev = None,
               dropout_rate = None):
    s = "{}".format(prefix) if prefix is not None else ""
    if ratio is not None:
        s += "_ratio{}".format(ratio)
    if min_scale is not None and max_scale is not None:
        s += "_scale{}to{}".format(min_scale, max_scale)
    if adam_lr is not None:
        s += "_adam{}".format(adam_lr)
    if n_epochs is not None:
        s += "_epoch{}".format(n_epochs)
    if n_inputs is not None:
        s += "_inputs{}".format(n_inputs)
    if n_neurons is not None:
        s += "_neurons{}".format(n_neurons)
    if n_neurons_per_layer is not None:
        s += "_neuronslayer{}".format(n_neurons_per_layer)
    if n_hidden_layers is not None:
        s += "_hiddenlayers{}".format(n_hidden_layers)
    if hidden_activation is not None:
        if hidden_activation == tf.nn.softmax:
            s += "_softmax"
        elif hidden_activation == tf.nn.elu:
            s += "_elu"
        elif hidden_activation == tf.nn.relu:
            s += "_relu"
        elif hidden_activation == tf.nn.tanh:
            s += "_tanh"
        else:
            raise ValueError("Invalid hidden activation function")
    if regularizer_value is not None: 
        s += "_reg{}".format(regularizer_value)
    if noise_stddev is not None:
        s += "_noise{}".format(noise_stddev)
    if dropout_rate is not None:
        s += "_drop{}".format(dropout_rate)
    if n_folds is not None:
        s += "_folds{}".format(n_folds)
    return s

# Create a time string
def timestr():
    today = datetime.today()
    month_dict = {1:"jan", 2:"feb", 3:"mar", 4:"apr", 5:"may", 6:"jun", 7:"jul", 8:"aug", 9:"sep", 10:"oct", 11:"nov", 12:"dec"}
    return "{}{}_{}h{}".format(month_dict[today.month], today.day, today.hour, today.minute)
