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
        else:
            raise ValueError("Invalid hidden activation function")
    if regularizer_value is not None: 
        s += "_reg{}".format(regularizer_value)
    if noise_stddev is not None:
        s += "_noise{}".format(noise_stddev)
    if dropout_rate is not None:
        s += "_drop{}".format(dropout_rate)
    return s

# Create a time string
def timestr():
    today = datetime.today()
    month_dict = {1:"jan", 2:"feb", 3:"mar", 4:"apr", 5:"may", 6:"jun", 7:"jul", 8:"aug", 9:"sep", 10:"oct", 11:"nov", 12:"dec"}
    return "{}{}_{}:{}".format(month_dict[today.month], today.day, today.hour, today.minute)
