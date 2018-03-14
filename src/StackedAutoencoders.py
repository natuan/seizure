import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import *
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numbers
from sklearn.preprocessing import MinMaxScaler

from Visual import *
from Utils import *

class UnitAutoencoder:
    """
    A single autoencoder which can be either an ordinary or a denoising unit. Denoising autoencoder can be 
    specified with Gaussian noises or dropout on the inputs.
    """
    def __init__(self,
                 name,
                 n_inputs,
                 n_neurons,
                 noise_stddev = None,
                 input_dropout_rate = None,
                 hidden_dropout_rate = None,
                 hidden_activation = tf.nn.softmax,
                 output_activation = None,
                 n_observable_hidden_neurons = 0,
                 regularizer = tf.contrib.layers.l2_regularizer(0.01),
                 initializer = tf.contrib.layers.variance_scaling_initializer(), # He initialization
                 optimizer = tf.train.AdamOptimizer(0.001),
                 tf_log_dir = "../tf_logs"):
        """
        Ctor
        
        Arguments:
        - name: name of the unit
        - n_inputs: number of inputs; also the number of neurons in the output layer
        - n_neurons: number of neurons in the hidden layer
        - noise_stddev: standard deviation of the Gaussian noise; not used if None
        - dropout_rate: if specified a Dropout layer will be added after the input layer; not used if None
        - regularizer: kernel regularizers for hidden and output layers

        Return: None
        """
        self.name = name
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.noise_stddev = noise_stddev
        self.input_dropout_rate = input_dropout_rate
        self.hidden_dropout_rate = hidden_dropout_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.initializer = initializer
        self.n_observable_hidden_neurons = 0
        if (n_observable_hidden_neurons > 0):
            if isinstance(n_observable_hidden_neurons, numbers.Integral):
                self.n_observable_hidden_neurons = min(n_observable_hidden_neurons, self.n_neurons)
            elif isinstance(n_observable_hidden_neurons, numbers.Real):
                assert(0.0 <= n_observable_hidden_neurons <= 1.0), "Invalid ratio"
                self.n_observable_hidden_neurons = int(n_observable_hidden_neurons * self.n_neurons)
            else:
                raise ValueError("Invalid type")
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
            self.training = tf.placeholder_with_default(False, shape=(), name='training')
            if (self.noise_stddev is not None):
                X_noisy = tf.cond(self.training,
                                  lambda: self.X + tf.random_normal(tf.shape(self.X), stddev = self.noise_stddev),
                                  lambda: self.X)
            elif (self.input_dropout_rate is not None):
                X_noisy = tf.layers.dropout(self.X, self.input_dropout_rate, training=self.training)
            else:
                X_noisy = self.X
            dense_hidden = tf.layers.dense(X_noisy, n_neurons, activation=hidden_activation, kernel_regularizer = regularizer, name="{}_hidden".format(self.name))
            if self.hidden_dropout_rate is None:
                self.hidden = dense_hidden
            else:
                self.hidden = tf.layers.dropout(dense_hidden, self.hidden_dropout_rate, training=self.training)
            self.outputs = tf.layers.dense(self.hidden, n_inputs, activation=output_activation, kernel_regularizer = regularizer, name="{}_outputs".format(self.name))
            self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
            self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)
            self.training_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            loss_str = "Reconstruction_and_regularizer loss" if regularizer else "Reconstruction_loss"
            self.loss_summary = tf.summary.scalar(loss_str, self.loss)
            # Ops to observe neurons
            if (self.n_observable_hidden_neurons > 0):
                trainable_vars = dict([(var.name, var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
                hidden_weights = trainable_vars["{}_hidden/kernel:0".format(self.name)]
                assert(hidden_weights.shape == (n_inputs, n_neurons)), "Invalid hidden weight shape"
                hidden_biases = trainable_vars["{}_hidden/bias:0".format(self.name)]
                if self.n_observable_hidden_neurons == self.n_neurons:
                    # Optimization for corner case to avoid permutation
                    neuron_indices = np.arange(self.n_neurons)
                else:
                    neuron_indices = list(np.random.permutation(np.arange(n_neurons))[:self.n_observable_hidden_neurons])
                for neuron_idx in neuron_indices:
                    self._variable_summaries(hidden_weights[:, neuron_idx], "weights_hidden_neuron_{}".format(neuron_idx))
                    self._variable_summaries(hidden_biases[neuron_idx], "bias_hidden_neuron_{}".format(neuron_idx))
                    self._variable_summaries(self.hidden[:, neuron_idx], "activation_hidden_neuron_{}".format(neuron_idx))
            
            self.summary = tf.summary.merge_all()
            tf_log_dir = "{}/{}_run-{}".format(tf_log_dir, self.name, timestr())
            self.train_file_writer = tf.summary.FileWriter(os.path.join(tf_log_dir, "train"), self.graph)
            self.valid_file_writer = tf.summary.FileWriter(os.path.join(tf_log_dir, "valid"), self.graph)
            
        # Dictionary of trainable parameters: key = variable name, values are their values (after training or
        # restored from a model)
        self.params = None

        # The trainable params with initial values (before traininig or restoration)
        self.initial_params = None

        # Stop file: if this file exists, the training will stop
        self.stop_file_path = os.path.join(tf_log_dir, "stop")
        
    def _variable_summaries(self, var, tag):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope(tag):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)    
        
    def fit(self, X_train, X_valid, n_epochs, model_path, save_best_only = True, batch_size = 256, checkpoint_steps = 100, seed = 42, tfdebug = False):
        """
        Train the unit autoencoder against a training set

        Params:
        - X_train: training set of shape (n_samples, n_features)
        - X_valid: validation set
        - n_epochs: number of epochs to train
        - batch_size: batch size
        - checkpoint_steps: number of steps to record checkpoint and logs
        - seed: random seed for tf
        - model_path: model full path file to be saved

        """
        assert(self.X.shape[1] == X_train.shape[1]), "Invalid input shape"
        with tf.Session(graph=self.graph) as sess:
            if tfdebug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if batch_size is None:
                batch_size = len(X_train)
            tf.set_random_seed(seed)
            self.init.run()
            self.initial_params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            best_loss_on_valid_set = 100000
            model_step = -1
            stop = False
            for epoch in tqdm(range(n_epochs)):
                X_train_indices = np.random.permutation(len(X_train))
                n_batches = len(X_train) // batch_size
                start_idx = 0
                for batch_idx in range(n_batches):
                    indices = X_train_indices[start_idx : start_idx + batch_size]
                    X_batch = X_train[indices]
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.training: True})                    
                    step = epoch * n_batches + batch_idx
                    if step % checkpoint_steps == 0:
                        train_summary = sess.run(self.summary, feed_dict={self.X: X_batch})
                        self.train_file_writer.add_summary(train_summary, step)
                        loss_on_valid_set, loss_summary_on_valid_set = sess.run([self.loss, self.loss_summary], feed_dict={self.X: X_valid})
                        self.valid_file_writer.add_summary(loss_summary_on_valid_set, step)
                        model_to_save = (not save_best_only) or (loss_on_valid_set < best_loss_on_valid_set)
                        if loss_on_valid_set < best_loss_on_valid_set:
                            best_loss_on_valid_set = loss_on_valid_set
                        if model_to_save:
                            self.saver.save(sess, model_path)
                            model_step = step
                        # Check if stop signal exists
                        if os.path.exists(self.stop_file_path):
                            stop = True
                    start_idx += batch_size
                if stop:
                    print("Stopping command detected: {}".format(self.stop_file_path))
                    break
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.train_file_writer.close()
            self.valid_file_writer.close()
            assert(model_step >= 0), "Invalid model step"
            all_steps = n_epochs * n_batches
            return model_step, all_steps
        
    def hidden_weights(self, file_path = None):
        assert(self.params is not None), "Invalid self.params"
        w = self.params["{}_hidden/kernel:0".format(self.name)]
        if file_path is not None:
            W = np.array(w)
            columns = ["neuron_{}".format(idx) for idx in range(W.shape[1])]
            df = pd.DataFrame(data=W, columns=columns)
            df.to_csv(file_path)
        return w

    def hidden_biases(self, file_path = None):
        assert(self.params is not None), "Invalid self.params"
        w = self.params["{}_hidden/bias:0".format(self.name)]
        if file_path is not None:
            W = np.array(w)
            columns = ["neuron_{}".format(idx) for idx in range(W.shape[1])]
            df = pd.DataFrame(data=W, columns=columns)
            df.to_csv(file_path)
        return w
    
    def output_weights(self):
        assert(self.params is not None), "Invalid self.params"
        return self.params["{}_output/kernel:0".format(self.name)]

    def output_biases(self):
        assert(self.params is not None), "Invalid self.params"
        return self.params["{}_output/bias:0".format(self.name)]
    
    def restore(self, model_path):
        if self.params is not None:
            print(">> Warning: self.params not empty and will be replaced")
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, model_path)
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

    def eval(self, X, varlist):
        assert(self.params), "Invalid self.params"
        with tf.Session(graph=self.graph) as sess:
            varmap = {"loss": self.loss,
                      "reconstruction_loss": self.reconstruction_loss,
                      "hidden_outputs": self.hidden,
                      "outputs": self.outputs}
            vars_to_eval = []
            for var in varlist:
                # The order of var in the list needs to be kept, thus this for-loop
                if var == "loss":
                    vars_to_eval += [self.loss]
                elif var == "reconstruction_loss":
                    vars_to_eval += [self.reconstruction_loss]
                elif var == "hidden_outputs":
                    vars_to_eval += [self.hidden]
                elif var == "outputs":
                    vars_to_eval += [self.outputs]
                else:
                    raise ValueError("Unrecognized variable {} to evaluate".format(var))
            return sess.run(vars_to_eval, feed_dict={self.X: X})
        
    def restore_and_eval(self, X, model_path, varlist, tfdebug = False):
        """
        Restore model's params and evaluate variables

        Params:
        - model_path: full path to the model file
        - varlist: list of variables to evaluate. Valid values: "loss", "reconstruction_loss", "hidden_outputs", "outputs"

        Return: a list of evaluated variables
        """
        assert(self.graph), "Invalid graph"
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, model_path)
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            if tfdebug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            varmap = {"loss": self.loss,
                      "reconstruction_loss": self.reconstruction_loss,
                      "hidden_outputs": self.hidden,
                      "outputs": self.outputs}
            vars_to_eval = []
            for var in varlist:
                # The order of var in the list needs to be kept, thus this for-loop
                if var == "loss":
                    vars_to_eval += [self.loss]
                elif var == "reconstruction_loss":
                    vars_to_eval += [self.reconstruction_loss]
                elif var == "hidden_outputs":
                    vars_to_eval += [self.hidden]
                elif var == "outputs":
                    vars_to_eval += [self.outputs]
            return sess.run(vars_to_eval, feed_dict={self.X: X})

##################################################################################
##
## StackAutoencoders class
##
##################################################################################
class StackedAutoencoders:
    def __init__(self,
                 name,
                 cache_dir = "../cache",
                 tf_log_dir = "../tf_logs"):
        self.name = name
        self.stacked_units = []
        self.graph = None
        self.initial_params = None
        self.params = None
        self.cache_dir = cache_dir
        self.tf_log_dir = tf_log_dir

        self.X = None
        self.training = None
        self.loss = None
        self.hidden = []  # outputs of all hidden layers
        self.outputs = None # the final output layer
        self.encoders = []
        self.decoders = []
        self.training_op = None
        self.saver = None
        self.loss_summary = None
        self.summary = None
        self.train_file_writer = None        
        self.stop_file_path = os.path.join(cache_dir, "stop")
      
    def _add_hidden_layer(self, input_tensor, unit, layer_name,
                          regularizer,
                          input_dropout_rate,
                          hidden_dropout_rate):
        assert(unit.params), "Invalid unit.params"
        with tf.name_scope(layer_name):
            input_drop = input_tensor if input_dropout_rate is None else tf.layers.dropout(input_tensor, rate=input_dropout_rate, training=self.training)
            weights = tf.Variable(unit.hidden_weights(), name = "weights")
            assert(weights.shape == (unit.n_inputs, unit.n_neurons)), "Wrong assumption about weight's shape"
            biases = tf.Variable(unit.hidden_biases(), name = "biases")
            assert(biases.shape == (unit.n_neurons,)), "Wrong assumption about bias's shape"
            pre_activations = tf.matmul(input_drop, weights) + biases
            if unit.hidden_activation is not None:
                hidden_outputs = unit.hidden_activation(pre_activations, name = "hidden_outputs")
            else:
                hidden_outputs = pre_activations
            hidden_drop = hidden_outputs if hidden_dropout_rate is None else tf.layers.dropout(hidden_outputs, rate=hidden_dropout_rate, training=self.training)
            reg_loss = regularizer(weights) if regularizer else None
            return hidden_drop, reg_loss

    def _add_output_layer(self, input_tensor, unit, layer_name,
                          regularizer,
                          input_dropout_rate,
                          output_dropout_rate):
        assert(unit.params), "Invalid unit.params"
        with tf.name_scope(layer_name):
            input_drop = input_tensor if input_dropout_rate is None else tf.layers.dropout(input_tensor, rate=input_dropout_rate, training=self.training)
            weights = tf.Variable(unit.output_weights(), name = "weights")
            assert(weights.shape == (unit.n_inputs, unit.n_neurons)), "Wrong assumption about weight's shape"
            biases = tf.Variable(unit.output_biases(), name = "biases")
            assert(biases.shape == (unit.n_neurons,)), "Wrong assumption about bias's shape"
            pre_activations = tf.matmul(input_drop, weights) + biases
            if unit.output_activation is not None:
                outputs = unit.output_activation(pre_activations, name = "hidden_outputs")
            else:
                outputs = pre_activations
            outputs_drop = outputs if output_dropout_rate is None else tf.layers.dropout(outputs, rate=output_dropout_rate, training=self.training)
            reg_loss = regularizer(weights) if regularizer else None
            return outputs_drop, reg_loss
        
    def stack_encoder(self, unit, layer_name,
                      regularizer = None,
                      input_dropout_rate = 0,
                      hidden_dropout_rate = 0):
        self.graph = tf.Graph() if self.graph is None else self.graph
        with self.graph.as_default():
            intput_tensor = None
            if not self.hidden: # empty hidden layers
                self.X = tf.placeholder(tf.float32, shape=(None, unit.n_inputs), name="X")
                self.y = tf.placeholder(tf.int64, shape=(None), name="y")
                self.training = tf.placeholder_with_default(False, shape=(), name="training")
                input_tensor = self.X
            else:
                input_tensor = self.hidden[-1]
            hidden, reg_loss = self._add_hidden_layer(input_tensor, unit, layer_name,
                                                      regularizer, input_dropout_rate, hidden_dropout_rate)
            self.hidden += [hidden]
            self.encoders += [hidden]
            self.stacked_units += [unit]
            if reg_loss is not None:
                self.loss = tf.add_n([self.loss, reg_loss]) if self.loss is not None else reg_loss

    def stack_decoder(self, unit, layer_name,
                      is_reconstruction_layer = False,
                      regularizer = None,
                      input_dropout_rate = 0,
                      output_dropout_rate = 0):
        self.graph = tf.Graph() if self.graph is None else self.graph
        with self.graph.as_default():
            assert(self.hidden), "Empty encoder layers"
            input_tensor = self.hidden[-1]
            outputs, reg_loss = self._add_output_layer(input_tensor, unit, layer_name,
                                                       regularizer, input_dropout_rate, output_dropout_rate)
            if reg_loss is not None:
                self.loss = tf.add_n([self.loss, reg_loss]) if self.loss is not None else reg_loss
            if is_reconstruction_layer:
                self.outputs = outputs
                reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
                self.loss = tf.add_n([self.loss, reconstruction_loss]) if self.loss is not None else reconstruction_loss
            else:
                self.hidden += [outputs]
                self.decoders += [outputs]
                
    def stack_softmax_output_layer(self,
                                   layer_name,
                                   kernel_regularizer = None,
                                   kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                                   bias_initializer = tf.zeros_initializer()):
        assert(self.hidden), "Empty hidden layers"
        assert(self.graph), "Empty graph"
        with self.graph.as_default():
            with tf.variable_scope(layer_name):
                weights = tf.get_variable(name="weights",
                                          shape=(self.hidden[-1].shape[1], self.X.shape[1]),
                                          initializer=kernel_initializer)
                biases = tf.get_variable(name="biases",
                                         shape=(self.X.shape[1], ),
                                         initializer=bias_initializer)
                self.outputs = tf.matmul(self.hidden[-1], weights) + biases
            with tf.variable_scope("loss"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
                entropy_loss = tf.reduce_mean(cross_entropy, name="entropy_loss")
                self.loss = tf.add_n([self.loss, entropy_loss]) if self.loss is not None else entropy_loss
                self.loss = tf.add_n([self.loss, kernel_regularizer(weights)]) if kernel_regularizer is not None else self.loss
            
    def finalize(self, optimizer):
        assert(self.graph), "Empty graph"
        with self.graph.as_default():
            with tf.variable_scope("training"):
                self.training_op = optimizer.minimize(self.loss)
            with tf.variable_scope("testing"):
                self.correct = tf.nn.in_top_k(self.outputs, self.y, 1)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            with tf.variable_scope("summary"):
                self.loss_summary = tf.summary.scalar("Loss", self.loss)
                self.summary = tf.summary.merge_all()
                tf_log_dir = "{}/{}_run-{}".format(self.tf_log_dir, self.name, timestr())
                self.train_file_writer = tf.summary.FileWriter(os.path.join(tf_log_dir, "train"), self.graph)
                self.valid_file_writer = tf.summary.FileWriter(os.path.join(tf_log_dir, "valid"), self.graph)
            with tf.variable_scope("global_initializer"):
                self.init = tf.global_variables_initializer()
            with tf.variable_scope("saver"):
                self.saver = tf.train.Saver()

    def save(self, model_path):
        with tf.Session(graph=self.graph) as sess:
            self.init.run()
            self.saver.save(sess, model_path)
        
    def fit(self, X_train, X_valid, y_train, y_valid, model_path, save_best_only = True, n_epochs = 1000, batch_size = 256, checkpoint_steps = 100, seed = 42, tfdebug = False):
        assert(self.training_op is not None), "Invalid self.training_op"
        assert(self.X.shape[1] == X_train.shape[1]), "Invalid input shape"
        with tf.Session(graph=self.graph) as sess:
            if tfdebug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if batch_size is None:
                batch_size = len(X_train)
            tf.set_random_seed(seed)
            self.init.run()
            self.initial_params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            best_loss_on_valid_set = 100000
            model_step = -1
            stop = False
            for epoch in tqdm(range(n_epochs)):
                X_train_indices = np.random.permutation(len(X_train))
                n_batches = len(X_train) // batch_size
                start_idx = 0
                for batch_idx in range(n_batches):
                    indices = X_train_indices[start_idx : start_idx + batch_size]
                    X_batch = X_train[indices]
                    y_batch = y_train[indices]
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch, self.training: True})                    
                    step = epoch * n_batches + batch_idx
                    if step % checkpoint_steps == 0:
                        train_summary = sess.run(self.summary, feed_dict={self.X: X_batch, self.y: y_batch})
                        self.train_file_writer.add_summary(train_summary, step)
                        loss_on_valid_set, loss_summary_on_valid_set = sess.run([self.loss, self.loss_summary],
                                                                                feed_dict={self.X: X_valid, self.y: y_valid})
                        self.valid_file_writer.add_summary(loss_summary_on_valid_set, step)
                        model_to_save = (not save_best_only) or (loss_on_valid_set < best_loss_on_valid_set)
                        if loss_on_valid_set < best_loss_on_valid_set:
                            best_loss_on_valid_set = loss_on_valid_set
                        if model_to_save:
                            self.saver.save(sess, model_path)
                            model_step = step
                        if os.path.exists(self.stop_file_path):
                            stop = True
                    start_idx += batch_size
                if stop:
                    print("Stopping command detected: {}".format(self.stop_file_path))
                    break
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.train_file_writer.close()
            self.valid_file_writer.close()
            assert(model_step >= 0), "Invalid model step"
            all_steps = n_epochs * n_batches
            return model_step, all_steps

    def restore_and_eval(self, model_path, X, y = None, varlist = [], tfdebug = False):
        """
        Restore model's params and evaluate variables

        Arguments:
        - X: the input to be fed into the network
        - varlist: list of variables to evaluate. Valid values: "loss", "reconstruction_loss", "hidden_outputs", "outputs"

        Return: a list of evaluated variables

        TODO: extend X to also accept a list of inputs; useful when evaluate with training and test sets so that 
        the network needs to be restored once
        """
        assert(self.graph), "Invalid graph"
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, model_path)
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            if tfdebug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if not varlist:
                return []
            assert(X is not None), "Invalid input samples"
            varmap = {"loss": self.loss,
                      "hidden_outputs": self.hidden[-1],
                      "outputs": self.outputs,
                      "accuracy": self.accuracy}
            vars_to_eval = []
            for var in varlist:
                # The order of var in the list needs to be kept, thus this for-loop
                if var == "loss":
                    vars_to_eval += [self.loss]
                elif var == "codings":
                    vars_to_eval += [self.encoders[-1]]
                elif var == "outputs":
                    vars_to_eval += [self.outputs]
                elif var == "accuracy":
                    assert(len(X) == len(y)), "Invalid examples and targets sizes"
                    vars_to_eval += [self.accuracy]
            y = np.zeros((len(X), 1)) if y is None else y
            return sess.run(vars_to_eval, feed_dict={self.X: X, self.y: y})
        
    def get_codings(self, model_path, X, file_path = None):
        """
        Compute the codings of an input by the network

        Arguments:
        - X: the input to be fed into the network with shape (n_examples, n_features)
        - file_path (optional): path to a csv file for storing the resulting codings

        Return: the codings of X with shape (n_examples, n_new_features)
        """
        [X_codings] = self.restore_and_eval(model_path, X, varlist=["codings"])
        assert(X.shape[0] == X_codings.shape[0]), "Invalid number of rows in the codings"
        if file_path is not None:
            columns = ["f_{}".format(idx) for idx in range(X_codings.shape[1])]
            df = pd.DataFrame(data=X_codings, columns=columns)
            df.to_csv(file_path)
        return X_codings

        
##################################################################################
##
## StackBuilder class
##
##################################################################################
class StackBuilder:
    """
    Utility class to build a stacked autoencoder with predefined specification
    """
    def __init__(self,
                 name,
                 preceding_units = [],
                 preceding_unit_model_paths = [],
                 n_neurons_per_layer = [],
                 unit_regularizer = [],
                 unit_noise_stddev = [],
                 unit_input_dropout_rate = [],
                 unit_hidden_dropout_rate = [],
                 stack_regularizer=None,
                 stack_input_dropout_rate = 0,
                 stack_hidden_dropout_rate = [],
                 unit_hidden_activations = tf.nn.softmax, # of hidden layers
                 unit_output_activations = None,          # of hidden layers
                 output_activation = tf.nn.softmax, # of output layer
                 output_kernel_regularizer = None,
                 output_kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                 output_bias_initializer = tf.zeros_initializer(),
                 adam_lr = 5*1e-6,
                 cache_dir = "../cache",
                 tf_log_dir = "../tf_logs"):
        """
        Ctor
        
        Arguments:
        - name: name of the stack
        - preceding_units: units that have been trained and will be reused
        - preceding_units_model_path: paths to model file of preceding units
        - n_neurons_per_layer: array of number of hidden neurons after the reused units
        - noise_stddev: array of noise_stddev to be used for new units after the reused ones
        - dropout_rate: similar to noise_stddev array

        """
        self.name = name
        self.preceding_units = preceding_units
        self.preceding_unit_model_paths = preceding_unit_model_paths # for reuse trained unit autoencoders
        assert(len(self.preceding_units) == len(self.preceding_unit_model_paths)), "Invalid preceding units"
        self.n_neurons_per_layer = n_neurons_per_layer
        self.unit_regularizer = unit_regularizer
        self.unit_noise_stddev = unit_noise_stddev
        self.unit_input_dropout_rate = unit_input_dropout_rate
        self.unit_hidden_dropout_rate = unit_hidden_dropout_rate
        self.n_hidden_layers = len(self.n_neurons_per_layer) + len(self.preceding_units)
        if self.n_hidden_layers <= 0:
            raise ValueError("Stack cannot be created empty")
        assert(len(self.unit_regularizer) == len(self.n_neurons_per_layer)), "Invalid noise_stddev array"
        assert(len(self.unit_noise_stddev) == len(self.n_neurons_per_layer)), "Invalid noise_stddev array"
        assert(len(self.unit_input_dropout_rate) == len(self.n_neurons_per_layer)), "Invalid input dropout rate array"
        assert(len(self.unit_hidden_dropout_rate) == len(self.n_neurons_per_layer)), "Invalid hidden dropout rate array"
        self.stack_regularizer=stack_regularizer
        self.stack_input_dropout_rate = stack_input_dropout_rate
        self.stack_hidden_dropout_rate = stack_hidden_dropout_rate
        assert(len(self.stack_hidden_dropout_rate) <= self.n_hidden_layers), "Invalid hidden dropout rate"        
        self.unit_model_paths = [None] * self.n_hidden_layers
        self.units = [None] * self.n_hidden_layers
        self.unit_hidden_activations = unit_hidden_activations
        self.unit_output_activations = unit_output_activations
        self.output_activation = output_activation
        self.output_kernel_regularizer = output_kernel_regularizer
        self.output_kernel_initializer = output_kernel_initializer
        self.output_bias_initializer = output_bias_initializer
        self.adam_lr = adam_lr 
        self.cache_dir = cache_dir
        self.tf_log_dir = tf_log_dir
        self.stack_cache_dir = os.path.join(self.cache_dir, "stack")
        self.stack_tf_log_dir = os.path.join(self.tf_log_dir, "stack")
        self.stack_model_path = os.path.join(self.stack_cache_dir, self.name) + ".model"
        self.stack = None

    def _save_X(self, X, file_path):
        columns = ["f_{}".format(idx) for idx in range(X.shape[1])]
        df = pd.DataFrame(data=X, columns=columns)
        df.to_csv(file_path)

    def get_stack(self):
        return self.stack

    def get_units_and_model_paths(self):
        return self.units, self.unit_model_paths
    
    def build_pretrained_stack(self,
                               X_train,
                               X_valid,
                               y_train,
                               ordinary_stack = False,
                               n_observable_hidden_neurons_per_layer = 0,
                               n_hidden_neurons_to_plot = 20,
                               n_reconstructed_examples_per_class_to_plot = 20,
                               n_epochs = 10000,
                               batch_size = 64,
                               checkpoint_steps = 1000,
                               seed = 42):
        assert(X_train.shape[1] == X_valid.shape[1]), "Invalid input shapes"
        units = []
        X_train_current = X_train
        X_valid_current = X_valid
        rows = {}
        for hidden_layer in range(self.n_hidden_layers):
            unit_name = "Unit_{}".format(hidden_layer)
            unit_cache_dir = os.path.join(self.cache_dir, unit_name)
            if not os.path.exists(unit_cache_dir):
                os.makedirs(unit_cache_dir)
            unit_tf_log_dir = os.path.join(self.tf_log_dir, unit_name)
            if not os.path.exists(unit_tf_log_dir):
                os.makedirs(unit_tf_log_dir)
            n_inputs = X_train_current.shape[1]
            is_preceding_unit = hidden_layer < len(self.preceding_units)
            n_neurons = self.preceding_units[hidden_layer].n_neurons if is_preceding_unit else self.n_neurons_per_layer[hidden_layer - len(self.preceding_units)]
            regularizer = self.preceding_units[hidden_layer].regularizer if is_preceding_unit else self.unit_regularizer[hidden_layer - len(self.preceding_units)]
            noise_stddev = self.preceding_units[hidden_layer].noise_stddev if is_preceding_unit else self.unit_noise_stddev[hidden_layer - len(self.preceding_units)]
            input_dropout_rate = self.preceding_units[hidden_layer].input_dropout_rate if is_preceding_unit else self.unit_input_dropout_rate[hidden_layer - len(self.preceding_units)]
            hidden_dropout_rate = self.preceding_units[hidden_layer].hidden_dropout_rate if is_preceding_unit else self.unit_hidden_dropout_rate[hidden_layer - len(self.preceding_units)]
            unit = UnitAutoencoder(unit_name,
                                   n_inputs,
                                   n_neurons,
                                   regularizer=regularizer,
                                   noise_stddev=noise_stddev,
                                   input_dropout_rate=input_dropout_rate,
                                   hidden_dropout_rate=hidden_dropout_rate,
                                   hidden_activation = self.unit_hidden_activations,
                                   output_activation = self.unit_output_activations,
                                   n_observable_hidden_neurons = n_observable_hidden_neurons_per_layer,
                                   optimizer = tf.train.AdamOptimizer(self.adam_lr),
                                   tf_log_dir = unit_tf_log_dir)
            
            # Try to reuse trained model if specified
            unit_model_path = self.preceding_unit_model_paths[hidden_layer] if hidden_layer < len(self.preceding_units) else os.path.join(unit_cache_dir, "{}.model".format(unit_name))
            if not os.path.exists("{}.meta".format(unit_model_path)):
                print("Training {} for hidden layer {}...\n".format(unit_name, hidden_layer))
                model_step, all_steps = unit.fit(X_train_current,
                                                 X_valid_current,
                                                 n_epochs=n_epochs,
                                                 model_path=unit_model_path,
                                                 batch_size=batch_size,
                                                 checkpoint_steps=checkpoint_steps,
                                                 seed=seed)
                print(">> Done\n")
                print(">> Best model saved at step {} out of {} total steps\n".format(model_step, all_steps))
                print("Plotting reconstructed outputs of unit at hidden layer {}...\n".format(hidden_layer))
                unit_plot_dir = os.path.join(unit_cache_dir, "plots")
                [X_recon] = unit.restore_and_eval(X_train_current, unit_model_path, ["outputs"])
                assert(X_recon.shape == X_train_current.shape), "Invalid output shape"
                unit_reconstructed_dir = os.path.join(unit_plot_dir, "reconstructed")
                plot_reconstructed_outputs(X_train_current, y_train, X_recon, size_per_class=n_reconstructed_examples_per_class_to_plot,
                                           plot_dir_path=unit_reconstructed_dir, seed=seed+10)
                print(">> Done\n")
                print("Plotting hidden weights of unit at hidden layer {}...\n".format(hidden_layer))
                hidden_weights = unit.hidden_weights()
                unit_hidden_weights_dir = os.path.join(unit_plot_dir, "hidden_weights")
                plot_hidden_weights(hidden_weights, n_hidden_neurons_to_plot, unit_hidden_weights_dir, seed =seed+20)
                print(">> Done\n")                
            else:
                print("Reloading model {} of {} for hidden layer {}...\n".format(unit_model_path, unit_name, hidden_layer))
                unit.restore(unit_model_path)
                model_step, all_steps = 0, 0
                print(">> Done\n")
            self.unit_model_paths[hidden_layer] = unit_model_path # This can be passed to subsequent stack built upon this one
            self.units[hidden_layer] = unit
            self._save_X(X_train_current, os.path.join(unit_cache_dir, "X_train_layer_{}.csv".format(hidden_layer)))
            self._save_X(X_valid_current, os.path.join(unit_cache_dir, "X_valid_layer_{}.csv".format(hidden_layer)))
            [train_reconstruction_loss, X_train_current] = unit.restore_and_eval(X_train_current, unit_model_path, ["reconstruction_loss", "hidden_outputs"])
            [valid_reconstruction_loss, X_valid_current] = unit.restore_and_eval(X_valid_current, unit_model_path, ["reconstruction_loss", "hidden_outputs"])
            rows[hidden_layer] = [train_reconstruction_loss, valid_reconstruction_loss, model_step, all_steps, unit_model_path]
            
        print("Stacking up pretrained units...\n")
        self.stack = StackedAutoencoders(name=self.name, cache_dir=self.stack_cache_dir, tf_log_dir=self.stack_tf_log_dir)
        if (not ordinary_stack):
            stack_hidden_layer_names = ["{}_hidden_{}".format(self.name, str(idx)) for idx in range(len(self.units))]
            for idx, unit in enumerate(self.units):
                stack_input_dropout_rate = self.stack_input_dropout_rate if idx == 0 else 0
                stack_hidden_dropout_rate = self.stack_hidden_dropout_rate[idx] if idx < len(self.stack_hidden_dropout_rate) else 0
                stack_regularizer = self.stack_regularizer
                self.stack.stack_encoder(unit, stack_hidden_layer_names[idx],
                                         regularizer=stack_regularizer,
                                         input_dropout_rate=stack_input_dropout_rate,
                                         hidden_dropout_rate=stack_hidden_dropout_rate)
            self.stack.stack_softmax_output_layer(layer_name="{}_softmax_outputs".format(self.name),
                                                  kernel_regularizer=self.output_kernel_regularizer,
                                                  kernel_initializer=self.output_kernel_initializer,
                                                  bias_initializer=self.output_bias_initializer)
        else:
            assert(False), "Not implemented"
            stack_encoder_layer_names = ["{}_encoder_{}".format(self.name, str(idx)) for idx in range(len(self.units))]
            for idx, unit in enumerate(self.units):
                self.stack.stack_encoder(unit, stack_encoder_layer_names[idx])
            stack_decoder_layer_names = ["{}_decoder_{}".format(self.name, str(idx)) for idx in range(len(self.units))]
            for idx, unit in enumerate(reversed(self.units)):
                is_reconstruction_layer = (idx == len(self.units) - 1)
                self.stack.stack_decoder(unit, stack_decoder_layer_names[idx], is_reconstruction_layer=is_reconstruction_layer)
        self.stack.finalize(optimizer=tf.train.AdamOptimizer(self.adam_lr))
        print(">> Done\n")
        print("Saving stack model to {}...\n".format(self.stack_model_path))
        self.stack.save(self.stack_model_path)
        print(">> Done\n")
        result_file_path = os.path.join(self.stack_cache_dir, "hidden_layer_units.csv")
        print("Saving results of building hidden layer units to {}...\n".format(result_file_path))
        columns = ["train_reconstruction_loss", "valid_reconstruction_loss", "step_of_best_model", "all_steps", "unit_model_path"]
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "hidden_layer"
        df.columns = columns
        df.sort_index(inplace=True)
        df.to_csv(result_file_path)
        print(">> Done\n")
                
    
##################################################################################
##
## Supporting functions
##
##################################################################################
def generate_unit_autoencoders(X_train,
                               X_valid,
                               y_train,
                               y_valid,
                               scaler,
                               n_neurons_range,
                               n_folds = 10,
                               noise_stddev = None,
                               dropout_rate = None,
                               hidden_activation = tf.nn.softmax,
                               output_activation = None,
                               regularizer_value = None,
                               initializer = tf.contrib.layers.variance_scaling_initializer(),
                               optimizer = tf.train.AdamOptimizer(0.001),
                               n_epochs = 5000,
                               batch_size = 256,
                               checkpoint_steps = 100,
                               n_observable_hidden_neurons = 1.0,
                               n_hidden_neurons_to_plot = 1.0,
                               n_reconstructed_examples_per_class_to_plot = 20,
                               seed = 0,                       
                               cache_dir = "../cache",
                               tf_log_dir = "../tf_logs"):
    n_inputs = X_train.shape[1]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if noise_stddev is not None:
        prefix = "denoise"
    elif dropout_rate is not None:
        prefix = "dropout"
    else:
        prefix = "ordinary"
    all_run_rows = {}
    all_run_idx = 0
    avg_recon_loss_rows = {}
    all_indices = list(np.random.permutation(len(X_train)))
    fold_sz = len(all_indices) // n_folds
    for n_neurons in n_neurons_range:
        avg_recon_loss = 0
        for fold_idx in range(n_folds):
            unit_name = config_str(prefix,
                                   n_epochs=n_epochs,
                                   n_inputs=n_inputs,
                                   n_neurons=n_neurons,
                                   hidden_activation=hidden_activation,
                                   regularizer_value=regularizer_value,
                                   noise_stddev=noise_stddev,
                                   dropout_rate=dropout_rate)
            unit_name += "_fold{}".format(fold_idx)
            print("\n\n*** Constructing and training unit {}, fold {}/{} ***".format(unit_name, fold_idx+1, n_folds))
            unit_cache_dir = os.path.join(cache_dir, unit_name)
            if not os.path.exists(unit_cache_dir):
                os.makedirs(unit_cache_dir)
            unit_tf_log_dir = tf_log_dir
            if not os.path.exists(unit_tf_log_dir):
                os.makedirs(unit_tf_log_dir)
            unit_regularizer = tf.contrib.layers.l2_regularizer(regularizer_value) if regularizer_value is not None else None
            unit = UnitAutoencoder(unit_name,
                                   n_inputs,
                                   n_neurons,
                                   n_observable_hidden_neurons=n_observable_hidden_neurons,
                                   noise_stddev=noise_stddev,
                                   dropout_rate=dropout_rate,
                                   hidden_activation=hidden_activation,
                                   output_activation=output_activation,
                                   regularizer=unit_regularizer,
                                   initializer=initializer,
                                   optimizer=optimizer,                               
                                   tf_log_dir=unit_tf_log_dir)
            unit_model_path = os.path.join(unit_cache_dir, "{}.model".format(unit_name))
            fold_start_idx = int(fold_idx * fold_sz)
            fold_end_idx = min(fold_start_idx + fold_sz, len(all_indices))
            if fold_end_idx - fold_start_idx == len(all_indices):
                X_train_indices = range(len(all_indices))
            else:
                X_train_indices = all_indices[:fold_start_idx] + all_indices[fold_end_idx:]
            X_train_scaled = scaler.fit_transform(X_train[X_train_indices])
            X_valid_scaled = scaler.transform(X_valid)
            model_step = unit.fit(X_train_scaled,
                                  X_valid_scaled,
                                  n_epochs=n_epochs,
                                  model_path=unit_model_path,
                                  batch_size=batch_size,
                                  checkpoint_steps=checkpoint_steps,
                                  seed=seed)
            print("\n>> Model {} saved at step {}\n".format(unit_model_path, model_step))
            [reconstruction_loss, outputs] = unit.restore_and_eval(X_train_scaled, unit_model_path, ["reconstruction_loss", "outputs"])
            all_run_row = [n_neurons, fold_idx, reconstruction_loss, unit_name]
            all_run_rows[all_run_idx] = all_run_row
            all_run_idx += 1
            assert(outputs.shape == X_train_scaled.shape), "Invalid output shape"
            unit_plot_dir = os.path.join(unit_cache_dir, "plots")
            unit_reconstructed_dir = os.path.join(unit_plot_dir, "reconstructed")
            X_recon = scaler.inverse_transform(outputs)
            plot_reconstructed_outputs(X_train[X_train_indices], y_train[X_train_indices], X_recon, size_per_class=n_reconstructed_examples_per_class_to_plot,
                                       plot_dir_path=unit_reconstructed_dir, seed=seed+10)
            hidden_weights = unit.hidden_weights()
            unit_hidden_weights_dir = os.path.join(unit_plot_dir, "hidden_weights")
            plot_hidden_weights(hidden_weights, n_hidden_neurons_to_plot, unit_hidden_weights_dir, seed =seed+20)

            # Cross validation on the remaining examples
            X_remaining_indices = all_indices[fold_start_idx:fold_end_idx]
            X_remaining_scaled = scaler.transform(X_train[X_remaining_indices])
            [valid_reconstruction_loss] = unit.restore_and_eval(X_remaining_scaled, unit_model_path, ["reconstruction_loss"])
            avg_recon_loss += valid_reconstruction_loss
        avg_recon_loss /= n_folds
        avg_recon_loss_rows[n_neurons] = [avg_recon_loss]
            
    columns = ["n_neurons", "fold_idx", "reconstruction_loss", "model_name"]
    all_runs_df = pd.DataFrame.from_dict(all_run_rows, orient="index")
    all_runs_df.index.name = "Idx"
    all_runs_df.columns = columns
    result_file_path = os.path.join(cache_dir, "results_all_runs.csv")
    if os.path.exists(result_file_path):
        existing_df = pd.DataFrame.from_csv(result_file_path, index_col = 0)
        all_runs_df = all_runs_df.append(existing_df)
    all_runs_df.sort_index(inplace=True)
    all_runs_df.to_csv(result_file_path)

    columns = ["reconstruction_loss"]
    avg_df = pd.DataFrame.from_dict(avg_recon_loss_rows, orient="index")
    avg_df.index.name = "n_neurons"
    avg_df.columns = columns
    result_file_path = os.path.join(cache_dir, "results_avg.csv")
    if os.path.exists(result_file_path):
        existing_df = pd.DataFrame.from_csv(result_file_path, index_col = 0)
        avg_df = avg_df.append(existing_df)
    avg_df.sort_index(inplace=True)
    avg_df.to_csv(result_file_path)
    
