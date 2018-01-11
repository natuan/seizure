import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import *

from Autoencoder import Autoencoder

class UnitAutoencoder(Autoencoder):
    def __init__(self, name, n_inputs, n_neurons, dropout_rate):
        """
        Create an autoencoder that has one hidden layer of neurons
        
        Params:
        - name: name of the unit autoencoder
        - n_inputs: number of inputs; also the number of neurons in the output layer
        - n_neurons: number of neurons in the hidden layer
        - dropout_rate: if specified a Dropout layer will be added after the input layer
        Return: None
        """
        now = datetime.utcnow().strftime("_Y%Y_m%m_d%d_H%H_M%M_S%S")
        root_logdir = "/home/natuan/MyHDD/ml_nano_capstone/tf_logs"
        logdir = "{}/run-{}/".format(root_logdir, now)

        Autoencoder.__init__(self)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.n_inputs = n_inputs
            self.n_neurons = n_neurons
            self.dropout_rate = dropout_rate
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
            if (self.dropout_rate):
                # Denoise autoencoder is used with dropout
                self.training = tf.placeholder_with_default(False, shape=(), name='training')
                self.X_drop = tf.layers.dropout(self.X, self.dropout_rate, training=self.training)
                self.hidden = tf.layers.dense(self.X_drop, self.n_neurons, activation=self.hidden_activation, name="hidden")
            else:
                # Ordinary autoencoder
                self.training = None
                self.X_drop = None
                self.hidden = tf.layers.dense(self.X, self.n_neurons, activation=self.hidden_activation, name="hidden")
            self.outputs = tf.layers.dense(self.hidden, self.n_inputs, activation=self.output_activation, name="outputs")
            self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
            #self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)
            #self.training_op = self.optimizer.minimize(loss)
            self.training_op = self.optimizer.minimize(self.reconstruction_loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.loss_summary = tf.summary.scalar('LOSS', self.reconstruction_loss)
            self.file_writer = tf.summary.FileWriter(logdir, self.graph)
            
        # Trained parameters
        self.params = None
        
    def fit(self, X_train, n_epochs = 100, batch_size = 256, seed = 42):
        assert(self.n_inputs == X_train.shape[1]), "Invalid input shape"
        with tf.Session(graph=self.graph) as sess:
            tf.set_random_seed(seed)
            self.init.run()
            for epoch in tqdm(range(n_epochs)):
                X_train_indices = np.random.permutation(len(X_train))
                n_batches = len(X_train) // batch_size
                start_idx = 0
                for batch_idx in range(n_batches):
                    indices = X_train_indices[start_idx : start_idx + batch_size]
                    X_batch = X_train[indices]
                    if batch_idx % 10 == 0:
                        summary_str = self.loss_summary.eval(feed_dict={self.X: X_batch})
                        step = epoch * n_batches + batch_idx
                        self.file_writer.add_summary(summary_str, step)
                    sess.run(self.training_op, feed_dict={self.X: X_batch})
                    start_idx += batch_size
                # The remaining (less than batch_size) samples
                indices = X_train_indices[start_idx : len(X_train)]
                X_batch = X_train[indices]
                sess.run(self.training_op, feed_dict={self.X: X_batch})                
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.file_writer.close()
            
    def reconstruction_loss(self, X):
        return self.reconstruction_loss.eval(feed_dict={self.X: X})

    def hidden_outputs(self, X):
        return self.hidden.eval(feed_dict={self.X: X})

    def outputs(self, X):
        return self.outputs.eval(feed_dict={self.X: X})
        
class TrainValidationGenerator:
    def __init__(self, X, n_folds):
        self.X = X
        self.n_folds = n_folds
        self.batch_size = len(self.X) // self.n_folds
        self.cur_idx = 0

    def next_train_validation_sets(self):
        """
        Generate the next train and validation sets
        
        Params: None

        Return: train_set, validation_set
        """
        start = self.cur_idx
        end = np.minimum(start + self.batch_size, len(self.X))
        train_set = np.vstack((X[:start], X[end:len(self.X)]))
        validation_set = X[start:end]
        self.cur_idx = end if end < len(self.X) else 0
        return train_set, validation_set
        
class DenoiseAutoencoder(Autoencoder):
    def __init__(self):
        Autoencoder.__init__(self, cache_dir = "../denoise_cache", name_prefix = "denoise")
        self.stacked_units = []
        self.graph = None
        self.cache_dir = cache_dir
        self.name_prefix = name_prefix

    def _sample_config(self, neurons_range, dropout_range):
        """
        Sample number of neurons and dropout rate
        
        Params:
        - neurons_range: tuple (min_neurons, max_neurons) specifying the min and max number of neurons of the hidden layer
        - dropout_range: tuple (lower, upper) specifying min and max dropout probability

        Return: n_neurons, dropout_rate
        """
        min_neurons, max_neurons = neurons_range
        min_dropout, max_dropout = dropout_range
        n_neurons = np.random.randint(min_neurons, max_neurons + 1)
        dropout_rate = np.random.random_sample() * (max_dropout - min_dropout) + min_dropout
        return n_neurons, dropout_rate
        
    def random_search_unit(self, X, n_folds, neurons_range = None, dropout_range = None,
                           n_trials = 10, n_epochs = 100, batch_size = 256, seed = 42):
        """
        Find the best unit autoencoder given an input set

        Params:
        - X: the input training set
        - n_folds: number of folds for cross-validation
        - neurons_range: tuple (min_neurons, max_neurons) specifying the min and max number of neurons of the hidden layer
        - dropout_range: tuple (lower, upper) specifying min and max dropout probability
        - n_trials: number of random configurations to try

        Return: an UnitAutoencoder with all the params trained; this unit autoencoder is later on
        supposed to be used to build the denoising autoencoder
        """
        n_inputs = X.shape[1]
        best_unit = None
        best_reconstruction_loss = -1
        for trial in range(n_trials):
            n_neurons, dropout_rate = self._sample_config(neurons_range, dropout_range)
            name = self.name_prefix + "_unit_trial" + str(trial)
            unit = UnitAutoencoder(name, n_inputs, n_neurons, dropout_rate)
            gen = TrainValidationGenerator(X, n_folds)
            reconstruction_loss = 0
            for fold_idx in range(n_folds):
                train, val = gen.next_train_validation_sets()
                unit.fit(train, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
                reconstruction_loss += unit.reconstruction_loss(val)
            reconstruction_loss /= n_folds
            if reconstruction_loss < best_reconstruction_loss:
                best_unit = unit
                best_reconstruction_loss = reconstruction_loss
        return best_unit

    def stack_unit(self, unit):
        self.stack_units.append(unit)

    def construct_graph(self):
        assert(len(self.stacked_units) > 0), "Invalid stack of unit autoencoders"
        self.graph = tf.Graph()
        with self.graph.as_default():
            n_inputs = self.stack_units[0].n_inputs
            layer_outputs = tf.placeholder(tf.float32, shape=[None, n_inputs], name = "inputs")
            for layer_idx in range(len(self.stacked_units)):
                params = self.stack_units[layer_idx].params
                hidden_W, hidden_b = params["hidden/kernel:0"], params["hidden/bias:0"]
                layer_outputs = self.activation(tf.matmul(layer_outputs, hidden_W) + hidden_b)
        
