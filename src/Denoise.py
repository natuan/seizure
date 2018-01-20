import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import *
from datetime import datetime
import matplotlib.pyplot as plt

from Autoencoder import *

# Create a time string
def timestr():
    today = datetime.today()
    month_dict = {1:"jan", 2:"feb", 3:"mar", 4:"apr", 5:"may", 6:"jun", 7:"jul", 8:"aug", 9:"sep", 10:"oct", 11:"nov", 12:"dec"}
    return "{}{}_{}:{}".format(month_dict[today.month], today.day, today.hour, today.minute)

class UnitAutoencoder:
    def __init__(self,
                 name,
                 n_inputs,
                 n_neurons,
                 noise_stddev = None,
                 dropout_rate = None,
                 tied_weights = False,
                 hidden_activation = tf.nn.softmax,
                 output_activation = None,
                 regularizer = tf.contrib.layers.l2_regularizer(0.01),
                 initializer = tf.contrib.layers.variance_scaling_initializer(),
                 optimizer = tf.train.AdamOptimizer(0.001),
                 tf_log_dir = "../tf_logs"):
        """
        Create an autoencoder that has one hidden layer of neurons
        
        Params:
        - n_inputs: number of inputs; also the number of neurons in the output layer
        - n_neurons: number of neurons in the hidden layer
        - noise_stddev: standard deviation of the Gaussian noise; not used if None
        - dropout_rate: if specified a Dropout layer will be added after the input layer; not used if None
        - regularizer: kernel regularizers for hidden and output layers
        Return: None
        """
        self.name = name
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
            self.training = tf.placeholder_with_default(False, shape=(), name='training')
            if (noise_stddev is not None):
                X_noisy = self.X + tf.random_normal(tf.shape(self.X), stddev = noise_stddev)
            elif (dropout_rate is not None):
                X_noisy = tf.layers.dropout(self.X, dropout_rate, training=self.training)
            else:
                X_noisy = self.X

            if (tied_weights):
                hidden_weights_init = initializer([n_inputs, n_neurons])
                hidden_weights = tf.Variable(hidden_weights_init, dtype=tf.float32, name="{}_hidden_weights".format(self.name))
                hidden_biases = tf.Variable(tf.zeros(n_neurons), name = "{}_hidden_biases".format(self.name))
                assert(hidden_activation), "Invalid hidden activation function"
                self.hidden = hidden_activation(tf.matmul(X_noisy, hidden_weights) + hidden_biases)
                output_weights = tf.transpose(hidden_weights, name = "{}_output_weights".format(self.name))
                output_biases = tf.Variable(tf.zeros(n_inputs), name = "{}_output_biases".format(self.name))
                output_linear = tf.matmul(self.hidden, output_weights) + output_biases
                self.outputs = output_activation(output_linear) if output_activation else output_linear
                self.reg_losses = regularizer(hidden_weights) if regularizer else []
            else:
                self.hidden = tf.layers.dense(X_noisy, n_neurons, activation=hidden_activation, kernel_regularizer = regularizer, name="{}_hidden".format(self.name))
                self.outputs = tf.layers.dense(self.hidden, n_inputs, activation=output_activation, kernel_regularizer = regularizer, name="{}_outputs".format(self.name))
                self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)                
                
            self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
            self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)
            self.training_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            loss_str = "Reconstruction and regularizer loss" if regularizer else "Reconstruction loss"
            self.loss_summary = tf.summary.scalar(loss_str, self.loss)
            tf_log_dir = "{}/run-{}".format(tf_log_dir, timestr())
            self.tf_log_file_writer = tf.summary.FileWriter(tf_log_dir, self.graph)
        # Dictionary of trainable parameters: key = variable name, values are their values (after training or
        # restored from a model)
        self.params = None        
        
    def fit(self, X_train, n_epochs, model_path, batch_size = 256, checkpoint_steps = 100, seed = 42, tfdebug = False):
        """
        Train the unit autoencoder against a training set

        Params:
        - X_train: the training set of shape (n_samples, n_features)
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
            for epoch in tqdm(range(n_epochs)):
                X_train_indices = np.random.permutation(len(X_train))
                n_batches = len(X_train) // batch_size
                start_idx = 0
                for batch_idx in range(n_batches):
                    indices = X_train_indices[start_idx : start_idx + batch_size]
                    X_batch = X_train[indices]
                    step = epoch * n_batches + batch_idx
                    if step % checkpoint_steps == 0:
                        summary_str = self.loss_summary.eval(feed_dict={self.X: X_batch})
                        self.tf_log_file_writer.add_summary(summary_str, step)
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.training: True})
                    start_idx += batch_size
                # The remaining (less than batch_size) samples
                indices = X_train_indices[start_idx : len(X_train)]
                if len(indices) > 0:
                    X_batch = X_train[indices]
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.training: True})                
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.tf_log_file_writer.close()
            print("Saving model to {}...".format(model_path))
            self.saver.save(sess, model_path)
            print(">> Done")

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
            print("Restoring model from {}...".format(model_path))
            self.saver.restore(sess, model_path)
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            print(">> Done")
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

    def plot_hidden_neurons(self, plot_dir_path):
        assert(self.params), "Error: self.params empty"
        assert(plot_dir_path), "Invalid dir path"
        if not os.path.exists(plot_dir_path):
            os.makedirs(plot_dir_path)
        kernel_name = "{}_hidden/kernel:0".format(self.name)
        weights_list = self.params[kernel_name]
        n_rows = 10
        n_cols = len(weights_list) // n_rows
        fig = plt.figure()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 48))
        idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                ax.plot(weights_list[idx])
                idx += 1
        plot_file = os.path.join(plot_dir_path, "neurons.png")
        plt.savefig(plot_file)
            
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
        
