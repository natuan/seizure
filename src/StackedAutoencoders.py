import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tqdm import *
from datetime import datetime
import matplotlib.pyplot as plt

from Utils import timestr

class UnitAutoencoder:
    def __init__(self,
                 name,
                 n_inputs,
                 n_neurons,
                 noise_stddev = None,
                 dropout_rate = None,
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
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.noise_stddev = noise_stddev
        self.dropout_rate = dropout_rate
        self.hidden_activation = hidden_activation
        self.regularizer = regularizer
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
            self.training = tf.placeholder_with_default(False, shape=(), name='training')
            if (noise_stddev is not None):
                X_noisy = tf.cond(self.training,
                                  lambda: self.X + tf.random_normal(tf.shape(self.X), stddev = noise_stddev),
                                  lambda: self.X)
            elif (dropout_rate is not None):
                X_noisy = tf.layers.dropout(self.X, dropout_rate, training=self.training)
            else:
                X_noisy = self.X
            self.hidden = tf.layers.dense(X_noisy, n_neurons, activation=hidden_activation, kernel_regularizer = regularizer, name="{}_hidden".format(self.name))
            self.outputs = tf.layers.dense(self.hidden, n_inputs, activation=output_activation, kernel_regularizer = regularizer, name="{}_outputs".format(self.name))
            self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)                
            self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
            self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)
            self.training_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
            loss_str = "Reconstruction_and_regularizer loss" if regularizer else "Reconstruction_loss"
            self.loss_summary = tf.summary.scalar(loss_str, self.loss)

            # Ops to summarize variables
            trainable_vars = dict([(var.name, var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            hidden_weights = trainable_vars["{}_hidden/kernel:0".format(self.name)]
            assert(hidden_weights.shape == (n_inputs, n_neurons)), "Invalid hidden weight shape"
            hidden_biases = trainable_vars["{}_hidden/bias:0".format(self.name)]
            import pdb
            pdb.set_trace()
            for neuron_idx in range(n_neurons):
                self._variable_summaries(hidden_weights[neuron_idx], "weights_hidden_neuron_{}".format(neuron_idx))
                self._variable_summaries(hidden_weights[neuron_idx], "bias_hidden_neuron_{}".format(neuron_idx))
                tf.summary.histogram("activation_hidden_neuron_{}".format(neuron_idx), self.hidden[neuron_idx])
            
            self.summary = tf.summary.merge_all()
            tf_log_dir = "{}/{}_run-{}".format(tf_log_dir, self.name, timestr())
            self.train_file_writer = tf.summary.FileWriter(tf_log_dir, self.graph)
            
        # Dictionary of trainable parameters: key = variable name, values are their values (after training or
        # restored from a model)
        self.params = None        

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
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.training: True})                    
                    step = epoch * n_batches + batch_idx
                    if step % checkpoint_steps == 0:
                        train_summary = sess.run(self.summary, feed_dict={self.X: X_batch})
                        self.train_file_writer.add_summary(train_summary, step)
                    start_idx += batch_size
                # The remaining (less than batch_size) samples
                indices = X_train_indices[start_idx : len(X_train)]
                if len(indices) > 0:
                    X_batch = X_train[indices]
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.training: True})                
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.train_file_writer.close()
            print("Saving model to {}...".format(model_path))
            self.saver.save(sess, model_path)
            print(">> Done")
            return sess.run(self.loss, feed_dict={self.X: X_train})

    def restore(self, model_path):
        print("Restoring model from {}...".format(model_path))
        if self.params is not None:
            print(">> Warning: self.params not empty and will be replaced")
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, model_path)
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        print(">> Done")

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
        

class StackedAutoencoders:
    def __init__(self,
                 name,
                 cache_dir = "../cache",
                 tf_log_dir = "../tf_logs"):
        self.name = name
        self.stacked_units = []
        self.graph = None
        self.params = None
        self.cache_dir = cache_dir
        self.tf_log_dir = tf_log_dir

        self.X = None
        self.training = None
        self.loss = None
        self.hidden = []
        self.outputs = None
        self.training_op = None
        self.saver = None
        self.loss_summary = None
        self.summary = None
        self.train_file_writer = None        

    def _check_unit_validity(self, unit):
        """
        Check if a unit is suitable to be stacked up

        Params:
        - unit: a unit that is about to be stacked up

        Return: true if this unit's name has not been used by any units on the stack, and 
        its inputs shape fits with the top one on the current stack; false otherwise.
        """
        n_units = len(self.stacked_units)
        if n_units > 0:
            top_unit = self.stacked_units[n_units - 1]
            if top_unit.n_neurons != unit.n_inputs:
                return False
            for u in self.stacked_units:
                if u.name == unit.name:
                    return False
        return True

    def _add_hidden_layer(self, input_tensor, unit, layer_name):
        assert(unit.params), "Invalid unit.params"
        with tf.name_scope(layer_name):           
            if (unit.dropout_rate is not None):
                input_noisy = tf.layers.dropout(input_tensor, unit.dropout_rate, training=self.training)
            else:
                # The Gaussian noise, if appliable during training of the units, is ignored in the stack
                input_noisy = input_tensor
                
            kernel_name = "{}_hidden/kernel:0".format(unit.name)
            bias_name = "{}_hidden/bias:0".format(unit.name)
            weights = tf.Variable(unit.params[kernel_name], name = "weights")
            assert(weights.shape == (unit.n_inputs, unit.n_neurons)), "Wrong assumption about weight's shape"
            biases = tf.Variable(unit.params[bias_name], name = "biases")
            assert(biases.shape == (unit.n_neurons,)), "Wrong assumption about bias's shape"
            pre_activations = tf.matmul(input_noisy, weights) + biases
            if unit.hidden_activation is not None:
                activations = unit.hidden_activation(pre_activations, name = "activations")
            else:
                activations = pre_activations
            reg_loss = unit.regularizer(weights) if unit.regularizer else None
            return activations, reg_loss          
    
    def stack_autoencoder(self, unit, layer_name):
        assert(self._check_unit_validity(unit)), "Invalid unit autoencoder"
        self.graph = tf.Graph() if self.graph is None else self.graph
        with self.graph.as_default():
            intput_tensor = None
            if not self.hidden:
                self.X = tf.placeholder(tf.float32, shape=[None, unit.n_inputs], name="X")
                self.training = tf.placeholder_with_default(False, shape=(), name="training")
                input_tensor = self.X
            else:
                input_tensor = self.hidden[-1]
            activations, reg_loss = self._add_hidden_layer(input_tensor, unit, layer_name)
            self.hidden += [activations]
            if reg_loss is not None:
                self.loss = tf.add_n([self.loss, reg_loss]) if self.loss is not None else reg_loss
        
    def stack_output_layer(self,
                           layer_name,
                           activation = None,
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
                pre_acts = tf.matmul(self.hidden[-1], weights) + biases
                self.outputs = activation(pre_acts, name="activations") if activation is not None else pre_acts

            self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
            self.loss = tf.add_n([self.loss, self.reconstruction_loss]) if self.loss is not None else self.reconstruction_loss
            self.loss = tf.add_n([self.loss, kernel_regularizer(weights)]) if kernel_regularizer is not None else self.loss
            
    def finalize(self, optimizer = tf.train.AdamOptimizer(0.001)):
        assert(self.graph), "Empty graph"
        with self.graph.as_default():
            if optimizer is not None:
                self.training_op = optimizer.minimize(self.loss)
                self.loss_summary = tf.summary.scalar("Loss", self.loss)
                self.summary = tf.summary.merge_all()
                tf_log_dir = "{}/{}_run-{}".format(self.tf_log_dir, self.name, timestr())
                self.train_file_writer = tf.summary.FileWriter(tf_log_dir, self.graph)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def save(self, model_path):
        with tf.Session(graph=self.graph) as sess:
            self.init.run()
            print("Saving model to {}...".format(model_path))
            self.saver.save(sess, model_path)
            print(">> Done")       
            
    def fit(self, X_train, model_path, n_epochs, batch_size = 256, checkpoint_steps = 100, seed = 42, tfdebug = False):
        assert(self.training_op is not None), "Invalid self.training_op"
        assert(self.X.shape[1] == X_train.shape[1]), "Invalid input shape"
        with tf.Session(graph=self.graph) as sess:
            if tfdebug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            tf.set_random_seed(seed)
            self.init.run()
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
                    start_idx += batch_size
                # The remaining (less than batch_size) samples
                indices = X_train_indices[start_idx : len(X_train)]
                if len(indices) > 0:
                    X_batch = X_train[indices]
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.training: True})
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            self.train_file_writer.close()
            print("Saving model to {}...".format(model_path))
            self.saver.save(sess, model_path)
            print(">> Done")
            return sess.run(self.loss, feed_dict={self.X: X_train})

    def restore(self, model_path):
        print("Restoring model from {}...".format(model_path))
        if self.params is not None:
            print(">> Warning: self.params not empty and will be replaced")
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, model_path)
            self.params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        print(">> Done")
        
    def restore_and_eval(self, model_path, X = None, varlist = [], tfdebug = False):
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
            if not varlist:
                return []
            assert(X is not None), "Invalid input samples"
            varmap = {"loss": self.loss,
                      "reconstruction_loss": self.reconstruction_loss,
                      "hidden_outputs": self.hidden[-1],
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
        
def stacked_autoencoders(name,
                         units,
                         hidden_layer_names = None,
                         include_output_layer = True,
                         output_layer_name = None,
                         output_activation = None,
                         output_kernel_regularizer = None,
                         output_kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                         output_bias_initializer = tf.zeros_initializer(),
                         optimizer = tf.train.AdamOptimizer(0.001),
                         cache_dir = "../cache",
                         tf_log_dir = "../tf_log",):
    """
    Building a stacked autoencoder

    Params:
    - name: name of the stacked autoencoder
    - units: list of unit autoencoders
    """
    encoder = StackedAutoencoders(name=name, cache_dir=cache_dir, tf_log_dir=tf_log_dir)
    if not hidden_layer_names:
        hidden_layer_names = ["{}_hidden_{}".format(name, str(idx)) for idx in range(len(units))]
    assert(len(hidden_layer_names) == len(units)), "Mismatch hidden layer names and units"
    for idx, unit in enumerate(units):
        encoder.stack_autoencoder(unit, hidden_layer_names[idx])
    if include_output_layer:
        output_layer_name = "{}_outputs".format(name) if output_layer_name is None else output_layer_name
        encoder.stack_output_layer(layer_name=output_layer_name,
                                   activation=output_activation,
                                   kernel_regularizer=output_kernel_regularizer,
                                   kernel_initializer=output_kernel_initializer,
                                   bias_initializer=output_bias_initializer)
    encoder.finalize(optimizer=optimizer)
    return encoder

def generate_unit_autoencoders(X_train,
                               y_train,
                               scaler,
                               n_neurons_range,
                               noise_stddev = None,
                               dropout_rate = None,
                               hidden_activation = tf.nn.softmax,
                               output_activation = None,
                               regularizer_value = 0.01,
                               initializer = tf.contrib.layers.variance_scaling_initializer(),
                               optimizer = tf.train.AdamOptimizer(0.001),
                               n_epochs = 5000,
                               batch_size = 256,
                               checkpoint_steps = 100,
                               seed = 0,                               
                               cache_dir = "../cache",
                               tf_log_dir = "../tf_logs"):
    n_inputs = X_train.shape[1]
    X_scaled = scaler.fit_transform(X_train)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if noise_stddev is not None:
        prefix = "denoise"
    elif dropout_rate is not None:
        prefix = "dropout"
    else:
        prefix = "ordinary"
    rows = {}
    for n_neurons in n_neurons_range:
        unit_name = config_str(prefix,
                               n_inputs,
                               n_neurons,
                               hidden_activation=hidden_activation,
                               regularizer_value=regularizer_value,
                               noise_stddev=noise_stddev,
                               dropout_rate=dropout_rate)
        print("Constructing and training unit {}".format(unit_name))
        unit_cache_dir = os.path.join(cache_dir, unit_name)
        if not os.path.exists(unit_cache_dir):
            os.makedirs(unit_cache_dir)
        unit_tf_log_dir = tf_log_dir
        if not os.path.exists(unit_tf_log_dir):
            os.makedirs(unit_tf_log_dir)
        unit = UnitAutoencoder(unit_name,
                               n_inputs,
                               n_neurons,
                               noise_stddev=noise_stddev,
                               dropout_rate=dropout_rate,
                               hidden_activation=hidden_activation,
                               output_activation=output_activation,
                               regularizer=tf.contrib.layers.l2_regularizer(regularizer_value),
                               initializer=initializer,
                               optimizer=optimizer,                               
                               tf_log_dir=unit_tf_log_dir)
        unit_model_path = os.path.join(unit_cache_dir, "{}.model".format(unit_name))
        unit.fit(X_scaled,
                 n_epochs=n_epochs,
                 model_path=unit_model_path,
                 batch_size=batch_size,
                 checkpoint_steps=checkpoint_steps,
                 seed=seed)
        [reconstruction_loss, outputs] = unit.eval(X_scaled, ["reconstruction_loss", "outputs"])
        row = [reconstruction_loss, unit_name]
        rows[n_neurons] = row
        assert(outputs.shape == X_scaled.shape), "Invalid output shape"
        unit_plot_dir = os.path.join(unit_cache_dir, "plots")
        unit_output_dir = os.path.join(unit_plot_dir, "outputs")
        X_recon = scaler.inverse_transform(outputs)
        plot_reconstructed_outputs(X_train, y_train, X_recon, size_per_class=10, plot_dir_path=unit_plot_dir_path, seed=seed+10):
        
    columns = ["reconstruction_loss", "model_name"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "n_neurons"
    df.columns = columns
    result_file_path = os.path.join(cache_dir, "results.csv")
    if os.path.exists(result_file_path):
        existing_df = pd.DataFrame.from_csv(result_file_path, index_col = 0)
        df = df.append(existing_df)
    df.sort_index(inplace=True)
    df.to_csv(result_file_path)
