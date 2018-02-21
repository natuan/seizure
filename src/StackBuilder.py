import os
import tensorflow as tf
from StackedAutoencoders import *
from Utils import *

class StackBuilder:
    def __init__(self,
                 name,
                 n_inputs,
                 noise_stddev = 0.3,
                 n_hidden_layers = 3,
                 n_neurons_per_layer = 200,
                 unit_hidden_activations = tf.nn.softmax,
                 unit_output_activations = None,
                 output_activation = tf.nn.softmax,
                 output_kernel_regularizer = None,
                 output_kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                 output_bias_initializer = tf.zeros_initializer(),
                 adam_lr = 5*1e-6,
                 cache_dir = "../cache",
                 tf_logs = "../tf_logs"):
        self.name = name
        self.n_inputs
        self.noise_stddev = noise_stddev
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.unit_hidden_activations = unit_hidden_activations
        self.unit_output_activations = unit_output_activations
        self.include_output_layer = include_output_layer
        self.output_activation = output_activation
        self.output_kernel_regularizer = output_kernel_regularizer
        self.output_kernel_initializer = output_kernel_initializer
        self.output_bias_initializer = output_bias_initializer
        self.adam_lr = adam_lr 
        self.cache_dir = cache_dir
        self.tf_logs = tf_logs
        self.model_path = None
        self.stack = None
        
    def pretrain(self,
                 X_train,
                 X_valid,
                 scaler = MinMaxScaler(feature_range=(0, 1)),
                 n_observable_hidden_neurons_per_layer = 0,
                 n_epochs = 10000,
                 batch_size = 64,
                 checkpoint_steps = 1000,
                 seed = 42):
        assert(X_train.shape[1] == X_valid.shape[1]), "Invalid input shapes"
        units = []
        n_inputs = X_train.shape[1]
        X_input_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        for hidden_layer in range(n_hidden_layers):
            print("** Pretraining hidden layer {}/{}\n**".format(hidden_layer+1, n_hidden_layers))
            unit_name = "Unit_{}_of_{}".format(hidden_layer+1, n_hidden_layers)
            unit_cache_dir = os.path.join(self.cache_dir, unit_name)
            if not os.path.exists(unit_cache_dir):
                os.makedirs(unit_cache_dir)
            unit_tf_log_dir = os.path.join(self.tf_log_dir, unit_name)
            if not os.path.exists(unit_tf_log_dir):
                os.makedirs(unit_tf_log_dir)
            unit = UnitAutoencoder(unit_name,
                                   n_inputs,
                                   n_neurons_per_layer,
                                   noise_stddev = self.noise_stddev,
                                   hidden_activation = self.unit_hidden_activations,
                                   output_activation = self.unit_output_activations,
                                   n_observable_hidden_neurons = n_observable_hidden_neurons_per_layer,
                                   regularizer = None,
                                   optimizer = tf.train.AdamOptimizer(self.adam_lr),
                                   tf_log_dir = unit_tf_log_dir)
            unit_model_path = os.path.join(unit_cache_dir, "{}.model".format(unit_name))
            model_step = unit.fit(X_input_scaled,
                                  X_valid_scaled,
                                  n_epochs=n_epochs,
                                  model_path=unit_model_path,
                                  batch_size=batch_size,
                                  checkpoint_steps=checkpoint_steps,
                                  seed=seed)
            units += [unit]
            X_input_scaled = unit.restore_and_eval(X_input_scaled, unit_model_path, ["hidden_ouputs"])
            X_valid_scaled = unit.restore_and_eval(X_valid_scaled, unit_model_path, ["hidden_ouputs"])
        print("** Stacking up pretrained units **\n")
        stack_cache_dir = os.path.join(self.cache_dir, "stack")
        stack_tf_log_dir = os.path.join(self.tf_log_dir, "stack")
        self.stack = StackedAutoencoders(name=self.name, cache_dir=stack_cache_dir, tf_log_dir=stack_tf_log_dir)
        stack_hidden_layer_names = ["{}_hidden_{}".format(self.name, str(idx)) for idx in range(len(units))]
        for idx, unit in enumerate(units):
            self.stack.stack_autoencoder(unit, stack_hidden_layer_names[idx])
        self.stack.stack_output_layer(layer_name="{}_outputs".format(self.name),
                                      activation=self.output_activation,
                                      kernel_regularizer=self.output_kernel_regularizer,
                                      kernel_initializer=self.output_kernel_initializer,
                                      bias_initializer=self.output_bias_initializer)
        self.stack.finalize(optimizer=tf.train.AdamOptimizer(self.adam_lr))
        self.model_path = os.path.join(stack_cache_dir, self.name)
        self.stack.save(model_path)

    def encode(self, X):
        return self.stack.restore_and_eval(model_path=self.model_path, X=X, varlist=["hidden_outputs"])

    
