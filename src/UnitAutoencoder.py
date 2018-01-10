import os
import numpy as np
import tensorflow as tf

from Autoencoder import Autoencoder

class UnitAutoencoder(Autoencoder):
    def __init__(self, n_inputs, n_neurons, dropout_rate = None):
        """
        Create an autoencoder that has one hidden layer of neurons
        
        Params:
        - n_inputs: number of inputs; also the number of neurons in the output layer
        - n_neurons: number of neurons in the hidden layer
        - dropout_rate: if specified a Dropout layer will be added after the input layer
        Return: None
        """
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
            self.outputs = my_dense_layer(self.hidden, self.n_inputs, activation=self.output_activation, name="outputs")
            self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.X))
            self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)
            self.training_op = self.optimizer.minimize(loss) # DESIGN CHOICE: should regularization loss be included?
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
    def fit(self, X_train, n_epochs, batch_size, seed = 42):
        assert(self.n_inputs == X_train.shape[1]), "Invalid input shape"
        with tf.Session(graph=self.graph) as sess:
            tf.set_random_seed(seed)
            self.init.run()
            for epoch in tqdm(range(n_epochs)):
                X_train_indices = rnd.permutation(len(X_train))
                n_batches = len(X_train) // batch_size
                for iteration in range(n_batches):
                    indices = X_train_indices[:batch_size]
                    X_batch = X_train[indices]
                    sess.run(self.training_op, feed_dict={self.X: X_batch})
            params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            hidden_val = self.hidden.eval(feed_dict={self.X: X_train})
            return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]
