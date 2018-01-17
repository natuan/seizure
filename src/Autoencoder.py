import tensorflow as tf

class Autoencoder:
    def __init__(self,
                 hidden_activation = tf.nn.softmax,
                 output_activation = None,
                 regularizer = tf.contrib.layers.l2_regularizer(0.0005),
                 initializer = tf.contrib.layers.variance_scaling_initializer(),
                 optimizer = tf.train.AdamOptimizer(0.0001)):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.regularizer = regularizer
        self.initializer = initializer
        self.optimizer = optimizer
        
